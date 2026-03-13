# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Persona store — CRUD for user_personalities table.
Personas are owned by users, identified by stable numeric IDs.
Default personas are seeded from personality_config.json on user registration.
"""

import os
import json
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

PG_CONFIG = {
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
    "database": os.getenv("PG_DATABASE"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
}

# Path to the seed config file (default personas)
SEED_CONFIG_PATH = os.path.join(
    BASE_DIR, "..", "..", "LLM-client", "config", "personality_config.json"
)


def get_connection():
    return psycopg2.connect(**PG_CONFIG)


def get_persona(persona_id: int) -> dict | None:
    """Load a persona by its numeric ID. Returns dict or None."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, user_id, slug, name, description, system_prompt,
                       nsfw_capable, nsfw_prompt_addon, memory_scope, is_public
                FROM user_personalities WHERE id = %s;
            """, (persona_id,))
            row = cur.fetchone()
            if not row:
                return None
            return _row_to_dict(row)
    finally:
        conn.close()


def get_persona_by_slug(user_id: int, slug: str) -> dict | None:
    """Load a persona by user_id + slug. Returns dict or None."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, user_id, slug, name, description, system_prompt,
                       nsfw_capable, nsfw_prompt_addon, memory_scope, is_public
                FROM user_personalities
                WHERE user_id = %s AND slug = %s;
            """, (user_id, slug))
            row = cur.fetchone()
            if not row:
                return None
            return _row_to_dict(row)
    finally:
        conn.close()


def list_personas(user_id: int) -> list[dict]:
    """List all personas for a user."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, user_id, slug, name, description, system_prompt,
                       nsfw_capable, nsfw_prompt_addon, memory_scope, is_public
                FROM user_personalities
                WHERE user_id = %s
                ORDER BY created_at ASC;
            """, (user_id,))
            return [_row_to_dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def create_persona(user_id: int, slug: str, name: str, description: str = "",
                   system_prompt: str = "", nsfw_capable: bool = False,
                   nsfw_prompt_addon: str = None, memory_scope: dict = None) -> int:
    """Create a new persona for a user. Returns the persona ID."""
    if memory_scope is None:
        memory_scope = {"tier1": True, "tier2": "all", "tier3": "private"}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_personalities
                    (user_id, slug, name, description, system_prompt,
                     nsfw_capable, nsfw_prompt_addon, memory_scope)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (user_id, slug, name, description, system_prompt,
                  nsfw_capable, nsfw_prompt_addon, Json(memory_scope)))
            persona_id = cur.fetchone()[0]
        conn.commit()
        return persona_id
    finally:
        conn.close()


def update_persona(persona_id: int, **fields) -> bool:
    """Update persona fields. Only provided fields are changed. Returns True if found."""
    allowed = {"slug", "name", "description", "system_prompt",
               "nsfw_capable", "nsfw_prompt_addon", "memory_scope"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    # Wrap memory_scope in Json adapter
    if "memory_scope" in updates:
        updates["memory_scope"] = Json(updates["memory_scope"])
    set_clause = ", ".join(f"{k} = %s" for k in updates)
    values = list(updates.values()) + [persona_id]
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE user_personalities SET {set_clause} WHERE id = %s;",
                values,
            )
            updated = cur.rowcount > 0
        conn.commit()
        return updated
    finally:
        conn.close()


def delete_persona(persona_id: int) -> bool:
    """Delete a persona. Returns True if found."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_personalities WHERE id = %s;", (persona_id,))
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


def seed_default_personas(user_id: int) -> list[int]:
    """
    Seed default personas for a user from personality_config.json.
    Skips any slug that already exists for this user.
    Returns list of created persona IDs.
    """
    config_path = os.path.abspath(SEED_CONFIG_PATH)
    if not os.path.isfile(config_path):
        return []
    with open(config_path) as f:
        defaults = json.load(f)

    created = []
    for p in defaults:
        # Skip if this user already has this slug
        existing = get_persona_by_slug(user_id, p["slug"])
        if existing:
            continue
        pid = create_persona(
            user_id=user_id,
            slug=p["slug"],
            name=p["name"],
            description=p.get("description", ""),
            system_prompt=p.get("system_prompt", ""),
            nsfw_capable=p.get("nsfw_capable", False),
            nsfw_prompt_addon=p.get("nsfw_prompt_addon"),
            memory_scope=p.get("memory_scope"),
        )
        created.append(pid)
    return created


def _row_to_dict(row) -> dict:
    """Convert a DB row to a persona dict."""
    return {
        "id": row[0],
        "user_id": row[1],
        "slug": row[2],
        "name": row[3],
        "description": row[4],
        "system_prompt": row[5],
        "nsfw_capable": row[6] or False,
        "nsfw_prompt_addon": row[7],
        # Alias for prompt_builder compatibility
        "nsfw_system_prompt_addon": row[7],
        "memory_scope": row[8] or {"tier1": True, "tier2": "all", "tier3": "private"},
        "is_public": row[9] or False,
    }
