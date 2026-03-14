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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Fact Store
----------
CRUD for structured facts in PostgreSQL.

Facts table columns:
    id, user_id, text, source_chat_id (legacy), relevance_score, tags,
    source_type, source_ref, tier, entity_type, created_at

Tier values: 'identity', 'knowledge', 'relationship'
Source types: 'conversation', 'image_analysis', 'document', 'photo_archive', 'diary_entry'
Entity types: 'person', 'pet', 'place', 'event', 'thing', or NULL
"""

import os
import psycopg2
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


def get_connection():
    return psycopg2.connect(**PG_CONFIG)


def store_fact(user_id: int, fact: str, tags: list = None, relevance_score: float = None,
               source_chat_id: int = None, source_type: str = "conversation",
               source_ref: str = None, tier: str = "knowledge", entity_type: str = None):
    """
    Store a structured fact associated with a user.
    Checks for near-duplicate text before inserting.
    """
    # Dedup check: skip if very similar fact already exists for this user
    if _fact_exists(user_id, fact):
        return None

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO facts (user_id, text, tags, relevance_score, source_chat_id,
                                   source_type, source_ref, tier, entity_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (user_id, fact, tags, relevance_score, source_chat_id,
                  source_type, source_ref, tier, entity_type))
            fact_id = cur.fetchone()[0]
        conn.commit()
        return fact_id
    finally:
        conn.close()


def _fact_exists(user_id: int, fact_text: str) -> bool:
    """
    Check if a very similar fact already exists for this user.
    Uses case-insensitive exact match on text.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM facts
                WHERE user_id = %s AND LOWER(text) = LOWER(%s);
            """, (user_id, fact_text))
            return cur.fetchone()[0] > 0
    finally:
        conn.close()


def make_fact_blob(text: str, tier: str = "knowledge", entity_type: str = None,
                   confidence: float = 0.5, tags: list = None,
                   source_type: str = "conversation", source_ref: str = None) -> dict:
    """
    Factory function for creating fact dicts in the standard schema.
    Use this instead of constructing dicts manually.
    """
    return {
        "text": text,
        "tier": tier,
        "entity_type": entity_type,
        "confidence": confidence,
        "tags": tags or [],
        "source_type": source_type,
        "source_ref": source_ref,
    }


def store_fact_blobs(user_id: int, blobs: list, source_type: str = None,
                     source_ref: str = None) -> list:
    """
    Bulk-insert a list of fact dicts (from knowledge_extractor or compression pipeline).
    Deduplicates within the batch and against existing facts.
    Optional source_type/source_ref override applies to all blobs that don't have their own.

    Args:
        user_id: the user who owns these facts
        blobs: list of dicts with keys: text, tier, entity_type, confidence, tags, source_type, source_ref
        source_type: fallback source_type if blob doesn't specify one
        source_ref: fallback source_ref if blob doesn't specify one

    Returns:
        list of inserted fact IDs (None entries for deduped/skipped blobs)
    """
    if not blobs:
        return []

    conn = get_connection()
    try:
        ids = []
        seen_texts = set()
        with conn.cursor() as cur:
            for blob in blobs:
                text = blob.get("text", "").strip()
                if not text:
                    ids.append(None)
                    continue

                # Dedup within batch
                text_lower = text.lower()
                if text_lower in seen_texts:
                    ids.append(None)
                    continue
                seen_texts.add(text_lower)

                # Dedup against DB
                cur.execute(
                    "SELECT COUNT(*) FROM facts WHERE user_id = %s AND LOWER(text) = LOWER(%s);",
                    (user_id, text)
                )
                if cur.fetchone()[0] > 0:
                    ids.append(None)
                    continue

                blob_source_type = blob.get("source_type") or source_type or "conversation"
                blob_source_ref = blob.get("source_ref") or source_ref

                cur.execute("""
                    INSERT INTO facts (user_id, text, tags, relevance_score,
                                       source_type, source_ref, tier, entity_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    user_id, text,
                    blob.get("tags"),
                    blob.get("confidence"),
                    blob_source_type,
                    blob_source_ref,
                    blob.get("tier", "knowledge"),
                    blob.get("entity_type"),
                ))
                ids.append(cur.fetchone()[0])
        conn.commit()
        return ids
    finally:
        conn.close()


def get_facts(user_id: int, tier: str = None):
    """
    Retrieve facts for a given user, optionally filtered by tier.
    Returns list of tuples: (id, text, tags, relevance_score, tier, entity_type)
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if tier:
                cur.execute("""
                    SELECT id, text, tags, relevance_score, tier, entity_type FROM facts
                    WHERE user_id = %s AND tier = %s
                    ORDER BY relevance_score DESC NULLS LAST, id DESC;
                """, (user_id, tier))
            else:
                cur.execute("""
                    SELECT id, text, tags, relevance_score, tier, entity_type FROM facts
                    WHERE user_id = %s
                    ORDER BY relevance_score DESC NULLS LAST, id DESC;
                """, (user_id,))
            return cur.fetchall()
    finally:
        conn.close()


def get_facts_by_tier(user_id: int, tiers: list):
    """
    Retrieve facts filtered by multiple tiers.
    Used by context_builder for persona memory scope.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, text, tags, relevance_score, tier, entity_type FROM facts
                WHERE user_id = %s AND tier = ANY(%s)
                ORDER BY relevance_score DESC NULLS LAST, id DESC;
            """, (user_id, tiers))
            return cur.fetchall()
    finally:
        conn.close()


def get_facts_by_tag(user_id: int, tag: str):
    """Return facts for a user filtered by tag."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, text, tags, tier, entity_type FROM facts
                WHERE user_id = %s AND %s = ANY(tags);
            """, (user_id, tag))
            return cur.fetchall()
    finally:
        conn.close()


def get_facts_by_entity_type(user_id: int, entity_type: str):
    """Return facts filtered by entity type (person, pet, place, etc.)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, text, tags, relevance_score, tier FROM facts
                WHERE user_id = %s AND entity_type = %s
                ORDER BY id DESC;
            """, (user_id, entity_type))
            return cur.fetchall()
    finally:
        conn.close()


def delete_fact(fact_id: int):
    """Delete a specific fact by its ID."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM facts WHERE id = %s;", (fact_id,))
        conn.commit()
    finally:
        conn.close()


def get_top_facts(user_id: int, limit: int = 5, sort_by: str = 'relevance_score'):
    """Return top-N facts sorted by relevance or ID."""
    if sort_by not in {"relevance_score", "id"}:
        sort_by = "relevance_score"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if sort_by == "relevance_score":
                cur.execute("""
                    SELECT id, text, relevance_score FROM facts
                    WHERE user_id = %s AND relevance_score IS NOT NULL
                    ORDER BY relevance_score DESC
                    LIMIT %s;
                """, (user_id, limit))
            else:
                cur.execute("""
                    SELECT id, text, relevance_score FROM facts
                    WHERE user_id = %s
                    ORDER BY id DESC
                    LIMIT %s;
                """, (user_id, limit))
            return cur.fetchall()
    finally:
        conn.close()


def update_fact(fact_id: int, new_text: str = None, tags: list = None,
                relevance_score: float = None, tier: str = None, entity_type: str = None):
    """Update fields of a fact."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            updates = []
            values = []

            if new_text:
                updates.append("text = %s")
                values.append(new_text)
            if tags is not None:
                updates.append("tags = %s")
                values.append(tags)
            if relevance_score is not None:
                updates.append("relevance_score = %s")
                values.append(relevance_score)
            if tier is not None:
                updates.append("tier = %s")
                values.append(tier)
            if entity_type is not None:
                updates.append("entity_type = %s")
                values.append(entity_type)

            if not updates:
                return

            values.append(fact_id)
            query = f"UPDATE facts SET {', '.join(updates)} WHERE id = %s;"
            cur.execute(query, tuple(values))
        conn.commit()
    finally:
        conn.close()
