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

import os
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

# Load environment
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

def start_chat_session(user_id: int, persona_id: int = None, context_summary: str = None,
                       incognito: bool = False, nsfw_mode: bool = False,
                       group_id: int = None) -> int:
    """
    Create a new chat session and return the session ID.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_sessions (user_id, persona_id, context_summary, incognito, nsfw_mode, group_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (user_id, persona_id, context_summary, incognito, nsfw_mode, group_id))
            session_id = cur.fetchone()[0]
        conn.commit()
        return session_id
    finally:
        conn.close()

def log_chat_message(
    session_id: int,
    user_id: int,
    role: str,
    content: str,
    embedding: list = None,
    sentiment: float = None,
    topics: list = None
) -> int:
    """
    Log a chat message in the database and return its message ID.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_messages (
                    session_id, user_id, role, content, embedding, sentiment, topics
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                session_id,
                user_id,
                role,
                content,
                embedding,
                sentiment,
                topics
            ))
            message_id = cur.fetchone()[0]
        conn.commit()
        return message_id
    finally:
        conn.close()

def get_chat_messages(session_id: int):
    """
    Retrieve all messages from a chat session, ordered by time.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, role, content, sentiment, topics
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY timestamp ASC;
            """, (session_id,))
            return cur.fetchall()
    finally:
        conn.close()

def get_last_session(user_id: int):
    """
    Get the most recent session ID for a user, if any.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM chat_sessions
                WHERE user_id = %s
                ORDER BY start_time DESC
                LIMIT 1;
            """, (user_id,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def get_last_session_for_persona(user_id: int, persona_id: int):
    """
    Get the most recent non-incognito session ID for a user+persona pair.
    Returns None if no matching session exists.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM chat_sessions
                WHERE user_id = %s AND persona_id = %s
                  AND (incognito IS NULL OR incognito = FALSE)
                ORDER BY start_time DESC
                LIMIT 1;
            """, (user_id, persona_id))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def archive_session(session_id: int) -> bool:
    """Archive a session (soft delete — hidden from UI but data preserved)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET archived = TRUE WHERE id = %s;",
                (session_id,),
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def unarchive_session(session_id: int) -> bool:
    """Restore an archived session."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET archived = FALSE WHERE id = %s;",
                (session_id,),
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# --- Session Groups ---

def create_session_group(user_id: int, persona_id: int, name: str) -> int:
    """Create a session group (project folder) and return its ID."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO session_groups (user_id, persona_id, name)
                VALUES (%s, %s, %s) RETURNING id;
            """, (user_id, persona_id, name))
            gid = cur.fetchone()[0]
        conn.commit()
        return gid
    finally:
        conn.close()


def list_session_groups(user_id: int, persona_id: int = None) -> list:
    """List session groups for a user, optionally filtered by persona."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if persona_id is not None:
                cur.execute("""
                    SELECT id, user_id, persona_id, name, created_at
                    FROM session_groups
                    WHERE user_id = %s AND persona_id = %s
                    ORDER BY created_at ASC;
                """, (user_id, persona_id))
            else:
                cur.execute("""
                    SELECT id, user_id, persona_id, name, created_at
                    FROM session_groups
                    WHERE user_id = %s
                    ORDER BY created_at ASC;
                """, (user_id,))
            return [
                {
                    "id": r[0], "user_id": r[1], "persona_id": r[2],
                    "name": r[3],
                    "created_at": r[4].isoformat() if r[4] else None,
                }
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


def rename_session_group(group_id: int, name: str) -> bool:
    """Rename a session group."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE session_groups SET name = %s WHERE id = %s;",
                (name, group_id),
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_session_group(group_id: int) -> bool:
    """Delete a session group. Sessions become ungrouped (not deleted)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # FK ON DELETE SET NULL handles ungrouping sessions
            cur.execute("DELETE FROM session_groups WHERE id = %s;", (group_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def move_session_to_group(session_id: int, group_id: int = None) -> bool:
    """Move a session into a group, or ungroup it (group_id=None)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET group_id = %s WHERE id = %s;",
                (group_id, session_id),
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_session_group_owner(group_id: int) -> int | None:
    """Return the user_id that owns a session group, or None."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM session_groups WHERE id = %s;", (group_id,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def list_sessions(user_id: int, limit: int = 50, include_archived: bool = False):
    """
    List sessions for a user with preview info.
    Archived sessions are hidden by default.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            archive_filter = "" if include_archived else "AND (s.archived IS NULL OR s.archived = FALSE)"
            cur.execute(f"""
                SELECT
                    s.id,
                    s.persona_id,
                    s.start_time,
                    COUNT(m.id) AS message_count,
                    (SELECT content FROM chat_messages
                     WHERE session_id = s.id AND role = 'user'
                     ORDER BY timestamp DESC LIMIT 1) AS last_user_msg,
                    MAX(m.timestamp) AS last_time,
                    s.incognito,
                    s.nsfw_mode,
                    p.slug,
                    p.name,
                    s.group_id,
                    s.archived
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON m.session_id = s.id
                LEFT JOIN user_personalities p ON p.id = s.persona_id
                WHERE s.user_id = %s {archive_filter}
                GROUP BY s.id, p.slug, p.name
                ORDER BY COALESCE(MAX(m.timestamp), s.start_time) DESC
                LIMIT %s;
            """, (user_id, limit))
            rows = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "persona_id": r[1],
                    "start_time": r[2].isoformat() if r[2] else None,
                    "message_count": r[3],
                    "last_user_msg": (r[4][:80] + "...") if r[4] and len(r[4]) > 80 else r[4],
                    "last_time": r[5].isoformat() if r[5] else None,
                    "incognito": r[6] or False,
                    "nsfw_mode": r[7] or False,
                    "persona_slug": r[8],
                    "persona_name": r[9],
                    "group_id": r[10],
                    "archived": r[11] or False,
                }
                for r in rows
            ]
    finally:
        conn.close()

