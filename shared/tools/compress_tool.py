# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Compress Tool
-------------
Triggers context summary compression for a user's unsummarized sessions.
Callable as a tool from the LLM or as a /command from the user.

This is a thin wrapper that calls the compression logic from compress_sessions.py
but operates on a single user's sessions (not all users).
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MEMORY_PATH = os.path.join(ROOT, "memory-server")
LLM_CLIENT_PATH = os.path.join(ROOT, "LLM-client")
SHARED_PATH = os.path.join(ROOT, "shared")

for p in [MEMORY_PATH, SHARED_PATH, LLM_CLIENT_PATH]:
    if p not in sys.path:
        sys.path.append(p)

import psycopg2
from dotenv import load_dotenv

SHARED_ENV = os.path.abspath(os.path.join(SHARED_PATH, "config", ".env"))
LOCAL_ENV = os.path.abspath(os.path.join(LLM_CLIENT_PATH, "config", ".env"))
load_dotenv(dotenv_path=SHARED_ENV)
load_dotenv(dotenv_path=LOCAL_ENV, override=True)

PG_CONFIG = {
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
    "database": os.getenv("PG_DATABASE"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
}

MIN_MESSAGES = 4


def compress_sessions(user_id: int, **kwargs) -> dict:
    """
    Compress unsummarized sessions for a specific user.

    Args:
        user_id: the user whose sessions to compress
        **kwargs: additional args (user_permission injected by engine)

    Returns:
        dict with keys: success, sessions_found, sessions_compressed, errors
    """
    try:
        sessions = _get_user_unsummarized_sessions(user_id)
    except Exception as e:
        return {"success": False, "error": f"DB error: {e}"}

    if not sessions:
        return {
            "success": True,
            "sessions_found": 0,
            "sessions_compressed": 0,
            "message": "All sessions already have summaries.",
        }

    compressed = 0
    errors = []

    for session_id, msg_count in sessions:
        try:
            messages = _get_session_messages(session_id)
            summary = _compress_via_llm(messages)
            _save_summary(session_id, summary)
            compressed += 1
        except Exception as e:
            errors.append(f"Session {session_id}: {e}")

    return {
        "success": True,
        "sessions_found": len(sessions),
        "sessions_compressed": compressed,
        "errors": errors if errors else None,
    }


def _get_user_unsummarized_sessions(user_id: int) -> list:
    """Find sessions for this user that need compression."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT s.id, COUNT(m.id) AS msg_count
                FROM chat_sessions s
                JOIN chat_messages m ON m.session_id = s.id
                WHERE s.user_id = %s
                  AND s.context_summary IS NULL
                  AND (s.incognito IS NULL OR s.incognito = FALSE)
                GROUP BY s.id
                HAVING COUNT(m.id) >= %s
                ORDER BY s.start_time ASC
                LIMIT 20;
            """, (user_id, MIN_MESSAGES))
            return cur.fetchall()
    finally:
        conn.close()


def _get_session_messages(session_id: int) -> list:
    """Get messages for compression."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT role, content FROM chat_messages
                WHERE session_id = %s
                ORDER BY timestamp ASC;
            """, (session_id,))
            return cur.fetchall()
    finally:
        conn.close()


def _compress_via_llm(messages: list) -> str:
    """Send conversation to LLM for compression."""
    from core.llm_client import call_llm

    conversation_text = "\n".join(
        f"{role.upper()}: {content}" for role, content in messages
    )

    prompt = (
        "Summarize this conversation into a concise context summary (2-5 sentences). "
        "Focus on key facts, decisions, topics, and emotional tone. "
        "Use third person ('The user...'). No greetings or filler."
    )

    llm_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": conversation_text},
    ]

    response = call_llm(llm_messages)
    return response["content"].strip()


def _save_summary(session_id: int, summary: str):
    """Write summary back to session."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET context_summary = %s WHERE id = %s;",
                (summary, session_id),
            )
        conn.commit()
    finally:
        conn.close()
