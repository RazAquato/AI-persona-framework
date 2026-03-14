# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Context Summary Compressor
--------------------------
Background service that scans chat_sessions where context_summary IS NULL,
pulls messages, sends them to the LLM for compression, and writes back
a compressed summary.

Usage:
    python3 compress_sessions.py              # process all unsummarized sessions
    python3 compress_sessions.py --limit 10   # process at most 10 sessions
    python3 compress_sessions.py --dry-run    # show what would be processed

Designed to run as a cron job. Requires llama-server to be running.

Two-tier recall pattern:
  1. Search summaries (fast, cheap) — "did we ever talk about X?"
  2. Dive into messages (only when summary matches) — pull actual conversation
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, ".."))
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared"))
LLM_CLIENT_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "LLM-client"))

for p in [MEMORY_PATH, SHARED_PATH, LLM_CLIENT_PATH]:
    if p not in sys.path:
        sys.path.append(p)

from dotenv import load_dotenv

SHARED_ENV = os.path.abspath(os.path.join(SHARED_PATH, "config", ".env"))
LOCAL_ENV = os.path.abspath(os.path.join(LLM_CLIENT_PATH, "config", ".env"))
load_dotenv(dotenv_path=SHARED_ENV)
load_dotenv(dotenv_path=LOCAL_ENV, override=True)

import psycopg2
from core.llm_client import call_llm

PG_CONFIG = {
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
    "database": os.getenv("PG_DATABASE"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
}

# Minimum messages to bother summarizing
MIN_MESSAGES = 4

COMPRESSION_PROMPT = """You are a memory compression assistant. Summarize the following conversation into a concise context summary. Focus on:
- Key facts learned about the user
- Decisions made or preferences stated
- Important topics discussed
- Emotional tone of the conversation

Be concise (2-5 sentences). Use third person ("The user..."). Do not include greetings or filler."""


def get_unsummarized_sessions(limit: int = 50) -> list:
    """Find sessions with enough messages but no context_summary."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT s.id, s.user_id, s.persona_id, s.start_time,
                       COUNT(m.id) AS msg_count
                FROM chat_sessions s
                JOIN chat_messages m ON m.session_id = s.id
                WHERE s.context_summary IS NULL
                  AND (s.incognito IS NULL OR s.incognito = FALSE)
                GROUP BY s.id
                HAVING COUNT(m.id) >= %s
                ORDER BY s.start_time ASC
                LIMIT %s;
            """, (MIN_MESSAGES, limit))
            return cur.fetchall()
    finally:
        conn.close()


def get_session_messages(session_id: int) -> list:
    """Get all messages for a session, ordered by time."""
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


def compress_conversation(messages: list) -> str:
    """Send conversation to LLM and get a compressed summary."""
    conversation_text = "\n".join(
        f"{role.upper()}: {content}" for role, content in messages
    )

    llm_messages = [
        {"role": "system", "content": COMPRESSION_PROMPT},
        {"role": "user", "content": conversation_text},
    ]

    response = call_llm(llm_messages)
    return response["content"].strip()


def save_summary(session_id: int, summary: str):
    """Write the context_summary back to the session row."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE chat_sessions SET context_summary = %s WHERE id = %s;
            """, (summary, session_id))
        conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Compress unsummarized chat sessions")
    parser.add_argument("--limit", type=int, default=50, help="Max sessions to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    args = parser.parse_args()

    sessions = get_unsummarized_sessions(limit=args.limit)

    if not sessions:
        print("[compress] No unsummarized sessions found.")
        return

    print(f"[compress] Found {len(sessions)} session(s) to process.")

    for session_id, user_id, persona_id, start_time, msg_count in sessions:
        print(f"  Session {session_id}: user={user_id}, persona={persona_id}, "
              f"msgs={msg_count}, started={start_time}")

        if args.dry_run:
            continue

        try:
            messages = get_session_messages(session_id)
            summary = compress_conversation(messages)
            save_summary(session_id, summary)
            print(f"    -> Saved summary ({len(summary)} chars)")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    if args.dry_run:
        print("[compress] Dry run — no changes made.")
    else:
        print(f"[compress] Done. Processed {len(sessions)} session(s).")


if __name__ == "__main__":
    main()
