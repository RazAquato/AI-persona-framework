# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Echo Corpus Builder
-------------------
Collects a user's conversation history, facts, and topic interests
from the database into a structured corpus dict for trait extraction.

Privacy boundary: Echo only sees identity-tier facts. Emotional-tier
facts (negative people mentions, struggles) are excluded.

The corpus is the raw material from which Echo personality signals
are extracted. It stays in-memory — no files written to disk.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, ".."))
if MEMORY_PATH not in sys.path:
    sys.path.append(MEMORY_PATH)

from memory.chat_store import get_connection
from memory.fact_store import get_facts_by_tier
from memory.topic_graph import get_user_topics


def build_corpus(user_id: int, max_messages: int = 500) -> dict:
    """
    Collect all available user data into a corpus dict.

    Args:
        user_id: the user to build a corpus for
        max_messages: cap on how many messages to include (most recent first)

    Returns:
        dict with keys: messages, facts, topics, stats
    """
    messages = _get_user_messages(user_id, limit=max_messages)
    # Echo only sees identity-tier facts — emotional facts are private to chatbot
    facts = get_facts_by_tier(user_id, ["identity"])
    topics = get_user_topics(user_id, limit=50)

    # Compute basic stats
    user_msgs = [m for m in messages if m["role"] == "user"]
    avg_length = (
        sum(len(m["content"]) for m in user_msgs) / len(user_msgs)
        if user_msgs else 0
    )

    return {
        "user_id": user_id,
        "messages": messages,
        "facts": [{"text": f[1], "tier": f[4], "entity_type": f[5]} for f in facts],
        "topics": [{"topic": t["topic"], "weight": t["weight"]} for t in topics],
        "stats": {
            "total_messages": len(messages),
            "user_messages": len(user_msgs),
            "avg_message_length": round(avg_length, 1),
            "unique_sessions": len(set(m["session_id"] for m in messages)),
        },
    }


def _get_user_messages(user_id: int, limit: int = 500) -> list:
    """
    Get the user's most recent messages across all non-incognito sessions.
    Returns list of dicts: [{role, content, session_id}]
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT m.role, m.content, m.session_id
                FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE m.user_id = %s
                  AND (s.incognito IS NULL OR s.incognito = FALSE)
                ORDER BY m.timestamp DESC
                LIMIT %s;
            """, (user_id, limit))
            rows = cur.fetchall()
            return [
                {"role": r[0], "content": r[1], "session_id": r[2]}
                for r in reversed(rows)  # chronological order
            ]
    finally:
        conn.close()
