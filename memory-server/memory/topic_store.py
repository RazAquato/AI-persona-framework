# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Topic Store — Topic registry, salience tracking, and fact-topic linking.

Topics are first-class entities with per-persona salience that rises with
conversation and decays over time. Facts are linked to topics via their tags.
"""

import os
import math
from datetime import datetime, timedelta
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

# Family-related topic names get a decay floor so they never fully fade
STICKY_TOPICS = {
    "family", "kids", "children", "wife", "husband", "son", "daughter",
    "brother", "sister", "mother", "father", "parent",
}
STICKY_DECAY_FLOOR = 0.15

DECAY_RATE = 0.97  # halves roughly every 23 days
SALIENCE_BUMP = 0.1  # per mention


def get_connection():
    return psycopg2.connect(**PG_CONFIG)


# --- Topic CRUD ---

def get_or_create_topic(user_id: int, name: str) -> int:
    """Get existing topic ID or create a new one. Name is normalized to lowercase."""
    name = name.lower().strip()
    if not name:
        return None
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM topics WHERE user_id = %s AND name = %s;",
                (user_id, name),
            )
            row = cur.fetchone()
            if row:
                return row[0]
            cur.execute(
                "INSERT INTO topics (user_id, name) VALUES (%s, %s) RETURNING id;",
                (user_id, name),
            )
            tid = cur.fetchone()[0]
        conn.commit()
        return tid
    finally:
        conn.close()


def get_topic(user_id: int, name: str) -> dict | None:
    """Get a topic by name, or None if not found."""
    name = name.lower().strip()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id, name, created_at FROM topics WHERE user_id = %s AND name = %s;",
                (user_id, name),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"id": row[0], "user_id": row[1], "name": row[2],
                    "created_at": row[3].isoformat() if row[3] else None}
    finally:
        conn.close()


def list_topics(user_id: int) -> list:
    """List all topics for a user."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, created_at FROM topics WHERE user_id = %s ORDER BY name;",
                (user_id,),
            )
            return [
                {"id": r[0], "name": r[1],
                 "created_at": r[2].isoformat() if r[2] else None}
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


# --- Fact-Topic linking ---

def link_fact_to_topics(user_id: int, fact_id: int, tags: list):
    """
    Link a fact to topics based on its tags.
    Creates topics if they don't exist. Skips empty/None tags.
    """
    if not tags:
        return
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for tag in tags:
                if not tag or not tag.strip():
                    continue
                tag_name = tag.lower().strip()
                # Get or create topic
                cur.execute(
                    "SELECT id FROM topics WHERE user_id = %s AND name = %s;",
                    (user_id, tag_name),
                )
                row = cur.fetchone()
                if row:
                    topic_id = row[0]
                else:
                    cur.execute(
                        "INSERT INTO topics (user_id, name) VALUES (%s, %s) RETURNING id;",
                        (user_id, tag_name),
                    )
                    topic_id = cur.fetchone()[0]
                # Link fact to topic (ignore duplicates)
                cur.execute("""
                    INSERT INTO fact_topics (fact_id, topic_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING;
                """, (fact_id, topic_id))
        conn.commit()
    finally:
        conn.close()


def get_fact_topic_ids(fact_id: int) -> list:
    """Get topic IDs linked to a fact."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT topic_id FROM fact_topics WHERE fact_id = %s;",
                (fact_id,),
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


# --- Salience ---

def get_salience(user_id: int, persona_id: int, topic_id: int) -> dict | None:
    """Get salience record for a user+persona+topic, or None."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT salience, mention_count, last_mentioned, decay_floor
                FROM topic_salience
                WHERE user_id = %s AND persona_id = %s AND topic_id = %s;
            """, (user_id, persona_id, topic_id))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "salience": row[0],
                "mention_count": row[1],
                "last_mentioned": row[2],
                "decay_floor": row[3],
            }
    finally:
        conn.close()


def bump_salience(user_id: int, persona_id: int, topic_names: list):
    """
    Bump salience for a list of topic names mentioned in a conversation turn.
    Creates topics and salience records as needed.

    Formula: salience = min(1.0, salience + 0.1 * (1 - salience))
    This gives diminishing returns — converges toward 1.0.
    """
    if not topic_names:
        return
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for name in topic_names:
                name = name.lower().strip()
                if not name:
                    continue

                # Get or create topic
                cur.execute(
                    "SELECT id FROM topics WHERE user_id = %s AND name = %s;",
                    (user_id, name),
                )
                row = cur.fetchone()
                if row:
                    topic_id = row[0]
                else:
                    cur.execute(
                        "INSERT INTO topics (user_id, name) VALUES (%s, %s) RETURNING id;",
                        (user_id, name),
                    )
                    topic_id = cur.fetchone()[0]

                # Determine decay floor
                floor = STICKY_DECAY_FLOOR if name in STICKY_TOPICS else 0.0

                # Upsert salience
                cur.execute("""
                    INSERT INTO topic_salience
                        (user_id, persona_id, topic_id, salience, mention_count, last_mentioned, decay_floor)
                    VALUES (%s, %s, %s, %s, 1, CURRENT_TIMESTAMP, %s)
                    ON CONFLICT (user_id, persona_id, topic_id)
                    DO UPDATE SET
                        salience = LEAST(1.0, topic_salience.salience + %s * (1.0 - topic_salience.salience)),
                        mention_count = topic_salience.mention_count + 1,
                        last_mentioned = CURRENT_TIMESTAMP,
                        decay_floor = GREATEST(topic_salience.decay_floor, %s);
                """, (
                    user_id, persona_id, topic_id,
                    min(1.0, SALIENCE_BUMP),  # initial salience for new record
                    floor,
                    SALIENCE_BUMP,
                    floor,
                ))
        conn.commit()
    finally:
        conn.close()


def get_persona_salience(user_id: int, persona_id: int,
                         min_salience: float = 0.0) -> list:
    """
    Get all topic salience records for a user+persona, optionally filtered
    by minimum salience threshold.

    Returns list of dicts: [{topic_id, topic_name, salience, mention_count, last_mentioned, decay_floor}]
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts.topic_id, t.name, ts.salience, ts.mention_count,
                       ts.last_mentioned, ts.decay_floor
                FROM topic_salience ts
                JOIN topics t ON t.id = ts.topic_id
                WHERE ts.user_id = %s AND ts.persona_id = %s
                  AND ts.salience >= %s
                ORDER BY ts.salience DESC;
            """, (user_id, persona_id, min_salience))
            return [
                {
                    "topic_id": r[0], "topic_name": r[1], "salience": r[2],
                    "mention_count": r[3], "last_mentioned": r[4],
                    "decay_floor": r[5],
                }
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


def decay_all_salience(decay_rate: float = DECAY_RATE):
    """
    Apply time-based decay to all salience records.
    Formula: salience *= decay_rate ^ days_since_last_mention
    Never drops below decay_floor.

    Returns number of records updated.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Calculate decay for each record based on days since last mention
            cur.execute("""
                UPDATE topic_salience
                SET salience = GREATEST(
                    decay_floor,
                    salience * POWER(%s, EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_mentioned)) / 86400.0)
                )
                WHERE salience > decay_floor;
            """, (decay_rate,))
            updated = cur.rowcount
        conn.commit()
        return updated
    finally:
        conn.close()


def boost_group_salience(user_id: int, persona_id: int, group_id: int,
                         boost: float = 0.05):
    """
    Give a small salience boost to all topics mentioned in sessions belonging
    to a session group. Called when resuming a session in a group.

    Finds all topics mentioned across the group's sessions and gives them
    a small bump.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Find all topics tagged in messages from sessions in this group
            cur.execute("""
                SELECT DISTINCT unnest(m.topics) AS topic_name
                FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE s.group_id = %s AND s.user_id = %s
                  AND m.topics IS NOT NULL;
            """, (group_id, user_id))
            topic_names = [r[0] for r in cur.fetchall()]
        conn.commit()

        if topic_names:
            # Reuse bump_salience with a smaller boost
            old_bump = SALIENCE_BUMP
            # We'll do a direct upsert with a smaller amount
            with conn.cursor() as cur:
                for name in topic_names:
                    name = name.lower().strip()
                    if not name:
                        continue
                    cur.execute(
                        "SELECT id FROM topics WHERE user_id = %s AND name = %s;",
                        (user_id, name),
                    )
                    row = cur.fetchone()
                    if row:
                        topic_id = row[0]
                    else:
                        cur.execute(
                            "INSERT INTO topics (user_id, name) VALUES (%s, %s) RETURNING id;",
                            (user_id, name),
                        )
                        topic_id = cur.fetchone()[0]
                    floor = STICKY_DECAY_FLOOR if name in STICKY_TOPICS else 0.0
                    cur.execute("""
                        INSERT INTO topic_salience
                            (user_id, persona_id, topic_id, salience, mention_count, last_mentioned, decay_floor)
                        VALUES (%s, %s, %s, %s, 0, CURRENT_TIMESTAMP, %s)
                        ON CONFLICT (user_id, persona_id, topic_id)
                        DO UPDATE SET
                            salience = LEAST(1.0, topic_salience.salience + %s * (1.0 - topic_salience.salience));
                    """, (
                        user_id, persona_id, topic_id,
                        min(1.0, boost),
                        floor,
                        boost,
                    ))
            conn.commit()
            return len(topic_names)
        return 0
    finally:
        conn.close()


# --- Salience-filtered fact retrieval ---

def get_salient_fact_ids(user_id: int, persona_id: int,
                         min_salience: float = 0.2) -> set:
    """
    Get IDs of facts whose linked topics have salience above the threshold
    for the given persona. Facts with no linked topics are NOT included
    (they're handled separately as always-visible).

    Returns a set of fact IDs.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ft.fact_id
                FROM fact_topics ft
                JOIN topic_salience ts ON ts.topic_id = ft.topic_id
                WHERE ts.user_id = %s AND ts.persona_id = %s
                  AND ts.salience >= %s;
            """, (user_id, persona_id, min_salience))
            return {r[0] for r in cur.fetchall()}
    finally:
        conn.close()


def get_fact_max_salience(user_id: int, persona_id: int, fact_id: int) -> float:
    """Get the maximum salience of any topic linked to a fact. Returns 0.0 if none."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(MAX(ts.salience), 0.0)
                FROM fact_topics ft
                JOIN topic_salience ts ON ts.topic_id = ft.topic_id
                WHERE ft.fact_id = %s AND ts.user_id = %s AND ts.persona_id = %s;
            """, (fact_id, user_id, persona_id))
            return cur.fetchone()[0]
    finally:
        conn.close()


def has_linked_topics(fact_id: int) -> bool:
    """Check if a fact has any linked topics."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM fact_topics WHERE fact_id = %s LIMIT 1;",
                (fact_id,),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()
