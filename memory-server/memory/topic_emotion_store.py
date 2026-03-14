# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Topic Emotion Store — User emotion classification per topic.

Tracks how the user feels about each topic (love, frustration, pride, etc.)
using the same 18-emotion vocabulary as the persona emotion system.

Emotions accumulate over time via running average — not replaced daily.
The nightly reflection job classifies emotions from recent messages and
merges them into existing state.
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

# Valid emotions (same vocabulary as persona emotion system)
VALID_EMOTIONS = {
    "love", "trust", "joy", "calm", "pride", "interest", "curiosity",
    "fear", "sadness", "anger", "disgust", "revulsion", "shame", "guilt",
    "hope", "jealousy", "dominance", "submission",
}

# Blending weight for new observations (running average)
# new_intensity = old * (1 - BLEND) + new * BLEND
BLEND_WEIGHT = 0.3


def get_connection():
    return psycopg2.connect(**PG_CONFIG)


def set_topic_emotion(user_id: int, topic_id: int, emotion: str,
                      intensity: float, source: str = "reflection") -> int:
    """
    Set or update a single emotion for a user-topic pair.
    Uses upsert — creates if new, blends if existing.

    Returns the row ID.
    """
    emotion = emotion.lower().strip()
    if emotion not in VALID_EMOTIONS:
        return None
    intensity = max(0.0, min(1.0, intensity))

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_topic_emotions
                    (user_id, topic_id, emotion, intensity, source, last_updated)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, topic_id, emotion)
                DO UPDATE SET
                    intensity = user_topic_emotions.intensity * (1 - %s) + %s * %s,
                    source = %s,
                    last_updated = CURRENT_TIMESTAMP
                RETURNING id;
            """, (
                user_id, topic_id, emotion, intensity, source,
                BLEND_WEIGHT, BLEND_WEIGHT, intensity,
                source,
            ))
            row = cur.fetchone()
        conn.commit()
        return row[0] if row else None
    finally:
        conn.close()


def set_topic_emotions(user_id: int, topic_id: int,
                       emotions: list, source: str = "reflection"):
    """
    Set multiple emotions for a user-topic pair at once.

    Args:
        emotions: list of {"emotion": str, "intensity": float}

    Returns list of row IDs.
    """
    ids = []
    for em in emotions:
        eid = set_topic_emotion(
            user_id, topic_id,
            em.get("emotion", ""), em.get("intensity", 0.0),
            source=source,
        )
        if eid:
            ids.append(eid)
    return ids


def get_topic_emotions(user_id: int, topic_id: int,
                       min_intensity: float = 0.0) -> list:
    """
    Get all emotions for a specific user-topic pair.

    Returns list of dicts: [{emotion, intensity, source, last_updated}]
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT emotion, intensity, source, last_updated
                FROM user_topic_emotions
                WHERE user_id = %s AND topic_id = %s AND intensity >= %s
                ORDER BY intensity DESC;
            """, (user_id, topic_id, min_intensity))
            return [
                {
                    "emotion": r[0], "intensity": r[1],
                    "source": r[2], "last_updated": r[3],
                }
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


def get_user_topic_emotions(user_id: int,
                            min_intensity: float = 0.1) -> list:
    """
    Get all topic-emotion pairs for a user above the intensity threshold.
    Joined with topic names for convenience.

    Returns list of dicts: [{topic_id, topic_name, emotion, intensity, last_updated}]
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ute.topic_id, t.name, ute.emotion, ute.intensity, ute.last_updated
                FROM user_topic_emotions ute
                JOIN topics t ON t.id = ute.topic_id
                WHERE ute.user_id = %s AND ute.intensity >= %s
                ORDER BY ute.intensity DESC;
            """, (user_id, min_intensity))
            return [
                {
                    "topic_id": r[0], "topic_name": r[1],
                    "emotion": r[2], "intensity": r[3],
                    "last_updated": r[4],
                }
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


def get_emotions_for_topics(user_id: int, topic_ids: list,
                            min_intensity: float = 0.1) -> dict:
    """
    Get emotions for a list of topic IDs in a single query.
    Returns dict keyed by topic_id: {topic_id: [{emotion, intensity}]}
    """
    if not topic_ids:
        return {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT topic_id, emotion, intensity
                FROM user_topic_emotions
                WHERE user_id = %s AND topic_id = ANY(%s) AND intensity >= %s
                ORDER BY topic_id, intensity DESC;
            """, (user_id, topic_ids, min_intensity))
            result = {}
            for r in cur.fetchall():
                tid = r[0]
                if tid not in result:
                    result[tid] = []
                result[tid].append({"emotion": r[1], "intensity": r[2]})
            return result
    finally:
        conn.close()


def delete_topic_emotion(user_id: int, topic_id: int, emotion: str):
    """Delete a specific emotion for a user-topic pair."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM user_topic_emotions
                WHERE user_id = %s AND topic_id = %s AND emotion = %s;
            """, (user_id, topic_id, emotion.lower().strip()))
        conn.commit()
    finally:
        conn.close()


def decay_topic_emotions(decay_rate: float = 0.95, min_intensity: float = 0.05):
    """
    Decay all topic emotion intensities. Called by nightly job.
    Removes emotions that fall below min_intensity.

    Returns (decayed_count, removed_count).
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Decay intensities
            cur.execute("""
                UPDATE user_topic_emotions
                SET intensity = intensity * %s
                WHERE intensity > %s;
            """, (decay_rate, min_intensity))
            decayed = cur.rowcount

            # Remove emotions that have faded to insignificance
            cur.execute("""
                DELETE FROM user_topic_emotions
                WHERE intensity < %s;
            """, (min_intensity,))
            removed = cur.rowcount

        conn.commit()
        return decayed, removed
    finally:
        conn.close()
