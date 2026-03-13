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
Persona Emotion Store
---------------------
CRUD operations for the persona's emotional state toward a specific user.
Uses `emotional_relationships` table keyed by (user_id, persona_id).
Also logs state changes to `emotion_history` for long-term tracking.
"""

import os
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
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

# Default baseline emotions — the "neutral resting state" of any persona
DEFAULT_EMOTIONS = {
    "love": 0.05,
    "trust": 0.1,
    "joy": 0.1,
    "calm": 0.3,
    "pride": 0.05,
    "interest": 0.2,
    "curiosity": 0.2,
    "fear": 0.0,
    "sadness": 0.0,
    "anger": 0.0,
    "disgust": 0.0,
    "revulsion": 0.0,
    "shame": 0.0,
    "guilt": 0.0,
    "hope": 0.15,
    "jealousy": 0.0,
    "dominance": 0.1,
    "submission": 0.1,
}


def get_connection():
    return psycopg2.connect(**PG_CONFIG)


def load_persona_emotion(user_id: int, persona_id: int) -> dict:
    """
    Load the persona's current emotional state toward a user.
    Returns dict with keys: emotions (dict), last_updated (datetime), is_new (bool).
    If no record exists, returns default emotions and is_new=True.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT emotions, last_updated
                FROM emotional_relationships
                WHERE user_id = %s AND persona_id = %s
                LIMIT 1;
            """, (user_id, persona_id))
            row = cur.fetchone()
            if row:
                emotions = row[0] if row[0] else dict(DEFAULT_EMOTIONS)
                return {
                    "emotions": emotions,
                    "last_updated": row[1],
                    "is_new": False,
                }
            else:
                return {
                    "emotions": dict(DEFAULT_EMOTIONS),
                    "last_updated": None,
                    "is_new": True,
                }
    finally:
        conn.close()


def save_persona_emotion(user_id: int, persona_id: int, emotions: dict):
    """
    Upsert the persona's emotional state toward a user.
    Creates the record if it doesn't exist, updates if it does.
    Also appends a snapshot to emotion_history.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO emotional_relationships (user_id, persona_id, emotions, last_updated)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (user_id, persona_id)
                DO UPDATE SET
                    emotions = EXCLUDED.emotions,
                    last_updated = EXCLUDED.last_updated
                RETURNING id;
            """, (user_id, persona_id, Json(emotions)))
            rel_id = cur.fetchone()[0]
            # Append to emotion history
            cur.execute("""
                INSERT INTO emotion_history (relationship_id, emotions)
                VALUES (%s, %s);
            """, (rel_id, Json(emotions)))
        conn.commit()
    finally:
        conn.close()


def get_all_persona_emotions(user_id: int) -> list:
    """
    Get all persona emotion states for a user (across all personas).
    Returns list of dicts with persona_id, emotions, last_updated.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT persona_id, emotions, last_updated
                FROM emotional_relationships
                WHERE user_id = %s
                ORDER BY last_updated DESC;
            """, (user_id,))
            return [
                {
                    "persona_id": row[0],
                    "emotions": row[1],
                    "last_updated": row[2],
                }
                for row in cur.fetchall()
            ]
    finally:
        conn.close()
