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

import psycopg2
import os
from dotenv import load_dotenv
from psycopg2.extras import Json
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

def store_emotion_vector(message_id: int, vector: dict, tone: str = None):
    """
    Stores a full emotion vector (as JSONB) and optional tone label for a message.
    If the message already has metadata, this will update the emotion fields.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """
            INSERT INTO message_metadata (message_id, emotions, emotional_tone)
            VALUES (%s, %s, %s)
            ON CONFLICt (message_id)
            DO UPDATE SET
                emotions = EXCLUDED.emotions,
                emotional_tone = EXCLUDED.emotional_tone,
                created_at = CURRENT_TIMESTAMP;
            """
            cur.execute(query, (message_id, Json(vector), tone))
        conn.commit()
    finally:
        conn.close()
        

def load_latest_emotion_state(user_id: int, role: str = "user") -> dict:
    """
    Loads the most recent emotion vector stored for a given user and role.
    Returns None if no data is found.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT m.emotions, m.emotional_tone, m.created_at
                FROM message_metadata m
                JOIN chat_messages c ON m.message_id = c.id
                WHERE c.user_id = %s AND c.role = %s
                ORDER BY m.created_at DESC
                LIMIT 1;
            """, (user_id, role))
            row = cur.fetchone()
            if row:
                return {
                    "emotions": row[0],
                    "tone": row[1],
                    "timestamp": row[2]
                }
            else:
                return None
    finally:
        conn.close()
 
 
    # cur = conn.cursor()

    # query = """
    # SELECT mmd.emotions
    # FROM chat_messages cm
    # JOIN message_metadata mmd ON mmd.message_id = cm.id
    # WHERE cm.user_id = %s AND cm.role = %s
    # ORDER BY cm.timestamp DESC
    # LIMIT 1;
    # """

    # cur.execute(query, (user_id, role))
    # row = cur.fetchone()
    # cur.close()
    # conn.close()

    # return row[0] if row else None
