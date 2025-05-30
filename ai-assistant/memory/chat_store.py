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

def start_chat_session(user_id: int, personality_id: str = None, context_summary: str = None) -> int:
    """
    Create a new chat session and return the session ID.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_sessions (user_id, personality_id, context_summary)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (user_id, personality_id, context_summary))
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

