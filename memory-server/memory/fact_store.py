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
from dotenv import load_dotenv

# Load environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

# Postgres config
PG_CONFIG = {
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
    "database": os.getenv("PG_DATABASE"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
}

def get_connection():
    return psycopg2.connect(**PG_CONFIG)

def store_fact(user_id: int, fact: str, tags: list = None, relevance_score: float = None, source_chat_id: int = None):
    """
    Store a structured fact associated with a user.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO facts (user_id, text, tags, relevance_score, source_chat_id)
        VALUES (%s, %s, %s, %s, %s);
    """, (user_id, fact, tags, relevance_score, source_chat_id))
    conn.commit()
    cur.close()
    conn.close()

def get_facts(user_id: int):
    """
    Retrieve all structured facts for a given user.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, text, tags, relevance_score FROM facts
        WHERE user_id = %s;
    """, (user_id,))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def get_facts_by_tag(user_id: int, tag: str):
    """
    Return facts for a user filtered by tag.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, text, tags FROM facts
        WHERE user_id = %s AND %s = ANY(tags);
    """, (user_id, tag))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def delete_fact(fact_id: int):
    """
    Delete a specific fact by its ID.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM facts WHERE id = %s;", (fact_id,))
    conn.commit()
    cur.close()
    conn.close()

def get_top_facts(user_id: int, limit: int = 5, sort_by: str = 'relevance_score'):
    """
    Return top-N facts sorted by relevance or ID, excluding NULL scores if sorting by relevance.
    """
    if sort_by not in {"relevance_score", "id"}:
        sort_by = "relevance_score"

    conn = get_connection()
    cur = conn.cursor()

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

    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def update_fact(fact_id: int, new_text: str = None, tags: list = None, relevance_score: float = None):
    """
    Update fields of a fact.
    """
    conn = get_connection()
    cur = conn.cursor()
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

    if not updates:
        return

    values.append(fact_id)
    query = f"UPDATE facts SET {', '.join(updates)} WHERE id = %s;"
    cur.execute(query, tuple(values))
    conn.commit()
    cur.close()
    conn.close()

