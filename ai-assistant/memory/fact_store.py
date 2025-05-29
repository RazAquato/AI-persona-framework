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

def store_fact(user_id: str, fact: str, tags: list = None):
    """
    Store a structured fact associated with a user.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO facts (user_id, text, tags)
        VALUES (%s, %s, %s);
    """, (user_id, fact, tags))
    conn.commit()
    cur.close()
    conn.close()

def get_facts(user_id: str):
    """
    Retrieve all structured facts for a given user.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT text, tags FROM facts
        WHERE user_id = %s;
    """, (user_id,))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

