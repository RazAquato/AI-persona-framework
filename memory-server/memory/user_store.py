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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
User store — CRUD operations for the users table.
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


def get_connection():
    return psycopg2.connect(**PG_CONFIG)


def create_user(name: str, password_hash: str) -> int:
    """Create a new user and return the user ID."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (name, password_hash) VALUES (%s, %s) RETURNING id;",
                (name, password_hash),
            )
            user_id = cur.fetchone()[0]
        conn.commit()
        return user_id
    finally:
        conn.close()


def get_user_by_name(name: str):
    """Return (id, name, password_hash) or None."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, password_hash FROM users WHERE LOWER(name) = LOWER(%s);",
                (name,),
            )
            return cur.fetchone()
    finally:
        conn.close()


def get_user_by_id(user_id: int):
    """Return (id, name) or None."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name FROM users WHERE id = %s;", (user_id,))
            return cur.fetchone()
    finally:
        conn.close()


def set_password(user_id: int, password_hash: str):
    """Set or update a user's password hash."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s;",
                (password_hash, user_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_session_owner(session_id: int) -> int | None:
    """Return the user_id that owns a session, or None."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id FROM chat_sessions WHERE id = %s;", (session_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()
