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

import os
import json
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(dotenv_path=os.path.expanduser("~/ai-assistant/config/.env"))

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT")
    )

def create_or_update_agent(user_id, name, description, config):
    conn = get_db_connection()
    cur = conn.cursor()

    # Check if it already exists
    cur.execute("""
        SELECT id FROM user_personalities
        WHERE user_id = %s AND name = %s
    """, (user_id, name))
    existing = cur.fetchone()

    if existing:
        cur.execute("""
            UPDATE user_personalities
            SET description = %s,
                personality_config = %s,
                created_at = %s
            WHERE id = %s
        """, (description, json.dumps(config), datetime.utcnow(), existing[0]))
        print(f"✅ Updated personality '{name}' for user {user_id}.")
    else:
        cur.execute("""
            INSERT INTO user_personalities (user_id, name, description, personality_config)
            VALUES (%s, %s, %s, %s)
        """, (user_id, name, description, json.dumps(config)))
        print(f"✅ Created new personality '{name}' for user {user_id}.")

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create or update an AI personality for a user.")
    parser.add_argument("--user", type=int, required=True, help="User ID")
    parser.add_argument("--name", required=True, help="Personality name")
    parser.add_argument("--desc", default="", help="Personality description")
    parser.add_argument("--config", required=True, help="Path to personality config JSON")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    create_or_update_agent(args.user, args.name, args.desc, config)