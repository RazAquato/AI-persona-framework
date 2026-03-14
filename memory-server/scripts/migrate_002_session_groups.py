# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Migration 002: Session Groups + Session Archiving

Creates session_groups table and adds group_id + archived columns to chat_sessions.
Safe to run multiple times (uses IF NOT EXISTS / IF NOT EXISTS checks).
"""

import os
import psycopg2
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

conn = psycopg2.connect(
    dbname=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
)
cur = conn.cursor()

# 1. Create session_groups table
cur.execute("""
CREATE TABLE IF NOT EXISTS session_groups (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    persona_id INT REFERENCES user_personalities(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
print("[migrate_002] session_groups table ready")

# 2. Add group_id column to chat_sessions
cur.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'chat_sessions' AND column_name = 'group_id';
""")
if not cur.fetchone():
    cur.execute("""
        ALTER TABLE chat_sessions
        ADD COLUMN group_id INT REFERENCES session_groups(id) ON DELETE SET NULL;
    """)
    print("[migrate_002] Added group_id column to chat_sessions")
else:
    print("[migrate_002] group_id column already exists")

# 3. Add archived column to chat_sessions
cur.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name = 'chat_sessions' AND column_name = 'archived';
""")
if not cur.fetchone():
    cur.execute("""
        ALTER TABLE chat_sessions
        ADD COLUMN archived BOOLEAN DEFAULT FALSE;
    """)
    print("[migrate_002] Added archived column to chat_sessions")
else:
    print("[migrate_002] archived column already exists")

conn.commit()
cur.close()
conn.close()
print("[migrate_002] Migration complete.")
