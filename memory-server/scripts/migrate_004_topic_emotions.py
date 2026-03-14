# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Migration 004: User-Topic Emotions

Creates user_topic_emotions table for Phase 3 — tracks how the user
feels about each topic, classified by nightly reflection job.
Safe to run multiple times (uses IF NOT EXISTS).
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

# User-topic emotions — how the user feels about each topic
# A topic can have multiple emotion rows simultaneously (love AND frustration)
cur.execute("""
CREATE TABLE IF NOT EXISTS user_topic_emotions (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    topic_id INT REFERENCES topics(id) ON DELETE CASCADE,
    emotion TEXT NOT NULL,
    intensity FLOAT DEFAULT 0.0,
    source TEXT DEFAULT 'reflection',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, topic_id, emotion)
);
""")
print("[migrate_004] user_topic_emotions table ready")

conn.commit()
cur.close()
conn.close()
print("[migrate_004] Migration complete.")
