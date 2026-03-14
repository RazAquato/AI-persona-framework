# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Migration 003: Topic Registry + Salience Tracking

Creates topics, topic_salience, and fact_topics tables for Phase 2.
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

# 1. Topics table — first-class topic entities
cur.execute("""
CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, name)
);
""")
print("[migrate_003] topics table ready")

# 2. Topic salience — per-user, per-persona salience tracking
cur.execute("""
CREATE TABLE IF NOT EXISTS topic_salience (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    persona_id INT REFERENCES user_personalities(id) ON DELETE CASCADE,
    topic_id INT REFERENCES topics(id) ON DELETE CASCADE,
    salience FLOAT DEFAULT 0.0,
    mention_count INT DEFAULT 0,
    last_mentioned TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decay_floor FLOAT DEFAULT 0.0,
    UNIQUE(user_id, persona_id, topic_id)
);
""")
print("[migrate_003] topic_salience table ready")

# 3. Fact-topic junction — links facts to topics via tags
cur.execute("""
CREATE TABLE IF NOT EXISTS fact_topics (
    fact_id INT REFERENCES facts(id) ON DELETE CASCADE,
    topic_id INT REFERENCES topics(id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, topic_id)
);
""")
print("[migrate_003] fact_topics table ready")

conn.commit()
cur.close()
conn.close()
print("[migrate_003] Migration complete.")
