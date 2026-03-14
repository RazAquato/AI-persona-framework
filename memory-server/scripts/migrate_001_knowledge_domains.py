# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Migration 001: Knowledge Domains + Persona Access Control

Adds:
  - knowledge_domains table (lookup for available domains)
  - domain TEXT column on facts (nullable — NULL = uncategorized, always visible)
  - persona_id INT column on facts (nullable — NULL = shared, non-NULL = persona-private)
  - domain_access TEXT[] column on user_personalities

Run once against the live database:
    source ~/venvs/AI-persona-framework-venv/bin/activate
    python3 memory-server/scripts/migrate_001_knowledge_domains.py
"""

import psycopg2
import os
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

# 1. Create knowledge_domains lookup table
cur.execute("""
CREATE TABLE IF NOT EXISTS knowledge_domains (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT
);
""")

# 2. Seed default domains
DOMAINS = [
    ("family", "Children, wife, parents, relatives, family events"),
    ("physical", "Health, fitness, workouts, body, medical"),
    ("hobbies", "Football, cooking, photography, gaming, music"),
    ("work", "Job, career, skills, colleagues, projects"),
    ("emotional", "Feelings, moods, mental health, struggles"),
    ("memories", "Life events, travel, nostalgia, milestones"),
    ("other", "Uncategorized / general knowledge"),
]
for name, desc in DOMAINS:
    cur.execute("""
        INSERT INTO knowledge_domains (name, description)
        VALUES (%s, %s)
        ON CONFLICT (name) DO NOTHING;
    """, (name, desc))

# 3. Add domain column to facts (nullable — NULL = uncategorized)
cur.execute("""
    ALTER TABLE facts ADD COLUMN IF NOT EXISTS domain TEXT;
""")

# 4. Add persona_id column to facts (nullable — NULL = shared across personas)
cur.execute("""
    ALTER TABLE facts ADD COLUMN IF NOT EXISTS persona_id INT
        REFERENCES user_personalities(id) ON DELETE SET NULL;
""")

# 5. Add domain_access column to user_personalities
cur.execute("""
    ALTER TABLE user_personalities ADD COLUMN IF NOT EXISTS domain_access TEXT[];
""")

# 6. Set default domain_access for existing personas based on slug
SLUG_DEFAULTS = {
    "psychiatrist": ["family", "physical", "hobbies", "work", "emotional", "memories", "other"],
    "girlfriend": ["physical", "hobbies", "work", "other"],
    "trainer": ["physical", "hobbies", "other"],
    "debug": ["work", "other"],
}
for slug, domains in SLUG_DEFAULTS.items():
    cur.execute("""
        UPDATE user_personalities
        SET domain_access = %s
        WHERE slug = %s AND domain_access IS NULL;
    """, (domains, slug))

conn.commit()
cur.close()
conn.close()

print("Migration 001 complete: knowledge_domains table, domain/persona_id on facts, domain_access on personas.")
