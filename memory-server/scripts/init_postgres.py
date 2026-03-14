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
# along with this program. If not, see <https://www.gnu.org/licenses/>

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
    port=os.getenv("PG_PORT")
)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Users
cur.execute("""
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    password_hash TEXT,
    discord_id TEXT,
    preferences JSONB,
    settings_json JSONB
);
""")

# AI Personas (owned by users, seeded on registration)
cur.execute("""
CREATE TABLE user_personalities (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    system_prompt TEXT,
    nsfw_capable BOOLEAN DEFAULT FALSE,
    nsfw_prompt_addon TEXT,
    memory_scope JSONB,
    personality_config JSONB,
    is_public BOOLEAN DEFAULT FALSE,
    domain_access TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, slug)
);
""")

# Session Groups (project folders)
cur.execute("""
CREATE TABLE session_groups (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    persona_id INT REFERENCES user_personalities(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# Chat Sessions
cur.execute("""
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    persona_id INT REFERENCES user_personalities(id),
    group_id INT REFERENCES session_groups(id) ON DELETE SET NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    context_summary TEXT,
    incognito BOOLEAN DEFAULT FALSE,
    nsfw_mode BOOLEAN DEFAULT FALSE,
    archived BOOLEAN DEFAULT FALSE
);
""")

# Chat Messages
cur.execute("""
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    session_id INT REFERENCES chat_sessions(id),
    user_id INT REFERENCES users(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role TEXT CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT,
    embedding VECTOR(384),
    sentiment FLOAT,
    topics TEXT[]
);
""")

# Message Metadata (1:1 with chat_messages)
cur.execute("""
CREATE TABLE message_metadata (
    message_id INT PRIMARY KEY REFERENCES chat_messages(id) ON DELETE CASCADE,
    was_buffered BOOLEAN,
    embedding_used BOOLEAN,
    memory_hits INT,
    memory_layers_used TEXT[],
    tool_triggered TEXT,
    response_latency_ms INT,
    emotional_tone TEXT,
    emotions JSONB,
    personality_tags TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# Knowledge Domains (lookup table for fact categorization)
cur.execute("""
CREATE TABLE knowledge_domains (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT
);
""")

# Seed default domains
for name, desc in [
    ("family", "Children, wife, parents, relatives, family events"),
    ("physical", "Health, fitness, workouts, body, medical"),
    ("hobbies", "Football, cooking, photography, gaming, music"),
    ("work", "Job, career, skills, colleagues, projects"),
    ("emotional", "Feelings, moods, mental health, struggles"),
    ("memories", "Life events, travel, nostalgia, milestones"),
    ("other", "Uncategorized / general knowledge"),
]:
    cur.execute("INSERT INTO knowledge_domains (name, description) VALUES (%s, %s);", (name, desc))

# Facts
cur.execute("""
CREATE TABLE facts (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    text TEXT,
    source_chat_id INT REFERENCES chat_messages(id),
    relevance_score FLOAT,
    tags TEXT[],
    source_type TEXT DEFAULT 'conversation',
    source_ref TEXT,
    tier TEXT DEFAULT 'identity',
    entity_type TEXT,
    valence TEXT,
    domain TEXT,
    persona_id INT REFERENCES user_personalities(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# Topic Tags (legacy per-session tags)
cur.execute("""
CREATE TABLE topic_tags (
    session_id INT REFERENCES chat_sessions(id),
    topic TEXT,
    confidence FLOAT,
    sentiment FLOAT
);
""")

# Topics (first-class topic entities)
cur.execute("""
CREATE TABLE topics (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, name)
);
""")

# Topic Salience (per-user, per-persona salience tracking)
cur.execute("""
CREATE TABLE topic_salience (
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

# Fact-Topic junction (links facts to topics via tags)
cur.execute("""
CREATE TABLE fact_topics (
    fact_id INT REFERENCES facts(id) ON DELETE CASCADE,
    topic_id INT REFERENCES topics(id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, topic_id)
);
""")

# User-Topic Emotions (how the user feels about each topic)
cur.execute("""
CREATE TABLE user_topic_emotions (
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

# Emotional Relationships (per user + persona)
cur.execute("""
CREATE TABLE emotional_relationships (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    persona_id INT REFERENCES user_personalities(id),
    emotions JSONB,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, persona_id)
);
""")

# Emotion History (append-only log of emotion state changes)
cur.execute("""
CREATE TABLE emotion_history (
    id SERIAL PRIMARY KEY,
    relationship_id INT REFERENCES emotional_relationships(id) ON DELETE CASCADE,
    emotions JSONB NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()
cur.close()
conn.close()

print("PostgreSQL schema created.")
