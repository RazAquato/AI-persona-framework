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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, slug)
);
""")

# Chat Sessions
cur.execute("""
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    persona_id INT REFERENCES user_personalities(id),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    context_summary TEXT,
    incognito BOOLEAN DEFAULT FALSE,
    nsfw_mode BOOLEAN DEFAULT FALSE
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# Topic Tags
cur.execute("""
CREATE TABLE topic_tags (
    session_id INT REFERENCES chat_sessions(id),
    topic TEXT,
    confidence FLOAT,
    sentiment FLOAT
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
