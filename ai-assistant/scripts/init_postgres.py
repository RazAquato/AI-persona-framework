import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.expanduser("~/ai-assistant/config/.env"))

conn = psycopg2.connect(
    dbname=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT")
)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Users table
cur.execute("""
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    discord_id TEXT,
    preferences JSONB,
    settings_json JSONB
);
""")

# User-defined AI personalities
cur.execute("""
CREATE TABLE user_personalities (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    name TEXT NOT NULL,
    description TEXT,
    personality_config JSONB,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# Chat sessions
cur.execute("""
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    personality_id TEXT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    context_summary TEXT
);
""")

# Chat messages
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

# Facts table
cur.execute("""
CREATE TABLE facts (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    text TEXT,
    source_chat_id INT REFERENCES chat_messages(id),
    relevance_score FLOAT,
    tags TEXT[]
);
""")

# Topic tags
cur.execute("""
CREATE TABLE topic_tags (
    session_id INT REFERENCES chat_sessions(id),
    topic TEXT,
    confidence FLOAT,
    sentiment FLOAT
);
""")

# Message metadata with JSONB emotion field
cur.execute("""
CREATE TABLE message_metadata (
    id SERIAL PRIMARY KEY,
    message_id INT REFERENCES chat_messages(id) ON DELETE CASCADE,

    -- Memory & context metadata
    was_buffered BOOLEAN,
    embedding_used BOOLEAN,
    memory_hits INT,
    memory_layers_used TEXT[],

    -- Tool and timing
    tool_triggered TEXT,
    response_latency_ms INT,

    -- Emotional and psychological state
    emotional_tone TEXT,
    emotions JSONB,
    personality_tags TEXT[],

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# Emotional relationship model (user → target)
cur.execute("""
CREATE TABLE emotional_relationships (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    target_type TEXT,
    target_name TEXT,
    emotions JSONB,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()
cur.close()
conn.close()
print("PostgreSQL schema created with user-defined AI personalities and emotion model.")