import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

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

# Expanded metadata table
cur.execute("""
CREATE TABLE message_metadata (
    id SERIAL PRIMARY KEY,
    message_id INT REFERENCES chat_messages(id) ON DELETE CASCADE,

    -- Memory & context metadata
    was_buffered BOOLEAN,
    embedding_used BOOLEAN,
    memory_hits INT,
    memory_layers_used TEXT[],  -- ['buffer', 'vector', 'graph']

    -- Tool and timing
    tool_triggered TEXT,
    response_latency_ms INT,

    -- Emotional and psychological state
    emotional_tone TEXT,
    trust_level FLOAT,
    love_level FLOAT,
    humor_level FLOAT,
    anger_level FLOAT,
    joy_level FLOAT,
    sadness_level FLOAT,
    fear_level FLOAT,
    curiosity_level FLOAT,
    dominance_level FLOAT,
    submission_level FLOAT,
    personality_tags TEXT[],

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()
cur.close()
conn.close()
print("PostgreSQL tables with full mood and metadata support created.")
