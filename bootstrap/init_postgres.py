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

cur.execute("""
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    discord_id TEXT,
    preferences JSONB,
    settings_json JSONB
);
""")

cur.execute("""
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    personality_id TEXT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    context_summary TEXT
);
""")

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

cur.execute("""
CREATE TABLE topic_tags (
    session_id INT REFERENCES chat_sessions(id),
    topic TEXT,
    confidence FLOAT,
    sentiment FLOAT
);
""")

conn.commit()
cur.close()
conn.close()
print("✅ PostgreSQL tables created.")
