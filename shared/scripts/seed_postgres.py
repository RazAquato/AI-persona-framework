# scripts/seed_postgres.py
import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
from psycopg2.extras import Json

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

# 1. Insert test user
cur.execute("INSERT INTO users (name, discord_id) VALUES (%s, %s) RETURNING id;", ("SeedUser", "seed_123"))
user_id = cur.fetchone()[0]

# 2. Insert personality
cur.execute("""
    INSERT INTO user_personalities (user_id, name, description, personality_config)
    VALUES (%s, %s, %s, %s) RETURNING id;
""", (user_id, "HelperBot", "Helpful assistant", Json({'tone': 'friendly'})))

# 3. Insert chat session
cur.execute("""
    INSERT INTO chat_sessions (user_id, personality_id, context_summary)
    VALUES (%s, %s, %s) RETURNING id;
""", (user_id, "HelperBot", "Seeded context"))
session_id = cur.fetchone()[0]

# 4. Insert chat message
cur.execute("""
    INSERT INTO chat_messages (session_id, user_id, role, content, embedding, sentiment, topics)
    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
""", (
    session_id,
    user_id,
    "user",
    "What's the weather like?",
    [0.1] * 384,
    0.5,
    ["weather", "chat"]
))
chat_message_id = cur.fetchone()[0]

# 5. Insert fact
cur.execute("""
    INSERT INTO facts (user_id, text, relevance_score, source_chat_id, tags)
    VALUES (%s, %s, %s, %s, %s);
""", (
    user_id,
    "User asked about weather.",
    0.85,
    chat_message_id,
    ["weather", "seed"]
))

# 6. Insert topic tag
cur.execute("""
    INSERT INTO topic_tags (session_id, topic, confidence, sentiment)
    VALUES (%s, %s, %s, %s);
""", (session_id, "weather", 0.9, 0.4))

# 7. Insert message metadata
cur.execute("""
    INSERT INTO message_metadata (
        message_id, was_buffered, embedding_used, memory_hits,
        memory_layers_used, tool_triggered, response_latency_ms,
        emotional_tone, emotions, personality_tags
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
""", (
    chat_message_id,
    True,
    True,
    2,
    ['short_term'],
    'weather_api',
    120,
    'neutral',
    Json({'joy': 0.2, 'curiosity': 0.7}),
    ['analytical']
))

# 8. Insert emotional relationship
cur.execute("""
    INSERT INTO emotional_relationships (user_id, target_type, target_name, emotions)
    VALUES (%s, %s, %s, %s);
""", (user_id, "assistant", "HelperBot", Json({"trust": 0.8, "comfort": 0.6})))

conn.commit()
cur.close()
conn.close()

print(f"Seeded test data for user_id={user_id}")

