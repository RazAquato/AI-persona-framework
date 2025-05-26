from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.expanduser("~/ai-assistant/config/.env"))

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

with driver.session() as session:
    # Node uniqueness constraints
    session.run("CREATE CONSTRAINT IF NOT EXISTS ON (u:User) ASSERT u.id IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS ON (t:Topic) ASSERT t.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS ON (e:Entity) ASSERT e.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS ON (s:Session) ASSERT s.id IS UNIQUE")

    # Full-text search index for topics
    session.run("""
        CALL db.index.fulltext.createNodeIndex("topicIndex", ["Topic"], ["name"])
    """)

    # Example relationship types:
    # (:User)-[:DISCUSSED {trust, joy, anger, ...}]->(:Topic)
    # (:User)-[:FEELS {love, fear, hate, ...}]->(:Entity)
    # (:Topic)-[:SUBTOPIC_OF]->(:Topic)
    # (:User)-[:REMEMBERS]->(:Session)

    # These relationships are meant to be updated dynamically based on chat inputs and metadata

print("✅ Neo4j schema initialized with emotional and topic graph support.")