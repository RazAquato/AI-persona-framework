from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")

load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)
with driver.session() as session:
    # Optional: Wipe all existing data and constraints
    session.run("MATCH (n) DETACH DELETE n")
    try:
        session.run("CALL apoc.schema.assert({}, {})")  # Only works if APOC is installed
    except Exception:
        print("⚠️ Skipped APOC schema cleanup (plugin not available or disabled)")

    # New-style schema (Neo4j v5+)
    session.run("CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
    session.run("CREATE CONSTRAINT topic_name_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
    session.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
    session.run("CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE")

    # Full-text search index for topics
    session.run("""
    CREATE FULLTEXT INDEX topicIndex IF NOT EXISTS FOR (t:Topic) ON EACH [t.name]
""")

print("eo4j schema initialized with updated constraint syntax.")

