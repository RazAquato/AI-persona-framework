from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

with driver.session() as session:
    # Node constraints
    session.run("CREATE CONSTRAINT IF NOT EXISTS ON (u:User) ASSERT u.id IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS ON (t:Topic) ASSERT t.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS ON (e:Entity) ASSERT e.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS ON (s:Session) ASSERT s.id IS UNIQUE")

    # Full-text search index on Topic
    session.run("""
        CALL db.index.fulltext.createNodeIndex("topicIndex", ["Topic"], ["name"])
    """)

    # Sample relationship types:
    # (:User)-[:DISCUSSED {confidence, trust_level, joy_level, ...}]->(:Topic)
    # (:User)-[:LIKES]->(:Entity)
    # (:Topic)-[:SUBTOPIC_OF]->(:Topic)

    # You can programmatically add more relationships at runtime like:
    # MATCH (u:User), (t:Topic) WHERE u.id = $uid AND t.name = $topic
    # MERGE (u)-[r:DISCUSSED]->(t)
    # SET r.joy_level = 0.8, r.trust_level = 0.6 ...

print("Neo4j schema initialized with emotional structure support.")
