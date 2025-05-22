from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

with driver.session() as session:
    session.run("CREATE CONSTRAINT IF NOT EXISTS ON (t:Topic) ASSERT t.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS ON (u:User) ASSERT u.id IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS ON (e:Entity) ASSERT e.name IS UNIQUE")

    session.run("""
        CALL db.index.fulltext.createNodeIndex("topicIndex", ["Topic"], ["name"])
    """)
print("✅ Neo4j schema initialized.")
