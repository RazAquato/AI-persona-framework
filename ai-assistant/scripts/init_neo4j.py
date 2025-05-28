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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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