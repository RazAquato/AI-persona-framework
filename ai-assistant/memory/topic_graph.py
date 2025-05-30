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
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def create_topic_relation(user_id: int, topic: str, emotion: dict = None):
    """
    Create or update a relationship between a user and a topic,
    including optional emotional metadata.
    """
    with get_driver() as driver:
        with driver.session() as session:
            session.run("""
                MERGE (u:User {id: $user_id})
                MERGE (t:Topic {name: $topic})
                MERGE (u)-[r:DISCUSSED]->(t)
                SET r += $emotion
            """, {
                "user_id": user_id,
                "topic": topic,
                "emotion": emotion or {}
            })


def get_related_topics(topic: str):
    """
    Retrieve other topics discussed by users who also discussed the given topic.
    """
    with get_driver() as driver:
        with driver.session() as session:
            result = session.run("""
                MATCH (:Topic {name: $topic})<-[:DISCUSSED]-(u:User)-[:DISCUSSED]->(related:Topic)
                RETURN DISTINCT related.name AS topic
            """, {"topic": topic})
            topics = [record["topic"] for record in result]
        return topics
