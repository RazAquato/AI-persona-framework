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

"""
Topic Graph (Neo4j)
-------------------
Manages the Zettelkasten-style knowledge graph:
- User → DISCUSSED → Topic (with weight and emotion)
- Topic → RELATED_TO → Topic (with co-occurrence weight)
- User → KNOWS → Entity (people, pets, places)
- Entity → MENTIONED_IN → Topic

IMPORTANT: The Neo4j instance also contains a house automation graph
(bolig_* constraints). Do NOT run MATCH (n) DETACH DELETE n.
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def get_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        notifications_min_severity="OFF"
    )


# --- User-Topic relationships ---

def create_topic_relation(user_id: int, topic: str, emotion: dict = None):
    """
    Create or update a relationship between a user and a topic.
    Increments a `weight` counter each time the user discusses this topic.
    """
    with get_driver() as driver:
        with driver.session() as session:
            session.run("""
                MERGE (u:User {id: $user_id})
                MERGE (t:Topic {name: $topic})
                MERGE (u)-[r:DISCUSSED]->(t)
                SET r.weight = COALESCE(r.weight, 0) + 1,
                    r.last_discussed = datetime(),
                    r += $emotion
            """, {
                "user_id": user_id,
                "topic": topic,
                "emotion": emotion or {}
            })


def get_related_topics(topic: str, user_id: int = None):
    """
    Retrieve other topics related to the given topic.
    When user_id is provided, only returns topics discussed by that user
    (prevents cross-user memory leakage).
    """
    with get_driver() as driver:
        with driver.session() as session:
            if user_id is not None:
                result = session.run("""
                    MATCH (u:User {id: $user_id})-[:DISCUSSED]->(:Topic {name: $topic})
                    MATCH (u)-[:DISCUSSED]->(related:Topic)
                    WHERE related.name <> $topic
                    RETURN DISTINCT related.name AS topic
                """, {"topic": topic, "user_id": user_id})
            else:
                result = session.run("""
                    MATCH (:Topic {name: $topic})<-[:DISCUSSED]-(u:User)-[:DISCUSSED]->(related:Topic)
                    RETURN DISTINCT related.name AS topic
                """, {"topic": topic})
            topics = [record["topic"] for record in result]
        return topics


def get_user_topics(user_id: int, limit: int = 20):
    """
    Get all topics a user has discussed, ordered by weight (most discussed first).
    Returns list of {topic, weight, last_discussed}.
    """
    with get_driver() as driver:
        with driver.session() as session:
            result = session.run("""
                MATCH (u:User {id: $user_id})-[r:DISCUSSED]->(t:Topic)
                RETURN t.name AS topic, r.weight AS weight, r.last_discussed AS last_discussed
                ORDER BY r.weight DESC
                LIMIT $limit
            """, {"user_id": user_id, "limit": limit})
            return [dict(record) for record in result]


# --- Topic-Topic relationships (Zettelkasten cross-linking) ---

def link_topics(topic_a: str, topic_b: str):
    """
    Create or strengthen a co-occurrence link between two topics.
    Called when both topics appear in the same message or session.
    """
    if topic_a == topic_b:
        return

    # Alphabetical ordering to prevent duplicate edges
    t1, t2 = sorted([topic_a, topic_b])

    with get_driver() as driver:
        with driver.session() as session:
            session.run("""
                MERGE (a:Topic {name: $t1})
                MERGE (b:Topic {name: $t2})
                MERGE (a)-[r:RELATED_TO]->(b)
                SET r.weight = COALESCE(r.weight, 0) + 1,
                    r.last_linked = datetime()
            """, {"t1": t1, "t2": t2})


def link_all_topics(topics: list):
    """
    Create pairwise RELATED_TO links between all topics in a list.
    Used when multiple topics are detected in the same message.
    """
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            link_topics(topics[i], topics[j])


def get_topic_network(topic: str, depth: int = 1):
    """
    Get the network of topics related to a given topic.
    Returns list of {related_topic, weight}.
    """
    with get_driver() as driver:
        with driver.session() as session:
            result = session.run("""
                MATCH (t:Topic {name: $topic})-[r:RELATED_TO]-(related:Topic)
                RETURN related.name AS related_topic, r.weight AS weight
                ORDER BY r.weight DESC
            """, {"topic": topic})
            return [dict(record) for record in result]


# --- Entity nodes ---

def create_entity(user_id: int, entity_name: str, entity_type: str, attributes: dict = None):
    """
    Create or update an entity node (person, pet, place, event) linked to a user.
    """
    with get_driver() as driver:
        with driver.session() as session:
            session.run("""
                MERGE (u:User {id: $user_id})
                MERGE (e:Entity {name: $entity_name, type: $entity_type})
                MERGE (u)-[r:KNOWS]->(e)
                SET r.last_mentioned = datetime(),
                    e += $attributes
            """, {
                "user_id": user_id,
                "entity_name": entity_name,
                "entity_type": entity_type,
                "attributes": attributes or {}
            })


def link_entity_to_topic(entity_name: str, topic: str):
    """
    Link an entity to a topic (e.g., pet "Arix" → topic "pets").
    """
    with get_driver() as driver:
        with driver.session() as session:
            session.run("""
                MERGE (e:Entity {name: $entity_name})
                MERGE (t:Topic {name: $topic})
                MERGE (e)-[r:MENTIONED_IN]->(t)
                SET r.count = COALESCE(r.count, 0) + 1
            """, {"entity_name": entity_name, "topic": topic})


def get_user_entities(user_id: int, entity_type: str = None):
    """
    Get all entities a user knows about, optionally filtered by type.
    """
    with get_driver() as driver:
        with driver.session() as session:
            if entity_type:
                result = session.run("""
                    MATCH (u:User {id: $user_id})-[:KNOWS]->(e:Entity {type: $entity_type})
                    RETURN e.name AS name, e.type AS type
                """, {"user_id": user_id, "entity_type": entity_type})
            else:
                result = session.run("""
                    MATCH (u:User {id: $user_id})-[:KNOWS]->(e:Entity)
                    RETURN e.name AS name, e.type AS type
                """, {"user_id": user_id})
            return [dict(record) for record in result]
