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

import unittest
import os
import psycopg2
from dotenv import load_dotenv
from neo4j import GraphDatabase

from memory.embedding import embed_text
from memory.vector_store import store_embedding, search_similar_vectors
from memory.fact_store import store_fact, get_facts
from memory.topic_graph import create_topic_relation, get_related_topics, close_driver


class TestMemoryFlow(unittest.TestCase):

    def setUp(self):
        self.sample_text = "I love hiking in the mountains."
        self.sample_topic = "hiking"
        self.sample_related_topic = "outdoors"
        self.sample_fact = "User enjoys hiking in nature."
        self.agent = "maya"

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
        load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

        # PostgreSQL connection
        self.conn = psycopg2.connect(
            dbname=os.getenv("PG_DATABASE"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )
        cur = self.conn.cursor()
        cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("Test User",))
        self.test_user_id = cur.fetchone()[0]
        self.conn.commit()
        cur.close()

        # Neo4j setup
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))

    def tearDown(self):
        # Cleanup Postgres
        cur = self.conn.cursor()
        cur.execute("DELETE FROM facts WHERE user_id = %s;", (self.test_user_id,))
        cur.execute("DELETE FROM users WHERE id = %s;", (self.test_user_id,))
        self.conn.commit()
        cur.close()
        self.conn.close()

        # Cleanup Neo4j
        with self.driver.session() as session:
            session.run("MATCH (u:User {id: $uid}) DETACH DELETE u", {"uid": self.test_user_id})
            session.run("MATCH (t:Topic) WHERE t.name IN [$t1, $t2] DETACH DELETE t", {
                "t1": self.sample_topic,
                "t2": self.sample_related_topic
            })
        self.driver.close()

    @classmethod
    def tearDownClass(cls):
        close_driver()

    def test_embedding_and_qdrant(self):
        embedding = embed_text(self.sample_text)
        metadata = {
            "role": "user",
            "agent": self.agent,
            "topics": ["hiking", "nature"],
            "joy_level": 0.95
        }
        point_id = store_embedding(embedding, metadata)
        self.assertIsNotNone(point_id)

        results = search_similar_vectors(embedding, top_k=1)
        self.assertGreater(len(results), 0)
        print("Qdrant result payload:", results[0].payload)

    def test_fact_store_postgres(self):
        store_fact(self.test_user_id, self.sample_fact, tags=["outdoors", "hobby"])
        facts = get_facts(self.test_user_id)
        self.assertTrue(any(self.sample_fact in f for f, _ in facts))
        print("Postgres facts:", facts)

    def test_topic_graph_neo4j(self):
        # Create relations to both topics
        create_topic_relation(self.test_user_id, self.sample_topic, {"joy_level": 0.9})
        create_topic_relation(self.test_user_id, self.sample_related_topic, {"joy_level": 0.7})

        # Fetch related topics through user-centered linkage
        related = get_related_topics(self.sample_topic)
        print("Neo4j related topics:", related)
        self.assertIn(self.sample_related_topic, related)

if __name__ == "__main__":
    unittest.main()

