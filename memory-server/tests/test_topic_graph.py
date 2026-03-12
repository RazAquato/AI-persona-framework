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
from dotenv import load_dotenv
from neo4j import GraphDatabase

from memory.topic_graph import (
    create_topic_relation,
    get_related_topics,
    get_user_topics,
    link_topics,
    link_all_topics,
    get_topic_network,
    create_entity,
    link_entity_to_topic,
    get_user_entities,
)

class TestTopicGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
        load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

        cls.neo4j_uri = os.getenv("NEO4J_URI")
        cls.neo4j_user = os.getenv("NEO4J_USER")
        cls.neo4j_password = os.getenv("NEO4J_PASSWORD")
        cls.driver = GraphDatabase.driver(cls.neo4j_uri, auth=(cls.neo4j_user, cls.neo4j_password))

        # Unique test values
        cls.test_user_id = "test_user_neo4j"
        cls.topic1 = "philosophy"
        cls.topic2 = "ethics"
        cls.topic3 = "mathematics"

        # Clean up before test
        with cls.driver.session() as session:
            session.run("MATCH (n {id: $uid}) DETACH DELETE n", {"uid": cls.test_user_id})
            session.run("MATCH (t:Topic) WHERE t.name IN [$t1, $t2, $t3] DETACH DELETE t", {
                "t1": cls.topic1,
                "t2": cls.topic2,
                "t3": cls.topic3
            })

    def test_create_and_retrieve_single_relation(self):
        #create two topics for same user
        create_topic_relation(self.test_user_id, self.topic1, {"joy_level": 0.8})
        create_topic_relation(self.test_user_id, self.topic2, {"joy_level": 0.5})

        # Query related to topic1 -> should include topic2
        related = get_related_topics(self.topic1)
        self.assertIn(self.topic2, related)

    def test_multiple_topic_relations(self):
        create_topic_relation(self.test_user_id, self.topic2, {"joy_level": 0.6})
        create_topic_relation(self.test_user_id, self.topic3, {"joy_level": 0.4})

        related = get_related_topics(self.topic2)
        self.assertTrue(self.topic3 in related or self.topic2 in related)

    def test_related_topics_empty(self):
        unrelated_topic = "unknown_topic"
        related = get_related_topics(unrelated_topic)
        self.assertIsInstance(related, list)
        self.assertEqual(len(related), 0)

    def test_create_relation_with_minimal_metadata(self):
        create_topic_relation(self.test_user_id, "minimal_topic", {})
        # Should not fail or throw

    # --- M2 additions ---

    def test_get_user_topics(self):
        """get_user_topics returns topics with weight and last_discussed."""
        topics = get_user_topics(self.test_user_id, limit=10)
        self.assertIsInstance(topics, list)
        if topics:
            self.assertIn("topic", topics[0])
            self.assertIn("weight", topics[0])

    def test_topic_weight_increments(self):
        """Discussing same topic twice should increment weight."""
        create_topic_relation(self.test_user_id, self.topic1)
        create_topic_relation(self.test_user_id, self.topic1)
        topics = get_user_topics(self.test_user_id)
        topic1_entry = next((t for t in topics if t["topic"] == self.topic1), None)
        self.assertIsNotNone(topic1_entry)
        self.assertGreaterEqual(topic1_entry["weight"], 2)

    def test_link_topics(self):
        """link_topics creates a RELATED_TO edge between two topics."""
        link_topics(self.topic1, self.topic2)
        network = get_topic_network(self.topic1)
        related_names = [r["related_topic"] for r in network]
        self.assertIn(self.topic2, related_names)

    def test_link_topics_self_link_noop(self):
        """Linking a topic to itself should be a no-op."""
        link_topics(self.topic1, self.topic1)
        # Should not raise

    def test_link_all_topics(self):
        """link_all_topics creates pairwise links."""
        link_all_topics([self.topic1, self.topic2, self.topic3])
        net1 = get_topic_network(self.topic1)
        related_names = [r["related_topic"] for r in net1]
        self.assertIn(self.topic2, related_names)
        self.assertIn(self.topic3, related_names)

    def test_get_topic_network(self):
        """get_topic_network returns related topics with weights."""
        link_topics(self.topic2, self.topic3)
        network = get_topic_network(self.topic2)
        self.assertIsInstance(network, list)
        if network:
            self.assertIn("related_topic", network[0])
            self.assertIn("weight", network[0])

    def test_create_and_get_entity(self):
        """create_entity + get_user_entities round-trip."""
        create_entity(self.test_user_id, "TestDog", "pet")
        entities = get_user_entities(self.test_user_id, entity_type="pet")
        names = [e["name"] for e in entities]
        self.assertIn("TestDog", names)

    def test_get_user_entities_all_types(self):
        """get_user_entities without type filter returns all."""
        create_entity(self.test_user_id, "TestPerson", "person")
        create_entity(self.test_user_id, "TestPlace", "place")
        entities = get_user_entities(self.test_user_id)
        types = {e["type"] for e in entities}
        self.assertTrue(len(types) >= 1)

    def test_link_entity_to_topic(self):
        """link_entity_to_topic creates MENTIONED_IN edge."""
        create_entity(self.test_user_id, "EntityTopicDog", "pet")
        create_topic_relation(self.test_user_id, "pets_topic_test")
        link_entity_to_topic("EntityTopicDog", "pets_topic_test")
        # Verify via direct query
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: 'EntityTopicDog'})-[r:MENTIONED_IN]->(t:Topic {name: 'pets_topic_test'})
                RETURN r.count AS count
            """)
            record = result.single()
            self.assertIsNotNone(record)
            self.assertGreaterEqual(record["count"], 1)

    @classmethod
    def tearDownClass(cls):
        with cls.driver.session() as session:
            session.run("MATCH (u:User {id: $uid}) DETACH DELETE u", {"uid": cls.test_user_id})
            session.run("MATCH (t:Topic) WHERE t.name IN [$t1, $t2, $t3] DETACH DELETE t", {
                "t1": cls.topic1,
                "t2": cls.topic2,
                "t3": cls.topic3
            })
            # Clean up M2 test data
            session.run("MATCH (t:Topic) WHERE t.name IN ['minimal_topic', 'pets_topic_test'] DETACH DELETE t")
            session.run("MATCH (e:Entity) WHERE e.name IN ['TestDog', 'TestPerson', 'TestPlace', 'EntityTopicDog'] DETACH DELETE e")
        cls.driver.close()


if __name__ == "__main__":
    unittest.main()

