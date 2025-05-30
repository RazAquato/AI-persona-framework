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
    close_driver
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

    @classmethod
    def tearDownClass(cls):
        with cls.driver.session() as session:
            session.run("MATCH (u:User {id: $uid}) DETACH DELETE u", {"uid": cls.test_user_id})
            session.run("MATCH (t:Topic) WHERE t.name IN [$t1, $t2, $t3] DETACH DELETE t", {
                "t1": cls.topic1,
                "t2": cls.topic2,
                "t3": cls.topic3
            })
        cls.driver.close()
        close_driver()

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

if __name__ == "__main__":
    unittest.main()

