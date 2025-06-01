# tests/test_init_neo4j.py
import os
import unittest
from neo4j import GraphDatabase
from dotenv import load_dotenv

class TestNeo4jInit(unittest.TestCase):
    def setUp(self):
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
        load_dotenv(dotenv_path=env_path)
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

    def test_constraints_exist(self):
        with self.driver.session() as session:
            result = session.run("SHOW CONSTRAINTS")
            names = [r["name"] for r in result]
            self.assertIn("user_id_unique", names)
            self.assertIn("topic_name_unique", names)

    def tearDown(self):
        self.driver.close()

