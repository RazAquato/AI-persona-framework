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
from memory.fact_store import store_fact
from memory.vector_store import store_embedding
from memory.topic_graph import create_topic_relation, close_driver
from memory.context_builder import build_context

class TestContextBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
        load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

        cls.conn = psycopg2.connect(
            dbname=os.getenv("PG_DATABASE"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )
        cls.cur = cls.conn.cursor()

        cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("ContextTestUser",))
        cls.user_id = cls.cur.fetchone()[0]
        cls.conn.commit()

        # Seed fact
        store_fact(cls.user_id, "User enjoys rock climbing", tags=["outdoors", "hobby"], relevance_score=0.9)

        # Seed vector
        sample_embedding = [0.01] * 384
        store_embedding(sample_embedding, {
            "agent": "maya",
            "role": "user",
            "topics": ["climbing", "sports"]
        })

        # Seed topic link
        create_topic_relation(cls.user_id, "climbing", {"joy_level": 0.8})
        create_topic_relation(cls.user_id, "adventure", {"joy_level": 0.7})

    @classmethod
    def tearDownClass(cls):
        cls.cur.execute("DELETE FROM facts WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM users WHERE id = %s;", (cls.user_id,))
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()
        close_driver()

    def test_context_builder_output(self):
        context = build_context(self.user_id, "Tell me something about climbing")

        self.assertIn("facts", context)
        self.assertIn("vectors", context)
        self.assertIn("topics", context)
        self.assertIn("raw_input", context)
        self.assertIn("embedded_input", context)

        self.assertIsInstance(context["facts"], list)
        self.assertIsInstance(context["vectors"], list)
        self.assertIsInstance(context["topics"], list)
        self.assertIsInstance(context["raw_input"], str)
        self.assertIsInstance(context["embedded_input"], list)

        print("Context facts:", context["facts"])
        print("Vector matches:", context["vectors"])
        print("Related topics:", context["topics"])

if __name__ == "__main__":
    unittest.main()

