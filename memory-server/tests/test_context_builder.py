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
from memory.topic_graph import create_topic_relation
#, close_driver
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

        # Seed vector (must include user_id for isolation filter)
        sample_embedding = [0.01] * 384
        store_embedding(sample_embedding, {
            "user_id": cls.user_id,
            "agent": "maya",
            "role": "user",
            "topics": ["climbing", "adventure"],
            "memory_class": "session_memory",
        })

        # Seed topic link
        create_topic_relation(cls.user_id, "hiking", {"joy_level": 0.8})
        create_topic_relation(cls.user_id, "nature", {"joy_level": 0.7})

    @classmethod
    def tearDownClass(cls):
        cls.cur.execute("DELETE FROM facts WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM users WHERE id = %s;", (cls.user_id,))
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()
        #close_driver()

    def test_context_builder_output(self):
        context = build_context(self.user_id, "Tell me something about climbing")

        self.assertIn("facts", context)
        self.assertIn("vectors", context)
        self.assertIn("topics", context)
        self.assertIn("user_topics", context)
        self.assertIn("raw_input", context)
        self.assertIn("embedded_input", context)

        self.assertIsInstance(context["facts"], list)
        self.assertIsInstance(context["vectors"], list)
        self.assertIsInstance(context["topics"], list)
        self.assertIsInstance(context["user_topics"], list)
        self.assertIsInstance(context["raw_input"], str)
        self.assertIsInstance(context["embedded_input"], list)

    def test_context_with_memory_scope_includes_both_tiers(self):
        """All chatbot personas should see both identity and emotional facts."""
        store_fact(self.user_id, "Context scope identity test", tier="identity")
        store_fact(self.user_id, "Context scope emotional test", tier="emotional")
        scope = {"tier1": True, "tier2": "all"}
        context = build_context(self.user_id, "test input", memory_scope=scope)
        fact_texts = [f[1] for f in context["facts"]]
        self.assertTrue(any("identity test" in t for t in fact_texts))
        self.assertTrue(any("emotional test" in t for t in fact_texts))

    def test_context_without_scope_returns_all(self):
        """No memory_scope = returns all facts."""
        store_fact(self.user_id, "No scope all tiers", tier="emotional")
        context = build_context(self.user_id, "test input", memory_scope=None)
        fact_texts = [f[1] for f in context["facts"]]
        self.assertTrue(any("No scope all tiers" in t for t in fact_texts))

    # --- Additional M2 coverage ---

    def test_context_with_scope_still_returns_all(self):
        """Any memory_scope config should still return all facts (two-tier model)."""
        store_fact(self.user_id, "Scope compat identity fact", tier="identity")
        store_fact(self.user_id, "Scope compat emotional fact", tier="emotional")
        scope = {"tier1": True, "tier2": True}
        context = build_context(self.user_id, "test input", memory_scope=scope)
        fact_texts = [f[1] for f in context["facts"]]
        self.assertTrue(any("Scope compat identity" in t for t in fact_texts))
        self.assertTrue(any("Scope compat emotional" in t for t in fact_texts))

    def test_context_empty_user(self):
        """New user with no data should return empty lists without errors."""
        self.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("EmptyCtxUser",))
        empty_user = self.cur.fetchone()[0]
        self.conn.commit()
        try:
            context = build_context(empty_user, "hello")
            self.assertIsInstance(context["facts"], list)
            self.assertIsInstance(context["vectors"], list)
            self.assertIsInstance(context["topics"], list)
            self.assertIsInstance(context["user_topics"], list)
        finally:
            self.cur.execute("DELETE FROM users WHERE id = %s;", (empty_user,))
            self.conn.commit()

    def test_context_returns_user_topics(self):
        """user_topics should include seeded topics."""
        context = build_context(self.user_id, "anything")
        topic_names = [t["topic"] for t in context["user_topics"]]
        self.assertIn("hiking", topic_names)


if __name__ == "__main__":
    unittest.main()

