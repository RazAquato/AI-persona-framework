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

from memory.fact_store import (
    store_fact,
    get_facts,
    get_facts_by_tag,
    delete_fact,
    get_top_facts,
    update_fact
)

class TestFactStore(unittest.TestCase):

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
        cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("TestUser",))
        cls.test_user_id = cls.cur.fetchone()[0]
        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        cls.cur.execute("DELETE FROM facts WHERE user_id = %s;", (cls.test_user_id,))
        cls.cur.execute("DELETE FROM users WHERE id = %s;", (cls.test_user_id,))
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    def test_store_and_get_fact_with_tags(self):
        store_fact(self.test_user_id, "User likes sci-fi", tags=["books", "genre"])
        results = get_facts(self.test_user_id)
        self.assertTrue(any("sci-fi" in f[1] for f in results))

    def test_store_fact_with_none_tags(self):
        store_fact(self.test_user_id, "Fact with no tags", tags=None)
        results = get_facts(self.test_user_id)
        self.assertTrue(any("no tags" in f[1] for f in results))

    def test_store_duplicate_fact(self):
        store_fact(self.test_user_id, "Duplicate fact", tags=["test"])
        store_fact(self.test_user_id, "Duplicate fact", tags=["test"])
        results = [fact[1] for fact in get_facts(self.test_user_id)]
        self.assertEqual(results.count("Duplicate fact"), 2)

    def test_get_facts_empty(self):
        self.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("EmptyUser",))
        empty_user_id = self.cur.fetchone()[0]
        self.conn.commit()
        facts = get_facts(empty_user_id)
        self.assertEqual(facts, [])
        self.cur.execute("DELETE FROM users WHERE id = %s;", (empty_user_id,))
        self.conn.commit()

    def test_invalid_user_id(self):
        with self.assertRaises(Exception):
            store_fact("", "Invalid user test", tags=["error"])

    def test_get_facts_by_tag(self):
        store_fact(self.test_user_id, "Tagged with AI", tags=["ai", "tech"])
        tag_results = get_facts_by_tag(self.test_user_id, "ai")
        self.assertTrue(any("AI" in f[1] for f in tag_results))

    def test_get_top_facts(self):
        store_fact(self.test_user_id, "High relevance", relevance_score=0.95)
        store_fact(self.test_user_id, "Low relevance", relevance_score=0.1)
        top_facts = get_top_facts(self.test_user_id, limit=1)
        self.assertEqual(top_facts[0][1], "High relevance")

    def test_update_fact(self):
        store_fact(self.test_user_id, "Temp fact", tags=["old"], relevance_score=0.5)
        fact_id = get_top_facts(self.test_user_id, limit=1)[0][0]
        update_fact(fact_id, new_text="Updated fact", tags=["new"], relevance_score=0.9)
        updated = get_facts(self.test_user_id)
        self.assertTrue(any("Updated fact" in f[1] for f in updated))

    def test_delete_fact(self):
        store_fact(self.test_user_id, "To be deleted", tags=["temp"], relevance_score=0.5)  # FIXED
    
        top_facts = get_top_facts(self.test_user_id, limit=1)
        self.assertTrue(len(top_facts) > 0, "No facts returned from get_top_facts")

        fact_id = top_facts[0][0]
        delete_fact(fact_id)

        remaining = get_facts(self.test_user_id)
        self.assertFalse(any("To be deleted" in f[1] for f in remaining))

if __name__ == "__main__":
    unittest.main()

