# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
import os
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from memory.ingest_pipeline import ingest
from memory.fact_store import get_facts, make_fact_blob, delete_facts_by_source
from memory.topic_store import get_or_create_topic, get_salience, list_topics


class TestIngestPipeline(unittest.TestCase):

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
            port=os.getenv("PG_PORT"),
        )
        cls.cur = cls.conn.cursor()
        cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;",
                        ("IngestPipelineTestUser",))
        cls.user_id = cls.cur.fetchone()[0]
        cls.cur.execute("""
            INSERT INTO user_personalities (user_id, slug, name, system_prompt, memory_scope)
            VALUES (%s, %s, %s, %s, %s) RETURNING id;
        """, (cls.user_id, "ingest_test_persona", "IngestPersona", "Test.", Json({"tier1": True})))
        cls.persona_id = cls.cur.fetchone()[0]
        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        cls.cur.execute("DELETE FROM user_topic_emotions WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM topic_salience WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM fact_topics WHERE fact_id IN (SELECT id FROM facts WHERE user_id = %s);", (cls.user_id,))
        cls.cur.execute("DELETE FROM facts WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM topics WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM chat_messages WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM chat_sessions WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM session_groups WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM user_personalities WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM users WHERE id = %s;", (cls.user_id,))
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    # --- Core ingestion ---

    def test_ingest_stores_facts(self):
        extracted = {
            "facts": [make_fact_blob("Pipeline test fact alpha", tags=["test"])],
            "entities": [],
            "topics": [{"topic": "testing", "confidence": 0.8}],
        }
        result = ingest(self.user_id, extracted)
        self.assertEqual(result["facts_stored"], 1)
        self.assertEqual(result["topics_found"], 1)

    def test_ingest_deduplicates(self):
        extracted = {
            "facts": [
                make_fact_blob("Dedup pipeline test beta"),
                make_fact_blob("Dedup pipeline test beta"),
            ],
            "entities": [],
            "topics": [],
        }
        result = ingest(self.user_id, extracted)
        self.assertEqual(result["facts_stored"], 1)
        self.assertEqual(result["facts_skipped"], 1)

    def test_ingest_returns_topic_names(self):
        extracted = {
            "facts": [make_fact_blob("Topic names test gamma", tags=["music"])],
            "entities": [],
            "topics": [
                {"topic": "music", "confidence": 0.8},
                {"topic": "art", "confidence": 0.6},
            ],
        }
        result = ingest(self.user_id, extracted)
        self.assertIn("music", result["topic_names"])
        self.assertIn("art", result["topic_names"])

    def test_ingest_empty_extracted(self):
        result = ingest(self.user_id, {"facts": [], "entities": [], "topics": []})
        self.assertEqual(result["facts_stored"], 0)
        self.assertEqual(result["topics_found"], 0)

    # --- Salience bumping ---

    def test_ingest_bumps_salience_with_persona(self):
        extracted = {
            "facts": [make_fact_blob("Salience bump test delta", tags=["salience_test_xyz"])],
            "entities": [],
            "topics": [{"topic": "salience_test_xyz", "confidence": 0.9}],
        }
        ingest(self.user_id, extracted, persona_id=self.persona_id)
        tid = get_or_create_topic(self.user_id, "salience_test_xyz")
        s = get_salience(self.user_id, self.persona_id, tid)
        self.assertIsNotNone(s)
        self.assertGreater(s["salience"], 0)

    def test_ingest_no_salience_without_persona(self):
        extracted = {
            "facts": [make_fact_blob("No persona salience test epsilon", tags=["no_persona_xyz"])],
            "entities": [],
            "topics": [{"topic": "no_persona_xyz", "confidence": 0.9}],
        }
        ingest(self.user_id, extracted)  # no persona_id
        tid = get_or_create_topic(self.user_id, "no_persona_xyz")
        s = get_salience(self.user_id, self.persona_id, tid)
        # Should be None — no salience record created without persona
        self.assertIsNone(s)

    # --- Snapshot mode ---

    def test_snapshot_deletes_old_facts(self):
        # First ingest
        extracted1 = {
            "facts": [make_fact_blob("Snapshot old fact zeta", source_type="test_snapshot")],
            "entities": [],
            "topics": [],
        }
        ingest(self.user_id, extracted1, source_type="test_snapshot")
        facts_before = get_facts(self.user_id)
        old_texts = [f[1] for f in facts_before]
        self.assertTrue(any("Snapshot old fact zeta" in t for t in old_texts))

        # Snapshot ingest — should delete old and insert new
        extracted2 = {
            "facts": [make_fact_blob("Snapshot new fact eta", source_type="test_snapshot")],
            "entities": [],
            "topics": [],
        }
        ingest(self.user_id, extracted2, source_type="test_snapshot", snapshot=True)
        facts_after = get_facts(self.user_id)
        after_texts = [f[1] for f in facts_after]
        self.assertFalse(any("Snapshot old fact zeta" in t for t in after_texts))
        self.assertTrue(any("Snapshot new fact eta" in t for t in after_texts))

    # --- Source type override ---

    def test_source_type_override(self):
        extracted = {
            "facts": [make_fact_blob("Source override test theta", source_type=None)],
            "entities": [],
            "topics": [],
        }
        ingest(self.user_id, extracted, source_type="custom_source")
        # Verify it was stored with the right source_type
        self.cur.execute(
            "SELECT source_type FROM facts WHERE user_id = %s AND text = %s;",
            (self.user_id, "Source override test theta"),
        )
        row = self.cur.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "custom_source")

    # --- Topic linking via tags ---

    def test_ingest_links_facts_to_topics_via_tags(self):
        extracted = {
            "facts": [make_fact_blob("Tag linking test iota", tags=["sports", "fitness"])],
            "entities": [],
            "topics": [],
        }
        ingest(self.user_id, extracted)
        topics = list_topics(self.user_id)
        topic_names = [t["name"] for t in topics]
        self.assertIn("sports", topic_names)
        self.assertIn("fitness", topic_names)

    # --- Entities ---

    def test_ingest_stores_entities(self):
        extracted = {
            "facts": [],
            "entities": [make_fact_blob("User has a dog named Rex",
                                         entity_type="pet", tags=["pets"])],
            "topics": [{"topic": "pets", "confidence": 0.7}],
        }
        result = ingest(self.user_id, extracted)
        self.assertEqual(result["facts_stored"], 1)


class TestIngestPipelineIsolation(unittest.TestCase):
    """Test that the pipeline respects memory boundaries."""

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
            port=os.getenv("PG_PORT"),
        )
        cls.cur = cls.conn.cursor()

        # Create two users
        cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;",
                        ("IsolationUser1",))
        cls.user1 = cls.cur.fetchone()[0]
        cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;",
                        ("IsolationUser2",))
        cls.user2 = cls.cur.fetchone()[0]
        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        for uid in [cls.user1, cls.user2]:
            cls.cur.execute("DELETE FROM fact_topics WHERE fact_id IN (SELECT id FROM facts WHERE user_id = %s);", (uid,))
            cls.cur.execute("DELETE FROM facts WHERE user_id = %s;", (uid,))
            cls.cur.execute("DELETE FROM topics WHERE user_id = %s;", (uid,))
            cls.cur.execute("DELETE FROM users WHERE id = %s;", (uid,))
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    def test_facts_isolated_between_users(self):
        """User 1's facts should not appear in user 2's query."""
        extracted = {
            "facts": [make_fact_blob("User1 private fact kappa")],
            "entities": [],
            "topics": [],
        }
        ingest(self.user1, extracted)

        user2_facts = get_facts(self.user2)
        user2_texts = [f[1] for f in user2_facts]
        self.assertFalse(any("User1 private fact kappa" in t for t in user2_texts))

    def test_topics_isolated_between_users(self):
        """User 1's topics should not appear in user 2's topic list."""
        extracted = {
            "facts": [make_fact_blob("Topic isolation test lambda", tags=["user1_only_topic"])],
            "entities": [],
            "topics": [{"topic": "user1_only_topic", "confidence": 0.9}],
        }
        ingest(self.user1, extracted)

        user2_topics = list_topics(self.user2)
        user2_names = [t["name"] for t in user2_topics]
        self.assertNotIn("user1_only_topic", user2_names)


if __name__ == "__main__":
    unittest.main()
