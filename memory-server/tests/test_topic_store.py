# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
import os
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from memory.topic_store import (
    get_or_create_topic, get_topic, list_topics,
    link_fact_to_topics, get_fact_topic_ids,
    bump_salience, get_salience, get_persona_salience,
    decay_all_salience, get_salient_fact_ids,
    has_linked_topics, boost_group_salience,
    STICKY_DECAY_FLOOR,
)


class TestTopicStore(unittest.TestCase):

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
        cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("TestTopicUser",))
        cls.user_id = cls.cur.fetchone()[0]
        cls.cur.execute("""
            INSERT INTO user_personalities (user_id, slug, name, system_prompt, memory_scope)
            VALUES (%s, %s, %s, %s, %s) RETURNING id;
        """, (cls.user_id, "test_topic_persona", "TopicPersona", "Test.", Json({"tier1": True})))
        cls.persona_id = cls.cur.fetchone()[0]
        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
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

    # --- Topic CRUD ---

    def test_get_or_create_topic(self):
        tid = get_or_create_topic(self.user_id, "football")
        self.assertIsInstance(tid, int)
        # Same name returns same ID
        tid2 = get_or_create_topic(self.user_id, "football")
        self.assertEqual(tid, tid2)

    def test_topic_name_normalized(self):
        tid1 = get_or_create_topic(self.user_id, "Python")
        tid2 = get_or_create_topic(self.user_id, "python")
        self.assertEqual(tid1, tid2)

    def test_get_topic(self):
        get_or_create_topic(self.user_id, "hiking")
        t = get_topic(self.user_id, "hiking")
        self.assertIsNotNone(t)
        self.assertEqual(t["name"], "hiking")

    def test_get_topic_not_found(self):
        t = get_topic(self.user_id, "nonexistent_topic_xyz")
        self.assertIsNone(t)

    def test_list_topics(self):
        get_or_create_topic(self.user_id, "cooking")
        topics = list_topics(self.user_id)
        names = [t["name"] for t in topics]
        self.assertIn("cooking", names)

    def test_empty_name_returns_none(self):
        tid = get_or_create_topic(self.user_id, "")
        self.assertIsNone(tid)

    # --- Fact-Topic linking ---

    def test_link_fact_to_topics(self):
        # Create a test fact
        self.cur.execute(
            "INSERT INTO facts (user_id, text, tags) VALUES (%s, %s, %s) RETURNING id;",
            (self.user_id, "Test fact for topic linking", ["tech", "gaming"]),
        )
        fact_id = self.cur.fetchone()[0]
        self.conn.commit()

        link_fact_to_topics(self.user_id, fact_id, ["tech", "gaming"])
        topic_ids = get_fact_topic_ids(fact_id)
        self.assertEqual(len(topic_ids), 2)

    def test_link_fact_empty_tags(self):
        link_fact_to_topics(self.user_id, 999999, [])
        link_fact_to_topics(self.user_id, 999999, None)

    def test_has_linked_topics(self):
        self.cur.execute(
            "INSERT INTO facts (user_id, text, tags) VALUES (%s, %s, %s) RETURNING id;",
            (self.user_id, "Linked fact", ["music"]),
        )
        fact_id = self.cur.fetchone()[0]
        self.conn.commit()

        self.assertFalse(has_linked_topics(fact_id))
        link_fact_to_topics(self.user_id, fact_id, ["music"])
        self.assertTrue(has_linked_topics(fact_id))

    # --- Salience ---

    def test_bump_salience_creates_record(self):
        bump_salience(self.user_id, self.persona_id, ["test_bump"])
        s = get_salience(self.user_id, self.persona_id,
                         get_or_create_topic(self.user_id, "test_bump"))
        self.assertIsNotNone(s)
        self.assertGreater(s["salience"], 0)
        self.assertEqual(s["mention_count"], 1)

    def test_bump_salience_increases(self):
        bump_salience(self.user_id, self.persona_id, ["test_increase"])
        tid = get_or_create_topic(self.user_id, "test_increase")
        s1 = get_salience(self.user_id, self.persona_id, tid)["salience"]
        bump_salience(self.user_id, self.persona_id, ["test_increase"])
        s2 = get_salience(self.user_id, self.persona_id, tid)["salience"]
        self.assertGreater(s2, s1)

    def test_bump_salience_caps_at_one(self):
        for _ in range(100):
            bump_salience(self.user_id, self.persona_id, ["test_cap"])
        tid = get_or_create_topic(self.user_id, "test_cap")
        s = get_salience(self.user_id, self.persona_id, tid)
        self.assertLessEqual(s["salience"], 1.0)

    def test_bump_salience_diminishing_returns(self):
        bump_salience(self.user_id, self.persona_id, ["test_diminish"])
        tid = get_or_create_topic(self.user_id, "test_diminish")
        s1 = get_salience(self.user_id, self.persona_id, tid)["salience"]
        bump_salience(self.user_id, self.persona_id, ["test_diminish"])
        s2 = get_salience(self.user_id, self.persona_id, tid)["salience"]
        delta1 = s2 - s1
        bump_salience(self.user_id, self.persona_id, ["test_diminish"])
        s3 = get_salience(self.user_id, self.persona_id, tid)["salience"]
        delta2 = s3 - s2
        # Each bump should be smaller than the last
        self.assertLess(delta2, delta1)

    def test_sticky_topic_gets_decay_floor(self):
        bump_salience(self.user_id, self.persona_id, ["family"])
        tid = get_or_create_topic(self.user_id, "family")
        s = get_salience(self.user_id, self.persona_id, tid)
        self.assertEqual(s["decay_floor"], STICKY_DECAY_FLOOR)

    def test_non_sticky_topic_zero_floor(self):
        bump_salience(self.user_id, self.persona_id, ["random_topic_xyz"])
        tid = get_or_create_topic(self.user_id, "random_topic_xyz")
        s = get_salience(self.user_id, self.persona_id, tid)
        self.assertEqual(s["decay_floor"], 0.0)

    def test_get_persona_salience(self):
        bump_salience(self.user_id, self.persona_id, ["persona_sal_test"])
        records = get_persona_salience(self.user_id, self.persona_id)
        names = [r["topic_name"] for r in records]
        self.assertIn("persona_sal_test", names)

    def test_get_persona_salience_min_filter(self):
        bump_salience(self.user_id, self.persona_id, ["low_sal_test"])
        # One bump gives salience ~0.1, so threshold 0.5 should exclude it
        records = get_persona_salience(self.user_id, self.persona_id, min_salience=0.5)
        names = [r["topic_name"] for r in records]
        self.assertNotIn("low_sal_test", names)

    def test_bump_multiple_topics(self):
        bump_salience(self.user_id, self.persona_id, ["multi_a", "multi_b", "multi_c"])
        records = get_persona_salience(self.user_id, self.persona_id)
        names = [r["topic_name"] for r in records]
        self.assertIn("multi_a", names)
        self.assertIn("multi_b", names)
        self.assertIn("multi_c", names)

    # --- Decay ---

    def test_decay_all_salience(self):
        bump_salience(self.user_id, self.persona_id, ["decay_test"])
        tid = get_or_create_topic(self.user_id, "decay_test")
        before = get_salience(self.user_id, self.persona_id, tid)["salience"]

        # Force last_mentioned to 10 days ago
        self.cur.execute("""
            UPDATE topic_salience SET last_mentioned = CURRENT_TIMESTAMP - INTERVAL '10 days'
            WHERE user_id = %s AND persona_id = %s AND topic_id = %s;
        """, (self.user_id, self.persona_id, tid))
        self.conn.commit()

        decay_all_salience()
        after = get_salience(self.user_id, self.persona_id, tid)["salience"]
        self.assertLess(after, before)

    def test_decay_respects_floor(self):
        # Bump multiple times to get salience above the decay floor (0.15)
        for _ in range(5):
            bump_salience(self.user_id, self.persona_id, ["children"])  # sticky topic
        tid = get_or_create_topic(self.user_id, "children")

        # Force last_mentioned to 1000 days ago
        self.cur.execute("""
            UPDATE topic_salience SET last_mentioned = CURRENT_TIMESTAMP - INTERVAL '1000 days'
            WHERE user_id = %s AND persona_id = %s AND topic_id = %s;
        """, (self.user_id, self.persona_id, tid))
        self.conn.commit()

        decay_all_salience()
        after = get_salience(self.user_id, self.persona_id, tid)["salience"]
        self.assertGreaterEqual(after, STICKY_DECAY_FLOOR)

    # --- Salient fact IDs ---

    def test_get_salient_fact_ids(self):
        # Create fact and link to a topic with high salience
        self.cur.execute(
            "INSERT INTO facts (user_id, text, tags) VALUES (%s, %s, %s) RETURNING id;",
            (self.user_id, "Salient fact test", ["salient_topic"]),
        )
        fact_id = self.cur.fetchone()[0]
        self.conn.commit()

        link_fact_to_topics(self.user_id, fact_id, ["salient_topic"])
        # Bump salience high
        for _ in range(10):
            bump_salience(self.user_id, self.persona_id, ["salient_topic"])

        salient_ids = get_salient_fact_ids(self.user_id, self.persona_id, min_salience=0.2)
        self.assertIn(fact_id, salient_ids)

    def test_low_salience_fact_excluded(self):
        self.cur.execute(
            "INSERT INTO facts (user_id, text, tags) VALUES (%s, %s, %s) RETURNING id;",
            (self.user_id, "Low salience fact", ["low_sal_topic"]),
        )
        fact_id = self.cur.fetchone()[0]
        self.conn.commit()

        link_fact_to_topics(self.user_id, fact_id, ["low_sal_topic"])
        bump_salience(self.user_id, self.persona_id, ["low_sal_topic"])
        # One bump = ~0.1 salience, threshold 0.5 should exclude
        salient_ids = get_salient_fact_ids(self.user_id, self.persona_id, min_salience=0.5)
        self.assertNotIn(fact_id, salient_ids)

    # --- Group salience boost ---

    def test_boost_group_salience(self):
        # Create a group, session, and message with topics
        self.cur.execute(
            "INSERT INTO session_groups (user_id, persona_id, name) VALUES (%s, %s, %s) RETURNING id;",
            (self.user_id, self.persona_id, "Boost Test"),
        )
        group_id = self.cur.fetchone()[0]
        self.cur.execute(
            "INSERT INTO chat_sessions (user_id, persona_id, group_id) VALUES (%s, %s, %s) RETURNING id;",
            (self.user_id, self.persona_id, group_id),
        )
        session_id = self.cur.fetchone()[0]
        self.cur.execute(
            "INSERT INTO chat_messages (session_id, user_id, role, content, topics) VALUES (%s, %s, %s, %s, %s);",
            (session_id, self.user_id, "user", "group test", ["group_topic_a", "group_topic_b"]),
        )
        self.conn.commit()

        count = boost_group_salience(self.user_id, self.persona_id, group_id)
        self.assertGreater(count, 0)

        # Check salience was created
        tid = get_or_create_topic(self.user_id, "group_topic_a")
        s = get_salience(self.user_id, self.persona_id, tid)
        self.assertIsNotNone(s)
        self.assertGreater(s["salience"], 0)


if __name__ == "__main__":
    unittest.main()
