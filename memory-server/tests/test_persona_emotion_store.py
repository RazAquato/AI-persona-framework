# AI-persona-framework - Persona Emotion Store Unit Tests
# Copyright (C) 2025 Kenneth Haider

import unittest
import os
import sys
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from memory.persona_emotion_store import (
    load_persona_emotion, save_persona_emotion,
    get_all_persona_emotions, DEFAULT_EMOTIONS,
)


class TestPersonaEmotionStore(unittest.TestCase):
    """
    Tests hit the live PostgreSQL database.
    Creates a temporary user and persona for testing, cleaned up in tearDownClass.
    """

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

        # Create test user
        cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("TestEmotionUser",))
        cls.user_id = cls.cur.fetchone()[0]

        # Create test persona
        cls.cur.execute("""
            INSERT INTO user_personalities (user_id, slug, name, system_prompt, memory_scope)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """, (cls.user_id, "test_emo", "TestEmoPersona", "Test.", Json({"tier1": True, "tier2": "all", "tier3": "private"})))
        cls.persona_id = cls.cur.fetchone()[0]

        # Second user for independence test
        cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("TestEmotionUser2",))
        cls.user_id_2 = cls.cur.fetchone()[0]
        cls.cur.execute("""
            INSERT INTO user_personalities (user_id, slug, name, system_prompt, memory_scope)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """, (cls.user_id_2, "test_emo", "TestEmoPersona2", "Test.", Json({"tier1": True, "tier2": "all", "tier3": "private"})))
        cls.persona_id_2 = cls.cur.fetchone()[0]

        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        # Clean up: emotion_history → emotional_relationships → sessions → personas → users
        cls.cur.execute("""
            DELETE FROM emotion_history WHERE relationship_id IN (
                SELECT id FROM emotional_relationships WHERE user_id IN (%s, %s)
            );
        """, (cls.user_id, cls.user_id_2))
        cls.cur.execute("DELETE FROM emotional_relationships WHERE user_id IN (%s, %s);", (cls.user_id, cls.user_id_2))
        cls.cur.execute("DELETE FROM user_personalities WHERE user_id IN (%s, %s);", (cls.user_id, cls.user_id_2))
        cls.cur.execute("DELETE FROM users WHERE id IN (%s, %s);", (cls.user_id, cls.user_id_2))
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    def test_01_load_new_returns_default(self):
        """Loading a nonexistent persona should return defaults with is_new=True."""
        # Use a persona_id that doesn't have an emotional_relationship yet
        state = load_persona_emotion(self.user_id_2, self.persona_id_2)
        self.assertTrue(state["is_new"])
        self.assertIsNone(state["last_updated"])
        self.assertEqual(set(state["emotions"].keys()), set(DEFAULT_EMOTIONS.keys()))

    def test_02_save_and_load(self):
        """Saving and reloading should return the same emotions."""
        emotions = dict(DEFAULT_EMOTIONS)
        emotions["love"] = 0.75
        emotions["joy"] = 0.6

        save_persona_emotion(self.user_id, self.persona_id, emotions)
        state = load_persona_emotion(self.user_id, self.persona_id)

        self.assertFalse(state["is_new"])
        self.assertIsNotNone(state["last_updated"])
        self.assertAlmostEqual(state["emotions"]["love"], 0.75, places=2)
        self.assertAlmostEqual(state["emotions"]["joy"], 0.6, places=2)

    def test_03_upsert_overwrites(self):
        """Saving again should update, not create duplicate."""
        emotions_v1 = dict(DEFAULT_EMOTIONS)
        emotions_v1["anger"] = 0.5
        save_persona_emotion(self.user_id, self.persona_id, emotions_v1)

        emotions_v2 = dict(DEFAULT_EMOTIONS)
        emotions_v2["anger"] = 0.1
        save_persona_emotion(self.user_id, self.persona_id, emotions_v2)

        state = load_persona_emotion(self.user_id, self.persona_id)
        self.assertAlmostEqual(state["emotions"]["anger"], 0.1, places=2)

    def test_04_get_all_personas(self):
        """get_all_persona_emotions should return at least our test persona."""
        save_persona_emotion(self.user_id, self.persona_id, DEFAULT_EMOTIONS)
        all_states = get_all_persona_emotions(self.user_id)
        self.assertIsInstance(all_states, list)
        persona_ids = [s["persona_id"] for s in all_states]
        self.assertIn(self.persona_id, persona_ids)

    def test_05_different_users_independent(self):
        """Different users should have independent persona emotions."""
        emotions_a = dict(DEFAULT_EMOTIONS)
        emotions_a["love"] = 0.9
        save_persona_emotion(self.user_id, self.persona_id, emotions_a)

        # User 2 should get defaults (no record yet for their persona)
        state_b = load_persona_emotion(self.user_id_2, self.persona_id_2)
        if state_b["is_new"]:
            self.assertAlmostEqual(state_b["emotions"]["love"], DEFAULT_EMOTIONS["love"], places=2)

    def test_06_emotion_history_appended(self):
        """Each save should append a row to emotion_history."""
        emotions = dict(DEFAULT_EMOTIONS)
        emotions["trust"] = 0.8
        save_persona_emotion(self.user_id, self.persona_id, emotions)

        # Check emotion_history has at least one entry
        self.cur.execute("""
            SELECT COUNT(*) FROM emotion_history
            WHERE relationship_id = (
                SELECT id FROM emotional_relationships
                WHERE user_id = %s AND persona_id = %s
            );
        """, (self.user_id, self.persona_id))
        count = self.cur.fetchone()[0]
        self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()
