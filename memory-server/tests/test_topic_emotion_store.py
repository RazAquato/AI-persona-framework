# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
import os
import psycopg2
from dotenv import load_dotenv
from memory.topic_store import get_or_create_topic
from memory.topic_emotion_store import (
    set_topic_emotion, set_topic_emotions,
    get_topic_emotions, get_user_topic_emotions,
    get_emotions_for_topics, delete_topic_emotion,
    decay_topic_emotions, VALID_EMOTIONS, BLEND_WEIGHT,
)


class TestTopicEmotionStore(unittest.TestCase):

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
                        ("TopicEmotionTestUser",))
        cls.user_id = cls.cur.fetchone()[0]
        cls.conn.commit()

        # Create test topics
        cls.topic_football = get_or_create_topic(cls.user_id, "football")
        cls.topic_cooking = get_or_create_topic(cls.user_id, "cooking")
        cls.topic_work = get_or_create_topic(cls.user_id, "work_stress")

    @classmethod
    def tearDownClass(cls):
        cls.cur.execute("DELETE FROM user_topic_emotions WHERE user_id = %s;",
                        (cls.user_id,))
        cls.cur.execute("DELETE FROM topic_salience WHERE user_id = %s;",
                        (cls.user_id,))
        cls.cur.execute("DELETE FROM fact_topics WHERE fact_id IN (SELECT id FROM facts WHERE user_id = %s);",
                        (cls.user_id,))
        cls.cur.execute("DELETE FROM facts WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM topics WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM users WHERE id = %s;", (cls.user_id,))
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    # --- Basic CRUD ---

    def test_set_topic_emotion(self):
        eid = set_topic_emotion(self.user_id, self.topic_football, "joy", 0.7)
        self.assertIsNotNone(eid)

    def test_set_emotion_invalid_name(self):
        eid = set_topic_emotion(self.user_id, self.topic_football, "bliss", 0.5)
        self.assertIsNone(eid)

    def test_set_emotion_clamps_intensity(self):
        eid = set_topic_emotion(self.user_id, self.topic_football, "pride", 1.5)
        self.assertIsNotNone(eid)
        emotions = get_topic_emotions(self.user_id, self.topic_football)
        pride = [e for e in emotions if e["emotion"] == "pride"]
        self.assertTrue(all(e["intensity"] <= 1.0 for e in pride))

    def test_get_topic_emotions(self):
        set_topic_emotion(self.user_id, self.topic_cooking, "joy", 0.6)
        set_topic_emotion(self.user_id, self.topic_cooking, "calm", 0.4)
        emotions = get_topic_emotions(self.user_id, self.topic_cooking)
        names = [e["emotion"] for e in emotions]
        self.assertIn("joy", names)
        self.assertIn("calm", names)

    def test_get_topic_emotions_min_intensity(self):
        set_topic_emotion(self.user_id, self.topic_cooking, "interest", 0.1)
        emotions = get_topic_emotions(self.user_id, self.topic_cooking,
                                       min_intensity=0.3)
        names = [e["emotion"] for e in emotions]
        self.assertNotIn("interest", names)

    def test_multiple_emotions_per_topic(self):
        """A topic can have love AND frustration simultaneously."""
        set_topic_emotion(self.user_id, self.topic_work, "anger", 0.6)
        set_topic_emotion(self.user_id, self.topic_work, "sadness", 0.4)
        set_topic_emotion(self.user_id, self.topic_work, "hope", 0.3)
        emotions = get_topic_emotions(self.user_id, self.topic_work)
        self.assertGreaterEqual(len(emotions), 3)

    # --- Blending ---

    def test_upsert_blends_intensity(self):
        """Re-setting an emotion should blend, not replace."""
        set_topic_emotion(self.user_id, self.topic_football, "love", 0.5)
        emotions_before = get_topic_emotions(self.user_id, self.topic_football)
        love_before = [e for e in emotions_before if e["emotion"] == "love"][0]

        # Second set should blend
        set_topic_emotion(self.user_id, self.topic_football, "love", 0.9)
        emotions_after = get_topic_emotions(self.user_id, self.topic_football)
        love_after = [e for e in emotions_after if e["emotion"] == "love"][0]

        # Should be between old and new, not exactly 0.9
        self.assertNotEqual(love_after["intensity"], 0.9)
        self.assertGreater(love_after["intensity"], love_before["intensity"])

    # --- Bulk operations ---

    def test_set_topic_emotions_bulk(self):
        emotions = [
            {"emotion": "joy", "intensity": 0.7},
            {"emotion": "pride", "intensity": 0.5},
        ]
        ids = set_topic_emotions(self.user_id, self.topic_cooking, emotions)
        self.assertEqual(len(ids), 2)

    def test_get_user_topic_emotions(self):
        set_topic_emotion(self.user_id, self.topic_football, "curiosity", 0.6)
        all_emotions = get_user_topic_emotions(self.user_id, min_intensity=0.1)
        self.assertIsInstance(all_emotions, list)
        # Should contain topic names
        topic_names = [e["topic_name"] for e in all_emotions]
        self.assertIn("football", topic_names)

    def test_get_emotions_for_topics(self):
        set_topic_emotion(self.user_id, self.topic_football, "trust", 0.5)
        set_topic_emotion(self.user_id, self.topic_cooking, "calm", 0.5)
        result = get_emotions_for_topics(
            self.user_id,
            [self.topic_football, self.topic_cooking],
            min_intensity=0.1,
        )
        self.assertIn(self.topic_football, result)
        self.assertIn(self.topic_cooking, result)

    def test_get_emotions_for_empty_topics(self):
        result = get_emotions_for_topics(self.user_id, [])
        self.assertEqual(result, {})

    # --- Delete ---

    def test_delete_topic_emotion(self):
        set_topic_emotion(self.user_id, self.topic_work, "guilt", 0.3)
        emotions_before = get_topic_emotions(self.user_id, self.topic_work)
        guilt_before = [e for e in emotions_before if e["emotion"] == "guilt"]
        self.assertTrue(len(guilt_before) > 0)

        delete_topic_emotion(self.user_id, self.topic_work, "guilt")
        emotions_after = get_topic_emotions(self.user_id, self.topic_work)
        guilt_after = [e for e in emotions_after if e["emotion"] == "guilt"]
        self.assertEqual(len(guilt_after), 0)

    # --- Decay ---

    def test_decay_topic_emotions(self):
        # Set a fresh emotion with known intensity
        set_topic_emotion(self.user_id, self.topic_football, "fear", 0.8)
        before = get_topic_emotions(self.user_id, self.topic_football)
        fear_before = [e for e in before if e["emotion"] == "fear"][0]["intensity"]

        decayed, removed = decay_topic_emotions(decay_rate=0.5)
        after = get_topic_emotions(self.user_id, self.topic_football)
        fear_after = [e for e in after if e["emotion"] == "fear"]
        if fear_after:
            self.assertLess(fear_after[0]["intensity"], fear_before)

    def test_decay_removes_faded_emotions(self):
        """Emotions below threshold should be removed on decay."""
        set_topic_emotion(self.user_id, self.topic_work, "shame", 0.06)
        decayed, removed = decay_topic_emotions(decay_rate=0.5, min_intensity=0.05)
        after = get_topic_emotions(self.user_id, self.topic_work)
        shame = [e for e in after if e["emotion"] == "shame"]
        self.assertEqual(len(shame), 0)


class TestTopicEmotionPromptFormatting(unittest.TestCase):
    """Test the prompt builder's topic emotion formatting."""

    def test_format_positive_emotions(self):
        from core.prompt_builder import _format_topic_emotions
        emotions = [
            {"topic_name": "cooking", "emotion": "joy", "intensity": 0.7},
            {"topic_name": "cooking", "emotion": "calm", "intensity": 0.5},
        ]
        lines = _format_topic_emotions(emotions)
        self.assertEqual(len(lines), 1)
        self.assertIn("cooking", lines[0])
        self.assertIn("safe to reference", lines[0])

    def test_format_negative_emotions(self):
        from core.prompt_builder import _format_topic_emotions
        emotions = [
            {"topic_name": "work", "emotion": "anger", "intensity": 0.6},
            {"topic_name": "work", "emotion": "sadness", "intensity": 0.4},
        ]
        lines = _format_topic_emotions(emotions)
        self.assertEqual(len(lines), 1)
        self.assertIn("work", lines[0])
        self.assertIn("empathy", lines[0])

    def test_format_mixed_emotions(self):
        from core.prompt_builder import _format_topic_emotions
        emotions = [
            {"topic_name": "family", "emotion": "love", "intensity": 0.8},
            {"topic_name": "family", "emotion": "sadness", "intensity": 0.3},
        ]
        lines = _format_topic_emotions(emotions)
        self.assertEqual(len(lines), 1)
        self.assertIn("mixed", lines[0])

    def test_format_empty_list(self):
        from core.prompt_builder import _format_topic_emotions
        lines = _format_topic_emotions([])
        self.assertEqual(lines, [])

    def test_format_multiple_topics(self):
        from core.prompt_builder import _format_topic_emotions
        emotions = [
            {"topic_name": "cooking", "emotion": "joy", "intensity": 0.7},
            {"topic_name": "work", "emotion": "anger", "intensity": 0.6},
        ]
        lines = _format_topic_emotions(emotions)
        self.assertEqual(len(lines), 2)

    def test_prompt_includes_topic_emotions(self):
        from core.prompt_builder import build_system_prompt
        persona = {"system_prompt": "You are a test assistant."}
        topic_emotions = [
            {"topic_name": "football", "emotion": "joy", "intensity": 0.8},
        ]
        prompt = build_system_prompt(persona, topic_emotions=topic_emotions)
        self.assertIn("How This User Feels", prompt)
        self.assertIn("football", prompt)

    def test_prompt_no_topic_emotions_when_empty(self):
        from core.prompt_builder import build_system_prompt
        persona = {"system_prompt": "You are a test assistant."}
        prompt = build_system_prompt(persona, topic_emotions=[])
        self.assertNotIn("How This User Feels", prompt)


class TestNightlyReflection(unittest.TestCase):
    """Test the nightly reflection script components."""

    def test_regex_fallback_returns_valid_emotions(self):
        from scripts.nightly_reflection import _regex_fallback
        msgs = [
            {"role": "user", "content": "I absolutely love cooking! It makes me so happy and peaceful."},
        ]
        emotions = _regex_fallback("cooking", msgs)
        self.assertIsInstance(emotions, list)
        for em in emotions:
            self.assertIn("emotion", em)
            self.assertIn("intensity", em)
            self.assertIn(em["emotion"], VALID_EMOTIONS)

    def test_regex_fallback_empty_input(self):
        from scripts.nightly_reflection import _regex_fallback
        emotions = _regex_fallback("nothing", [])
        self.assertIsInstance(emotions, list)

    def test_classify_short_messages_uses_fallback(self):
        """Fewer than 2 user messages should use regex fallback."""
        from scripts.nightly_reflection import classify_topic_emotions
        msgs = [{"role": "user", "content": "hi"}]
        emotions = classify_topic_emotions("test", msgs)
        self.assertIsInstance(emotions, list)


if __name__ == "__main__":
    unittest.main()
