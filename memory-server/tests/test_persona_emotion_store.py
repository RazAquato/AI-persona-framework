# AI-persona-framework - Persona Emotion Store Unit Tests
# Copyright (C) 2025 Kenneth Haider

import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from memory.persona_emotion_store import (
    load_persona_emotion, save_persona_emotion,
    get_all_persona_emotions, DEFAULT_EMOTIONS,
)


class TestPersonaEmotionStore(unittest.TestCase):
    """
    Tests hit the live PostgreSQL database (test user 9999).
    Uses persona name 'test_persona_unit' to avoid colliding with real data.
    """

    TEST_USER = 9999
    TEST_PERSONA = "test_persona_unit"

    def test_01_load_new_returns_default(self):
        """Loading a nonexistent persona should return defaults with is_new=True."""
        state = load_persona_emotion(self.TEST_USER, "nonexistent_persona_xyz")
        self.assertTrue(state["is_new"])
        self.assertIsNone(state["last_updated"])
        self.assertEqual(set(state["emotions"].keys()), set(DEFAULT_EMOTIONS.keys()))

    def test_02_save_and_load(self):
        """Saving and reloading should return the same emotions."""
        emotions = dict(DEFAULT_EMOTIONS)
        emotions["love"] = 0.75
        emotions["joy"] = 0.6

        save_persona_emotion(self.TEST_USER, self.TEST_PERSONA, emotions)
        state = load_persona_emotion(self.TEST_USER, self.TEST_PERSONA)

        self.assertFalse(state["is_new"])
        self.assertIsNotNone(state["last_updated"])
        self.assertAlmostEqual(state["emotions"]["love"], 0.75, places=2)
        self.assertAlmostEqual(state["emotions"]["joy"], 0.6, places=2)

    def test_03_upsert_overwrites(self):
        """Saving again should update, not create duplicate."""
        emotions_v1 = dict(DEFAULT_EMOTIONS)
        emotions_v1["anger"] = 0.5
        save_persona_emotion(self.TEST_USER, self.TEST_PERSONA, emotions_v1)

        emotions_v2 = dict(DEFAULT_EMOTIONS)
        emotions_v2["anger"] = 0.1
        save_persona_emotion(self.TEST_USER, self.TEST_PERSONA, emotions_v2)

        state = load_persona_emotion(self.TEST_USER, self.TEST_PERSONA)
        self.assertAlmostEqual(state["emotions"]["anger"], 0.1, places=2)

    def test_04_get_all_personas(self):
        """get_all_persona_emotions should return at least our test persona."""
        # Ensure at least one exists
        save_persona_emotion(self.TEST_USER, self.TEST_PERSONA, DEFAULT_EMOTIONS)
        all_states = get_all_persona_emotions(self.TEST_USER)
        self.assertIsInstance(all_states, list)
        persona_ids = [s["personality_id"] for s in all_states]
        self.assertIn(self.TEST_PERSONA, persona_ids)

    def test_05_different_users_independent(self):
        """Different users should have independent persona emotions."""
        emotions_a = dict(DEFAULT_EMOTIONS)
        emotions_a["love"] = 0.9
        save_persona_emotion(self.TEST_USER, self.TEST_PERSONA, emotions_a)

        # User 9998 should get defaults (assuming no prior data)
        state_b = load_persona_emotion(9998, self.TEST_PERSONA)
        # If 9998 has no record, it's new with defaults
        if state_b["is_new"]:
            self.assertAlmostEqual(state_b["emotions"]["love"], DEFAULT_EMOTIONS["love"], places=2)


if __name__ == "__main__":
    unittest.main()
