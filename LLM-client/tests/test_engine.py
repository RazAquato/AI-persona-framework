# AI-persona-framework - Engine Test With Emotion Hook
# Copyright (C) 2025 Kenneth Haider

import unittest
from unittest.mock import patch, MagicMock
from core.engine import run_conversation_turn
from memory.persona_emotion_store import DEFAULT_EMOTIONS


# Fake persona dict matching what persona_store.get_persona() returns
FAKE_PERSONA = {
    "id": 1,
    "user_id": 9999,
    "slug": "girlfriend",
    "name": "Maya",
    "description": "Test persona",
    "system_prompt": "You are Maya, a helpful assistant.",
    "nsfw_capable": True,
    "nsfw_prompt_addon": None,
    "nsfw_system_prompt_addon": None,
    "memory_scope": {"tier1": True, "tier2": "all", "tier3": "private"},
    "is_public": False,
}

FAKE_EMOTION_STATE = {
    "emotions": dict(DEFAULT_EMOTIONS),
    "last_updated": None,
    "is_new": True,
}

# Common patches for all engine integration tests
ENGINE_PATCHES = [
    patch("core.engine.get_persona", return_value=FAKE_PERSONA),
    patch("core.engine.load_persona_emotion", return_value=FAKE_EMOTION_STATE),
    patch("core.engine.save_persona_emotion"),
    patch("core.engine.store_emotion_vector"),
    patch("core.engine.get_last_session_for_persona", return_value=None),
    patch("core.engine.start_chat_session", return_value=1),
]


def apply_engine_patches(func):
    """Apply all common engine test patches to a test method."""
    for p in reversed(ENGINE_PATCHES):
        func = p(func)
    return func


class TestEngine(unittest.TestCase):

    @apply_engine_patches
    def test_run_conversation(self, *mocks):
        user_id = 9999
        input_text = "I'm so excited about the future!"
        result = run_conversation_turn(user_id=user_id, user_input=input_text, persona_id=1)

        self.assertIn("assistant_reply", result)
        self.assertIn("session_id", result)
        self.assertGreater(len(result["assistant_reply"]), 0)

    @apply_engine_patches
    def test_result_contains_emotions(self, *mocks):
        """Result should include persona and user emotion dicts."""
        result = run_conversation_turn(user_id=9999, user_input="Hello, how are you?", persona_id=1)

        self.assertIn("persona_emotions", result)
        self.assertIn("user_emotions", result)
        self.assertIn("emotion_description", result)
        self.assertIsInstance(result["persona_emotions"], dict)
        self.assertIsInstance(result["user_emotions"], dict)
        self.assertEqual(len(result["persona_emotions"]), 18)
        self.assertEqual(len(result["user_emotions"]), 18)

    @apply_engine_patches
    def test_persona_id_passed(self, *mocks):
        """Should accept and use persona_id parameter."""
        result = run_conversation_turn(
            user_id=9999, user_input="Hi!", persona_id=1
        )
        self.assertIn("assistant_reply", result)

    @apply_engine_patches
    def test_result_contains_extracted_knowledge(self, *mocks):
        """Result should include extracted_knowledge dict from M2."""
        result = run_conversation_turn(user_id=9999, user_input="My name is Kenneth", persona_id=1)
        self.assertIn("extracted_knowledge", result)
        knowledge = result["extracted_knowledge"]
        self.assertIn("facts", knowledge)
        self.assertIn("entities", knowledge)
        self.assertIn("topics", knowledge)

    @apply_engine_patches
    def test_knowledge_extraction_finds_identity(self, *mocks):
        """Knowledge extractor should find identity facts in user input."""
        result = run_conversation_turn(user_id=9999, user_input="My name is Kenneth and I live in Norway", persona_id=1)
        facts = result["extracted_knowledge"]["facts"]
        self.assertTrue(any("Kenneth" in f["text"] for f in facts))

    @apply_engine_patches
    def test_knowledge_extraction_detects_topics(self, *mocks):
        """Topics should be detected and included in extracted_knowledge."""
        result = run_conversation_turn(
            user_id=9999,
            user_input="I've been coding in python and learning machine learning",
            persona_id=1,
        )
        topics = result["extracted_knowledge"]["topics"]
        topic_names = [t["topic"] for t in topics]
        self.assertIn("technology", topic_names)

    @apply_engine_patches
    def test_process_input_returns_string(self, *mocks):
        """process_input wrapper should return a non-empty string."""
        from core.engine import process_input
        reply = process_input("Hello!", user_id=9999, persona_id=1)
        self.assertIsInstance(reply, str)
        self.assertGreater(len(reply), 0)

    @apply_engine_patches
    def test_result_contains_llm_raw(self, *mocks):
        """Result dict should include llm_raw field."""
        result = run_conversation_turn(user_id=9999, user_input="Hi!", persona_id=1)
        self.assertIn("llm_raw", result)

    def test_missing_persona_raises_error(self):
        """Calling without persona_id should raise ValueError."""
        with self.assertRaises(ValueError):
            run_conversation_turn(user_id=9999, user_input="Hi!")


class TestIncognitoMode(unittest.TestCase):

    @patch("core.engine.get_persona", return_value=FAKE_PERSONA)
    @patch("core.engine.load_persona_emotion", return_value=FAKE_EMOTION_STATE)
    @patch("core.engine.ingest_extracted_knowledge")
    @patch("core.engine.store_fact_blobs")
    @patch("core.engine.save_persona_emotion")
    @patch("core.engine.store_embedding")
    @patch("core.engine.store_emotion_vector")
    @patch("core.engine.log_chat_message")
    @patch("core.engine.start_chat_session", return_value=1)
    @patch("core.engine.get_last_session_for_persona", return_value=None)
    def test_incognito_skips_all_persistence(self, mock_last_persona, mock_start,
                                              mock_log, mock_emo, mock_embed,
                                              mock_save_persona, mock_fact_blobs,
                                              mock_ingest,
                                              mock_load_emo, mock_persona):
        result = run_conversation_turn(
            user_id=9999, user_input="Secret message", incognito=True, persona_id=1
        )
        self.assertIn("assistant_reply", result)
        self.assertTrue(result["incognito"])
        mock_log.assert_not_called()
        mock_emo.assert_not_called()
        mock_embed.assert_not_called()
        mock_save_persona.assert_not_called()
        mock_fact_blobs.assert_not_called()
        mock_ingest.assert_not_called()

    @apply_engine_patches
    def test_nsfw_mode_in_result(self, *mocks):
        result = run_conversation_turn(
            user_id=9999, user_input="Hello", nsfw_mode=True, persona_id=1
        )
        self.assertTrue(result["nsfw_mode"])


if __name__ == "__main__":
    unittest.main()
