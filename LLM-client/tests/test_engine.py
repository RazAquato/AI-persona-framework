# AI-persona-framework - Engine Test With Emotion Hook
# Copyright (C) 2025 Kenneth Haider

import unittest
from unittest.mock import patch, MagicMock
from core.engine import run_conversation_turn


class TestEngine(unittest.TestCase):

    @patch("core.engine.store_emotion_vector")
    def test_run_conversation(self, mock_store_emotion):
        user_id = 9999
        input_text = "I'm so excited about the future!"
        result = run_conversation_turn(user_id=user_id, user_input=input_text)

        self.assertIn("assistant_reply", result)
        self.assertIn("session_id", result)
        self.assertGreater(len(result["assistant_reply"]), 0)

        # Emotion vector storage should be triggered
        mock_store_emotion.assert_called_once()

    @patch("core.engine.store_emotion_vector")
    def test_result_contains_emotions(self, mock_store_emotion):
        """Result should include persona and user emotion dicts."""
        result = run_conversation_turn(user_id=9999, user_input="Hello, how are you?")

        self.assertIn("persona_emotions", result)
        self.assertIn("user_emotions", result)
        self.assertIn("emotion_description", result)
        self.assertIsInstance(result["persona_emotions"], dict)
        self.assertIsInstance(result["user_emotions"], dict)
        self.assertEqual(len(result["persona_emotions"]), 18)
        self.assertEqual(len(result["user_emotions"]), 18)

    @patch("core.engine.store_emotion_vector")
    def test_personality_id_passed(self, mock_store_emotion):
        """Should accept and use personality_id parameter."""
        result = run_conversation_turn(
            user_id=9999, user_input="Hi!", personality_id="default"
        )
        self.assertIn("assistant_reply", result)

    @patch("core.engine.store_emotion_vector")
    def test_result_contains_extracted_knowledge(self, mock_store_emotion):
        """Result should include extracted_knowledge dict from M2."""
        result = run_conversation_turn(user_id=9999, user_input="My name is Kenneth")
        self.assertIn("extracted_knowledge", result)
        knowledge = result["extracted_knowledge"]
        self.assertIn("facts", knowledge)
        self.assertIn("entities", knowledge)
        self.assertIn("topics", knowledge)
        self.assertIn("classification", knowledge)

    @patch("core.engine.store_emotion_vector")
    def test_knowledge_extraction_finds_identity(self, mock_store_emotion):
        """Knowledge extractor should find identity facts in user input."""
        result = run_conversation_turn(user_id=9999, user_input="My name is Kenneth and I live in Norway")
        facts = result["extracted_knowledge"]["facts"]
        self.assertTrue(any("Kenneth" in f["text"] for f in facts))


if __name__ == "__main__":
    unittest.main()
