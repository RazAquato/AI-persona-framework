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


if __name__ == "__main__":
    unittest.main()
