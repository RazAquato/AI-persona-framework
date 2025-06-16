# AI-persona-framework - Engine Test With Emotion Hook
# Copyright (C) 2025 Kenneth Haider

import unittest
from unittest.mock import patch
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

if __name__ == "__main__":
    unittest.main()
