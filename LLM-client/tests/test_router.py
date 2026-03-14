# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
from unittest.mock import patch
from core import router
from memory.persona_emotion_store import DEFAULT_EMOTIONS

FAKE_PERSONA = {
    "id": 1, "user_id": 9999, "slug": "girlfriend", "name": "Maya",
    "description": "Test", "system_prompt": "You are Maya.",
    "nsfw_capable": False, "nsfw_prompt_addon": None,
    "nsfw_system_prompt_addon": None,
    "memory_scope": {"tier1": True, "tier2": "all", "tier3": "private"},
    "is_public": False,
}

FAKE_EMOTION_STATE = {
    "emotions": dict(DEFAULT_EMOTIONS),
    "last_updated": None,
    "is_new": True,
}


class TestRouter(unittest.TestCase):

    @patch("core.engine.start_chat_session", return_value=1)
    @patch("core.engine.get_last_session_for_persona", return_value=None)
    @patch("core.engine.save_persona_emotion")
    @patch("core.engine.load_persona_emotion", return_value=FAKE_EMOTION_STATE)
    @patch("core.engine.get_persona", return_value=FAKE_PERSONA)
    def test_non_tool_input(self, *mocks):
        input_text = "Hello, how are you?"
        response = router.handle_user_input(input_text, persona_id=1)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_valid_tool_command(self):
        input_text = '/generate_image "cat in space"'
        response = router.handle_user_input(input_text)
        self.assertIsInstance(response, str)
        # Result should mention image (success or failure)
        self.assertTrue("image" in response.lower() or "tool" in response.lower())

    def test_invalid_tool_command(self):
        input_text = '/nonexistent_tool "test"'
        response = router.handle_user_input(input_text)
        self.assertIsInstance(response, str)
        self.assertIn("not found", response.lower())

if __name__ == "__main__":
    unittest.main()
