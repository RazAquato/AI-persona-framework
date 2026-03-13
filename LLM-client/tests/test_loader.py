# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
import json
from unittest.mock import patch, mock_open
from agents import loader


class TestPersonaLoader(unittest.TestCase):

    def setUp(self):
        self.mock_config = {
            "girlfriend": {
                "name": "Eva",
                "system_prompt": "You are Eva, an affectionate companion...",
                "nsfw_capable": True,
                "nsfw_system_prompt_addon": "You follow creative direction.",
            },
            "debug": {
                "name": "DebugBot",
                "system_prompt": "You are DebugBot.",
                "nsfw_capable": False,
            }
        }

    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.loader.CONFIG_PATH", "mock/path/personality_config.json")
    def test_load_existing_personality(self, mock_file):
        mock_file().read.return_value = json.dumps(self.mock_config)
        result = loader.load_persona_config("debug")
        self.assertEqual(result["name"], "DebugBot")

    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.loader.CONFIG_PATH", "mock/path/personality_config.json")
    def test_load_fallback_to_girlfriend(self, mock_file):
        mock_file().read.return_value = json.dumps(self.mock_config)
        result = loader.load_persona_config("nonexistent")
        self.assertEqual(result["name"], "Eva")

    @patch("agents.loader.CONFIG_PATH", "invalid/path/config.json")
    def test_load_fallback_on_error(self):
        result = loader.load_persona_config("any")
        self.assertEqual(result["name"], "Fallback")
        self.assertIn("helpful", result["system_prompt"])

    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.loader.CONFIG_PATH", "mock/path/personality_config.json")
    def test_memory_scope_default_added(self, mock_file):
        mock_file().read.return_value = json.dumps(self.mock_config)
        result = loader.load_persona_config("girlfriend")
        self.assertIn("memory_scope", result)
        self.assertTrue(result["memory_scope"]["tier1"])
        self.assertEqual(result["memory_scope"]["tier2"], "all")
        self.assertEqual(result["memory_scope"]["tier3"], "private")

    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.loader.CONFIG_PATH", "mock/path/personality_config.json")
    def test_memory_scope_preserved_if_present(self, mock_file):
        config_with_scope = {
            "debug": {
                "name": "DebugBot",
                "system_prompt": "Debug.",
                "memory_scope": {"tier1": True, "tier2": ["technology"], "tier3": "private"}
            }
        }
        mock_file().read.return_value = json.dumps(config_with_scope)
        result = loader.load_persona_config("debug")
        self.assertEqual(result["memory_scope"]["tier2"], ["technology"])

    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.loader.CONFIG_PATH", "mock/path/personality_config.json")
    def test_memory_scope_partial_filled(self, mock_file):
        config_partial = {
            "partial": {
                "name": "Partial",
                "system_prompt": "Partial.",
                "memory_scope": {"tier1": True}
            }
        }
        mock_file().read.return_value = json.dumps(config_partial)
        result = loader.load_persona_config("partial")
        self.assertEqual(result["memory_scope"]["tier2"], "all")
        self.assertEqual(result["memory_scope"]["tier3"], "private")

    @patch("agents.loader.CONFIG_PATH", "invalid/path/config.json")
    def test_fallback_has_memory_scope(self):
        result = loader.load_persona_config("any")
        self.assertIn("memory_scope", result)
        self.assertTrue(result["memory_scope"]["tier1"])

    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.loader.CONFIG_PATH", "mock/path/personality_config.json")
    def test_nsfw_capable_default_false(self, mock_file):
        config_no_nsfw = {
            "plain": {
                "name": "Plain",
                "system_prompt": "Plain assistant.",
            }
        }
        mock_file().read.return_value = json.dumps(config_no_nsfw)
        result = loader.load_persona_config("plain")
        self.assertFalse(result["nsfw_capable"])

    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.loader.CONFIG_PATH", "mock/path/personality_config.json")
    def test_nsfw_capable_preserved(self, mock_file):
        mock_file().read.return_value = json.dumps(self.mock_config)
        result = loader.load_persona_config("girlfriend")
        self.assertTrue(result["nsfw_capable"])

    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.loader.CONFIG_PATH", "mock/path/personality_config.json")
    def test_nsfw_system_prompt_addon_preserved(self, mock_file):
        mock_file().read.return_value = json.dumps(self.mock_config)
        result = loader.load_persona_config("girlfriend")
        self.assertIn("nsfw_system_prompt_addon", result)
        self.assertIn("creative direction", result["nsfw_system_prompt_addon"])


if __name__ == "__main__":
    unittest.main()
