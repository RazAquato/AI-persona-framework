# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import unittest
import os
import json
from unittest.mock import patch, mock_open
from agents import loader


class TestPersonaLoader(unittest.TestCase):

    def setUp(self):
        self.mock_config = {
            "default": {
                "name": "Eva",
                "system_prompt": "You are Eva, a poetic and thoughtful assistant..."
            },
            "friendly": {
                "name": "Bob",
                "system_prompt": "You are Bob, a cheerful and curious assistant."
            }
        }

    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.loader.CONFIG_PATH", "mock/path/personality_config.json")
    def test_load_existing_personality(self, mock_file):
        mock_file().read.return_value = json.dumps(self.mock_config)
        result = loader.load_persona_config("friendly")
        self.assertEqual(result["name"], "Bob")
        self.assertIn("cheerful", result["system_prompt"])

    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.loader.CONFIG_PATH", "mock/path/personality_config.json")
    def test_load_fallback_to_default(self, mock_file):
        mock_file().read.return_value = json.dumps(self.mock_config)
        result = loader.load_persona_config("nonexistent")
        self.assertEqual(result["name"], "Eva")

    @patch("agents.loader.CONFIG_PATH", "invalid/path/config.json")
    def test_load_fallback_on_error(self):
        result = loader.load_persona_config("any")
        self.assertEqual(result["name"], "Fallback")
        self.assertIn("helpful", result["system_prompt"])


if __name__ == "__main__":
    unittest.main()

