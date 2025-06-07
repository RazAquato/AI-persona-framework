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
from interface.cli_chat import format_prompt

class TestCLIChat(unittest.TestCase):
    def test_format_prompt_creates_expected_prompt(self):
        persona = {
            "name": "maya",
            "tone": "affectionate",
            "default_emotions": {"love": 0.6, "trust": 0.5}
        }
        history = "User: Hello\nMaya: Hi darling\n"
        user_input = "What's on your mind?"

        result = format_prompt(persona, history, user_input)
        self.assertIn("User: What's on your mind?", result)
        self.assertIn("Emotion state", result)
        self.assertIn("Maya:", result)


if __name__ == "__main__":
    unittest.main()

