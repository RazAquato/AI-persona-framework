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
from core.engine import run_conversation_turn

class TestEngine(unittest.TestCase):

    def test_run_conversation(self):
        user_id = 9999  # Use a test user ID
        input_text = "What's the weather like today?"
        result = run_conversation_turn(user_id=user_id, user_input=input_text)

        self.assertIn("assistant_reply", result)
        self.assertIsInstance(result["assistant_reply"], str)
        self.assertGreater(len(result["assistant_reply"]), 0)
        self.assertIn("session_id", result)

if __name__ == "__main__":
    unittest.main()

