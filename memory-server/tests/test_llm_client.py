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
from core.llm_client import call_llm


class TestLLMClient(unittest.TestCase):

    def setUp(self):
        self.messages = [
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": "Hello! What's your name?"}
        ]

    def test_llm_basic_response(self):
        response = call_llm(self.messages)
        self.assertIn("content", response)
        self.assertIsInstance(response["content"], str)
        self.assertGreater(len(response["content"]), 0)

    def test_llm_with_tool_stub(self):
        tools = [
            {
                "name": "get_time",
                "description": "Returns the current time.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]

        response = call_llm(self.messages, tools=tools)
        self.assertIn("content", response)
        self.assertIsInstance(response["content"], str)
        self.assertGreaterEqual(len(response["content"]), 1)


if __name__ == "__main__":
    unittest.main()

