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
        # max_tokens must be large enough for thinking-mode models (Qwen3.5)
        # which use tokens for <think> reasoning before producing content
        response = call_llm(self.messages, max_tokens=2048)
        self.assertIn("content", response)
        self.assertIsInstance(response["content"], str)
        # Model must produce content or reasoning (thinking models may put output in reasoning)
        has_output = len(response["content"]) > 0 or len(response.get("reasoning", "")) > 0
        self.assertTrue(has_output, "Model produced neither content nor reasoning")

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

        response = call_llm(self.messages, tools=tools, max_tokens=2048)
        self.assertIn("content", response)
        self.assertIsInstance(response["content"], str)
        has_output = len(response["content"]) > 0 or len(response.get("reasoning", "")) > 0
        self.assertTrue(has_output, "Model produced neither content nor reasoning")


if __name__ == "__main__":
    unittest.main()

