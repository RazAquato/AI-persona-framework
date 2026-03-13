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
from unittest.mock import MagicMock
from tools.tool_call_parser import (
    parse_tool_calls,
    has_tool_calls,
    execute_tool_calls,
    _format_tool_result,
)


class TestParseToolCalls(unittest.TestCase):
    """Test parsing <tool_call> blocks from LLM responses."""

    def test_no_tool_calls(self):
        text = "Happy birthday! I hope you have a wonderful day."
        calls = parse_tool_calls(text)
        self.assertEqual(calls, [])

    def test_single_tool_call(self):
        text = (
            'Happy birthday!\n'
            '<tool_call>\n'
            '{"name": "generate_image", "arguments": {"prompt": "birthday cake"}}\n'
            '</tool_call>'
        )
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "generate_image")
        self.assertEqual(calls[0]["arguments"]["prompt"], "birthday cake")

    def test_multiple_tool_calls(self):
        text = (
            'Here are two images!\n'
            '<tool_call>\n'
            '{"name": "generate_image", "arguments": {"prompt": "flowers"}}\n'
            '</tool_call>\n'
            'And another:\n'
            '<tool_call>\n'
            '{"name": "generate_image", "arguments": {"prompt": "sunset"}}\n'
            '</tool_call>'
        )
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["arguments"]["prompt"], "flowers")
        self.assertEqual(calls[1]["arguments"]["prompt"], "sunset")

    def test_malformed_json(self):
        text = '<tool_call>\nnot valid json\n</tool_call>'
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].get("parse_error"))

    def test_inline_tool_call(self):
        """Tool call on a single line with no extra whitespace."""
        text = '<tool_call>{"name": "generate_image", "arguments": {"prompt": "cat"}}</tool_call>'
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "generate_image")

    def test_tool_call_with_negative_prompt(self):
        text = (
            '<tool_call>\n'
            '{"name": "generate_image", "arguments": '
            '{"prompt": "a rose", "negative_prompt": "thorns"}}\n'
            '</tool_call>'
        )
        calls = parse_tool_calls(text)
        self.assertEqual(calls[0]["arguments"]["negative_prompt"], "thorns")


class TestHasToolCalls(unittest.TestCase):

    def test_has_tool_calls_true(self):
        text = 'blah <tool_call>{"name": "x", "arguments": {}}</tool_call> blah'
        self.assertTrue(has_tool_calls(text))

    def test_has_tool_calls_false(self):
        self.assertFalse(has_tool_calls("just regular text"))

    def test_has_tool_calls_partial_tag(self):
        self.assertFalse(has_tool_calls("<tool_call> but no closing tag"))


class TestExecuteToolCalls(unittest.TestCase):
    """Test full parse → execute → inline flow."""

    def _mock_getter(self, name):
        """Returns a mock tool function for generate_image, None otherwise."""
        if name == "generate_image":
            mock_fn = MagicMock(return_value={
                "success": True,
                "images": ["/output/test_image.png"],
                "error": None,
            })
            return mock_fn
        return None

    def test_no_tool_calls_passthrough(self):
        text = "Just a normal response."
        clean, results = execute_tool_calls(text, self._mock_getter, user_id=9999)
        self.assertEqual(clean, text)
        self.assertEqual(results, [])

    def test_successful_tool_execution(self):
        text = (
            'Happy birthday! Here is a gift for you:\n'
            '<tool_call>\n'
            '{"name": "generate_image", "arguments": {"prompt": "birthday cake with candles"}}\n'
            '</tool_call>\n'
            'I hope you like it!'
        )
        clean, results = execute_tool_calls(text, self._mock_getter, user_id=9999)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].tool_name, "generate_image")
        self.assertIn("/output/test_image.png", clean)
        self.assertNotIn("<tool_call>", clean)
        self.assertIn("Happy birthday", clean)
        self.assertIn("I hope you like it", clean)

    def test_user_context_injected(self):
        """Verify user_id and user_permission are injected into tool args."""
        captured_kwargs = {}

        def mock_tool(**kwargs):
            captured_kwargs.update(kwargs)
            return {"success": True, "images": ["/img.png"], "error": None}

        def getter(name):
            return mock_tool if name == "generate_image" else None

        text = '<tool_call>{"name": "generate_image", "arguments": {"prompt": "test"}}</tool_call>'
        execute_tool_calls(text, getter, user_id=52, user_permission="teen")

        self.assertEqual(captured_kwargs["user_id"], 52)
        self.assertEqual(captured_kwargs["user_permission"], "teen")
        self.assertEqual(captured_kwargs["prompt"], "test")

    def test_unknown_tool_removed(self):
        text = (
            'Let me try this:\n'
            '<tool_call>{"name": "nonexistent_tool", "arguments": {}}</tool_call>\n'
            'Did it work?'
        )
        clean, results = execute_tool_calls(text, self._mock_getter, user_id=9999)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIn("Unknown tool", results[0].error)
        self.assertNotIn("<tool_call>", clean)
        self.assertIn("Let me try this", clean)

    def test_malformed_tool_call_removed(self):
        text = 'Before\n<tool_call>\ngarbage\n</tool_call>\nAfter'
        clean, results = execute_tool_calls(text, self._mock_getter, user_id=9999)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertNotIn("<tool_call>", clean)
        self.assertIn("Before", clean)
        self.assertIn("After", clean)

    def test_tool_execution_exception(self):
        """Tool function raises an exception."""
        def exploding_tool(**kwargs):
            raise RuntimeError("ComfyUI crashed")

        def getter(name):
            return exploding_tool if name == "generate_image" else None

        text = '<tool_call>{"name": "generate_image", "arguments": {"prompt": "test"}}</tool_call>'
        clean, results = execute_tool_calls(text, getter, user_id=9999)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIn("ComfyUI crashed", results[0].error)

    def test_failed_image_generation_silent(self):
        """Failed image gen should not show error text inline."""
        def failing_tool(**kwargs):
            return {"success": False, "images": [], "error": "blocked"}

        def getter(name):
            return failing_tool if name == "generate_image" else None

        text = 'Here you go:\n<tool_call>{"name": "generate_image", "arguments": {"prompt": "x"}}</tool_call>'
        clean, results = execute_tool_calls(text, getter, user_id=9999)
        self.assertNotIn("blocked", clean)
        self.assertNotIn("<tool_call>", clean)


class TestFormatToolResult(unittest.TestCase):

    def test_format_image_success(self):
        result = {"success": True, "images": ["/path/img.png"], "error": None}
        text = _format_tool_result("generate_image", result)
        self.assertIn("[Image: /path/img.png]", text)

    def test_format_image_failure(self):
        result = {"success": False, "images": [], "error": "blocked"}
        text = _format_tool_result("generate_image", result)
        self.assertEqual(text, "")

    def test_format_generic_dict(self):
        result = {"output": "search results here"}
        text = _format_tool_result("web_search", result)
        self.assertIn("search results here", text)

    def test_format_string_result(self):
        text = _format_tool_result("some_tool", "done")
        self.assertIn("done", text)


class TestToolRegistry(unittest.TestCase):
    """Test registry functions used by the engine."""

    def test_get_tool_definitions_structure(self):
        from tools.tool_registry import get_tool_definitions
        defs = get_tool_definitions()
        self.assertIsInstance(defs, list)
        self.assertGreater(len(defs), 0)
        for d in defs:
            self.assertEqual(d["type"], "function")
            self.assertIn("name", d["function"])
            self.assertIn("description", d["function"])
            self.assertIn("parameters", d["function"])

    def test_get_tool_returns_callable(self):
        from tools.tool_registry import get_tool
        fn = get_tool("generate_image")
        self.assertTrue(callable(fn))

    def test_get_tool_returns_none_for_unknown(self):
        from tools.tool_registry import get_tool
        self.assertIsNone(get_tool("nonexistent"))

    def test_describe_tools(self):
        from tools.tool_registry import describe_tools
        descs = describe_tools()
        self.assertIn("generate_image", descs)
        self.assertIsInstance(descs["generate_image"], str)


if __name__ == "__main__":
    unittest.main()
