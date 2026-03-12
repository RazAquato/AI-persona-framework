# AI-persona-framework - Conversation Buffer Unit Tests
# Copyright (C) 2025 Kenneth Haider

import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from memory.buffer import ConversationBuffer


class TestConversationBuffer(unittest.TestCase):

    def setUp(self):
        self.buffer = ConversationBuffer(max_turns=5)

    def test_add_and_get(self):
        """Basic add and retrieve."""
        self.buffer.add_message(1, 100, "user", "Hello")
        self.buffer.add_message(1, 100, "assistant", "Hi there")
        msgs = self.buffer.get_messages(1, 100)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["content"], "Hi there")

    def test_eviction(self):
        """Buffer should evict oldest when over max_turns."""
        for i in range(7):
            self.buffer.add_message(1, 100, "user", f"msg-{i}")
        msgs = self.buffer.get_messages(1, 100)
        self.assertEqual(len(msgs), 5)
        self.assertEqual(msgs[0]["content"], "msg-2")  # oldest 2 evicted
        self.assertEqual(msgs[-1]["content"], "msg-6")

    def test_limit_parameter(self):
        """get_messages with limit should return only last N."""
        for i in range(5):
            self.buffer.add_message(1, 100, "user", f"msg-{i}")
        msgs = self.buffer.get_messages(1, 100, limit=2)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["content"], "msg-3")
        self.assertEqual(msgs[1]["content"], "msg-4")

    def test_separate_sessions(self):
        """Different sessions should have independent buffers."""
        self.buffer.add_message(1, 100, "user", "Session A")
        self.buffer.add_message(1, 200, "user", "Session B")
        msgs_a = self.buffer.get_messages(1, 100)
        msgs_b = self.buffer.get_messages(1, 200)
        self.assertEqual(len(msgs_a), 1)
        self.assertEqual(len(msgs_b), 1)
        self.assertEqual(msgs_a[0]["content"], "Session A")
        self.assertEqual(msgs_b[0]["content"], "Session B")

    def test_separate_users(self):
        """Different users should have independent buffers."""
        self.buffer.add_message(1, 100, "user", "User 1")
        self.buffer.add_message(2, 100, "user", "User 2")
        msgs_1 = self.buffer.get_messages(1, 100)
        msgs_2 = self.buffer.get_messages(2, 100)
        self.assertEqual(msgs_1[0]["content"], "User 1")
        self.assertEqual(msgs_2[0]["content"], "User 2")

    def test_clear_session(self):
        """clear() should remove a specific session's buffer."""
        self.buffer.add_message(1, 100, "user", "Hello")
        self.buffer.add_message(1, 200, "user", "World")
        self.buffer.clear(1, 100)
        self.assertEqual(len(self.buffer.get_messages(1, 100)), 0)
        self.assertEqual(len(self.buffer.get_messages(1, 200)), 1)

    def test_clear_all(self):
        """clear_all() should remove everything."""
        self.buffer.add_message(1, 100, "user", "A")
        self.buffer.add_message(2, 200, "user", "B")
        self.buffer.clear_all()
        self.assertEqual(self.buffer.session_count(), 0)

    def test_empty_get(self):
        """Getting from nonexistent session should return empty list."""
        msgs = self.buffer.get_messages(999, 999)
        self.assertEqual(msgs, [])

    def test_session_count(self):
        """session_count should reflect active sessions."""
        self.buffer.add_message(1, 100, "user", "A")
        self.buffer.add_message(1, 200, "user", "B")
        self.buffer.add_message(2, 300, "user", "C")
        self.assertEqual(self.buffer.session_count(), 3)


if __name__ == "__main__":
    unittest.main()
