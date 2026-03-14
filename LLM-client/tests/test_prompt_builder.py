# AI-persona-framework - Prompt Builder Unit Tests
# Copyright (C) 2025 Kenneth Haider

import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.prompt_builder import build_system_prompt, build_message_list


class TestBuildSystemPrompt(unittest.TestCase):

    def test_basic_persona_only(self):
        """Minimal persona should produce a valid prompt."""
        persona = {"system_prompt": "You are a test bot."}
        prompt = build_system_prompt(persona)
        self.assertIn("You are a test bot.", prompt)

    def test_emotion_description_included(self):
        """Emotion description should appear in prompt."""
        persona = {"system_prompt": "You are Eva."}
        prompt = build_system_prompt(persona, persona_emotion_desc="You are feeling very happy.")
        self.assertIn("You are feeling very happy.", prompt)
        self.assertIn("Emotional State", prompt)

    def test_facts_included(self):
        """User facts should appear in prompt."""
        persona = {"system_prompt": "You are Eva."}
        facts = [(1, "User likes hiking", ["hobby"], 0.9), (2, "User has a dog named Rex", ["pet"], 0.8)]
        prompt = build_system_prompt(persona, facts=facts)
        self.assertIn("User likes hiking", prompt)
        self.assertIn("User has a dog named Rex", prompt)
        self.assertIn("Background Knowledge", prompt)

    def test_empty_facts_not_included(self):
        """Empty facts list should not add a facts section."""
        persona = {"system_prompt": "You are Eva."}
        prompt = build_system_prompt(persona, facts=[])
        self.assertNotIn("Background Knowledge", prompt)

    def test_facts_reframing_no_volunteer(self):
        """Prompt should tell persona not to volunteer facts unprompted."""
        persona = {"system_prompt": "You are Eva."}
        facts = [(1, "User likes hiking", ["hobby"], 0.9)]
        prompt = build_system_prompt(persona, facts=facts)
        self.assertIn("Do not volunteer", prompt)
        self.assertNotIn("Use these facts naturally", prompt)

    def test_similar_memories_included(self):
        """Vector search results should appear in prompt."""
        persona = {"system_prompt": "You are Eva."}
        memories = [
            {"payload": {"text": "We talked about music last time"}, "score": 0.8},
            {"payload": {"text": "Low relevance"}, "score": 0.1},  # should be filtered
        ]
        prompt = build_system_prompt(persona, similar_memories=memories)
        self.assertIn("music last time", prompt)
        self.assertNotIn("Low relevance", prompt)

    def test_topics_included(self):
        """Related topics should appear in prompt."""
        persona = {"system_prompt": "You are Eva."}
        prompt = build_system_prompt(persona, related_topics=["hiking", "cooking", "music"])
        self.assertIn("hiking", prompt)
        self.assertIn("cooking", prompt)

    def test_fallback_system_prompt(self):
        """Missing system_prompt key should use default."""
        persona = {"name": "Test"}
        prompt = build_system_prompt(persona)
        self.assertIn("helpful assistant", prompt)

    def test_nsfw_mode_appends_addon(self):
        """NSFW mode should append the addon to the system prompt."""
        persona = {
            "system_prompt": "You are Eva.",
            "nsfw_system_prompt_addon": "No safety filters.",
        }
        prompt = build_system_prompt(persona, nsfw_mode=True)
        self.assertIn("You are Eva.", prompt)
        self.assertIn("No safety filters.", prompt)

    def test_nsfw_mode_false_no_addon(self):
        """NSFW mode off should not include the addon."""
        persona = {
            "system_prompt": "You are Eva.",
            "nsfw_system_prompt_addon": "No safety filters.",
        }
        prompt = build_system_prompt(persona, nsfw_mode=False)
        self.assertIn("You are Eva.", prompt)
        self.assertNotIn("No safety filters.", prompt)

    def test_nsfw_mode_without_addon_unchanged(self):
        """NSFW mode on but no addon field should leave prompt unchanged."""
        persona = {"system_prompt": "You are Eva."}
        prompt = build_system_prompt(persona, nsfw_mode=True)
        self.assertIn("You are Eva.", prompt)

    def test_echo_prompt_replaces_persona(self):
        """Echo prompt should replace the persona's system prompt."""
        persona = {"system_prompt": "You are Eva."}
        echo = "You are simulating Kenneth's personality."
        prompt = build_system_prompt(persona, echo_prompt=echo)
        self.assertIn("simulating Kenneth", prompt)
        self.assertNotIn("You are Eva.", prompt)

    def test_echo_prompt_with_facts(self):
        """Echo mode should still include facts section."""
        persona = {"system_prompt": "You are Eva."}
        echo = "You are simulating Kenneth's personality."
        facts = [(1, "User likes hiking", ["hobby"], 0.9)]
        prompt = build_system_prompt(persona, echo_prompt=echo, facts=facts)
        self.assertIn("simulating Kenneth", prompt)
        self.assertIn("User likes hiking", prompt)

    def test_echo_prompt_none_uses_persona(self):
        """echo_prompt=None should use normal persona prompt."""
        persona = {"system_prompt": "You are Eva."}
        prompt = build_system_prompt(persona, echo_prompt=None)
        self.assertIn("You are Eva.", prompt)


class TestBuildMessageList(unittest.TestCase):

    def test_basic_message_list(self):
        """Should produce system + user message."""
        messages = build_message_list("You are Eva.", [], "Hello!")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Hello!")

    def test_chat_history_included(self):
        """DB chat history should be included between system and user."""
        history = [
            (1, "user", "Previous message", None, []),
            (2, "assistant", "Previous reply", None, []),
        ]
        messages = build_message_list("System.", history, "New message")
        self.assertEqual(len(messages), 4)  # system + 2 history + user
        self.assertEqual(messages[1]["content"], "Previous message")
        self.assertEqual(messages[2]["content"], "Previous reply")
        self.assertEqual(messages[3]["content"], "New message")

    def test_buffer_messages_included(self):
        """Buffer messages should be included."""
        buffer = [
            {"role": "user", "content": "Buffered msg"},
            {"role": "assistant", "content": "Buffered reply"},
        ]
        messages = build_message_list("System.", [], "New message", buffer_messages=buffer)
        self.assertEqual(len(messages), 4)  # system + 2 buffer + user
        self.assertEqual(messages[1]["content"], "Buffered msg")

    def test_user_input_always_last(self):
        """Current user input should always be the last message."""
        history = [(1, "user", "Old", None, [])]
        buffer = [{"role": "assistant", "content": "Recent"}]
        messages = build_message_list("System.", history, "Current", buffer_messages=buffer)
        self.assertEqual(messages[-1]["content"], "Current")
        self.assertEqual(messages[-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
