# AI-persona-framework - Engine Test With Emotion Hook
# Copyright (C) 2025 Kenneth Haider

import unittest
from unittest.mock import patch, MagicMock
from core.engine import run_conversation_turn


class TestEngine(unittest.TestCase):

    @patch("core.engine.store_emotion_vector")
    def test_run_conversation(self, mock_store_emotion):
        user_id = 9999
        input_text = "I'm so excited about the future!"
        result = run_conversation_turn(user_id=user_id, user_input=input_text)

        self.assertIn("assistant_reply", result)
        self.assertIn("session_id", result)
        self.assertGreater(len(result["assistant_reply"]), 0)

        # Emotion vector storage should be triggered
        mock_store_emotion.assert_called_once()

    @patch("core.engine.store_emotion_vector")
    def test_result_contains_emotions(self, mock_store_emotion):
        """Result should include persona and user emotion dicts."""
        result = run_conversation_turn(user_id=9999, user_input="Hello, how are you?")

        self.assertIn("persona_emotions", result)
        self.assertIn("user_emotions", result)
        self.assertIn("emotion_description", result)
        self.assertIsInstance(result["persona_emotions"], dict)
        self.assertIsInstance(result["user_emotions"], dict)
        self.assertEqual(len(result["persona_emotions"]), 18)
        self.assertEqual(len(result["user_emotions"]), 18)

    @patch("core.engine.store_emotion_vector")
    def test_personality_id_passed(self, mock_store_emotion):
        """Should accept and use personality_id parameter."""
        result = run_conversation_turn(
            user_id=9999, user_input="Hi!", personality_id="default"
        )
        self.assertIn("assistant_reply", result)

    @patch("core.engine.store_emotion_vector")
    def test_result_contains_extracted_knowledge(self, mock_store_emotion):
        """Result should include extracted_knowledge dict from M2."""
        result = run_conversation_turn(user_id=9999, user_input="My name is Kenneth")
        self.assertIn("extracted_knowledge", result)
        knowledge = result["extracted_knowledge"]
        self.assertIn("facts", knowledge)
        self.assertIn("entities", knowledge)
        self.assertIn("topics", knowledge)
        self.assertIn("classification", knowledge)

    @patch("core.engine.store_emotion_vector")
    def test_knowledge_extraction_finds_identity(self, mock_store_emotion):
        """Knowledge extractor should find identity facts in user input."""
        result = run_conversation_turn(user_id=9999, user_input="My name is Kenneth and I live in Norway")
        facts = result["extracted_knowledge"]["facts"]
        self.assertTrue(any("Kenneth" in f["text"] for f in facts))

    @patch("core.engine.store_emotion_vector")
    def test_knowledge_extraction_detects_topics(self, mock_store_emotion):
        """Topics should be detected and included in extracted_knowledge."""
        result = run_conversation_turn(
            user_id=9999,
            user_input="I've been coding in python and learning machine learning"
        )
        topics = result["extracted_knowledge"]["topics"]
        topic_names = [t["topic"] for t in topics]
        self.assertIn("technology", topic_names)

    @patch("core.engine.store_emotion_vector")
    def test_knowledge_extraction_classification(self, mock_store_emotion):
        """Message classification should be included."""
        result = run_conversation_turn(user_id=9999, user_input="What is your name?")
        self.assertIn("question", result["extracted_knowledge"]["classification"])

    @patch("core.engine.store_emotion_vector")
    def test_process_input_returns_string(self, mock_store_emotion):
        """process_input wrapper should return a non-empty string."""
        from core.engine import process_input
        reply = process_input("Hello!", user_id=9999)
        self.assertIsInstance(reply, str)
        self.assertGreater(len(reply), 0)

    @patch("core.engine.store_emotion_vector")
    def test_result_contains_llm_raw(self, mock_store_emotion):
        """Result dict should include llm_raw field."""
        result = run_conversation_turn(user_id=9999, user_input="Hi!")
        self.assertIn("llm_raw", result)


class TestStoreEntityInGraph(unittest.TestCase):
    """Test the _store_entity_in_graph helper in isolation."""

    @patch("core.engine.create_entity")
    def test_entity_with_named_pattern(self, mock_create):
        """Entities with 'named X' in text should extract the name."""
        from core.engine import _store_entity_in_graph
        entity = {"text": "User has a pet named Arix", "entity_type": "pet"}
        _store_entity_in_graph(9999, entity)
        mock_create.assert_called_once_with(9999, "Arix", "pet")

    @patch("core.engine.create_entity")
    def test_entity_fallback_to_capitalized_word(self, mock_create):
        """Entities without 'named X' should use last capitalized word."""
        from core.engine import _store_entity_in_graph
        entity = {"text": "User visited Paris", "entity_type": "place"}
        _store_entity_in_graph(9999, entity)
        mock_create.assert_called_once_with(9999, "Paris", "place")

    @patch("core.engine.create_entity")
    def test_entity_no_name_no_type_skipped(self, mock_create):
        """Entity with no name match and no entity_type should not create."""
        from core.engine import _store_entity_in_graph
        entity = {"text": "something generic", "entity_type": None}
        _store_entity_in_graph(9999, entity)
        mock_create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
