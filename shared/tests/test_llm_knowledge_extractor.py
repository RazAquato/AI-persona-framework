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
import json
from unittest.mock import patch, MagicMock
from analysis.llm_knowledge_extractor import (
    LLMKnowledgeExtractor,
    _build_extraction_prompt,
    PRIVATE_KEYWORDS,
)


def _make_llm_response(facts=None, topics=None):
    """Build a mock LLM response dict."""
    payload = {
        "facts": facts or [],
        "topics": topics or [],
    }
    return {
        "content": json.dumps(payload),
        "tool_calls": [],
        "reasoning": "",
        "finish_reason": "stop",
        "raw": {},
    }


class TestLLMKnowledgeExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = LLMKnowledgeExtractor()

    # --- Return structure ---

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_return_structure_matches_regex(self, mock_llm):
        """LLM extractor should return same keys as regex extractor."""
        mock_llm.return_value = _make_llm_response(
            facts=[{"text": "User's name is Kenneth", "tier": "identity"}],
            topics=["family"],
        )
        result = self.extractor.extract_all("My name is Kenneth and I live in Norway")
        self.assertIn("facts", result)
        self.assertIn("entities", result)
        self.assertIn("topics", result)
        self.assertIn("classification", result)
        self.assertIn("raw_text", result)
        self.assertIsInstance(result["facts"], list)
        self.assertIsInstance(result["entities"], list)
        self.assertIsInstance(result["topics"], list)
        self.assertIsInstance(result["classification"], list)

    # --- Tier classification ---

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_identity_tier_classification(self, mock_llm):
        """Name, preferences, positive people mentions should be identity tier."""
        mock_llm.return_value = _make_llm_response(
            facts=[
                {"text": "User's name is Kenneth", "tier": "identity"},
                {"text": "User loves hiking", "tier": "identity"},
                {"text": "User's son Erik", "tier": "identity",
                 "entity_type": "person", "valence": "positive"},
            ],
        )
        result = self.extractor.extract_all("My name is Kenneth and I'm proud of my son Erik")
        all_items = result["facts"] + result["entities"]
        for item in all_items:
            self.assertEqual(item["tier"], "identity")

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_emotional_tier_classification(self, mock_llm):
        """Negative people mentions should be emotional tier."""
        mock_llm.return_value = _make_llm_response(
            facts=[
                {"text": "User is frustrated with coworker Tom", "tier": "emotional",
                 "entity_type": "person", "valence": "negative"},
            ],
        )
        result = self.extractor.extract_all("I'm really frustrated with my coworker Tom lately")
        entities = result["entities"]
        self.assertTrue(len(entities) > 0)
        self.assertEqual(entities[0]["tier"], "emotional")
        self.assertEqual(entities[0]["valence"], "negative")

    # --- Private content not extracted ---

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_private_content_not_extracted(self, mock_llm):
        """Facts with private keywords should be dropped by safety filter."""
        mock_llm.return_value = _make_llm_response(
            facts=[
                {"text": "User has sexual fantasies about X", "tier": "emotional"},
                {"text": "User's name is Kenneth", "tier": "identity"},
            ],
        )
        result = self.extractor.extract_all("Some message that the LLM misclassified")
        fact_texts = [f["text"] for f in result["facts"] + result["entities"]]
        self.assertFalse(any("sexual" in t for t in fact_texts))
        self.assertTrue(any("Kenneth" in t for t in fact_texts))

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_private_keyword_safety_filter(self, mock_llm):
        """All private keywords should be filtered."""
        for keyword in PRIVATE_KEYWORDS[:3]:
            mock_llm.return_value = _make_llm_response(
                facts=[{"text": f"Fact about {keyword}", "tier": "identity"}],
            )
            result = self.extractor.extract_all("Test message with enough words here")
            self.assertEqual(len(result["facts"]), 0,
                             f"Keyword '{keyword}' should have been filtered")

    # --- Regex fallback ---

    def test_short_messages_skip_llm(self):
        """Messages under 4 words should use regex fallback, not LLM."""
        with patch("analysis.llm_knowledge_extractor.call_llm") as mock_llm:
            result = self.extractor.extract_all("Hi there!")
            mock_llm.assert_not_called()
            self.assertIn("facts", result)

    def test_assistant_messages_use_regex(self):
        """Assistant messages should use regex fallback."""
        with patch("analysis.llm_knowledge_extractor.call_llm") as mock_llm:
            result = self.extractor.extract_all(
                "Let me help you with python coding", role="assistant")
            mock_llm.assert_not_called()
            self.assertEqual(result["facts"], [])
            self.assertEqual(result["entities"], [])

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_llm_failure_falls_back_to_regex(self, mock_llm):
        """LLM call failure should gracefully fall back to regex."""
        mock_llm.side_effect = Exception("Connection refused")
        result = self.extractor.extract_all("My name is Kenneth and I live in Norway")
        self.assertIn("facts", result)
        # Regex should still catch the name
        self.assertTrue(any("Kenneth" in f["text"] for f in result["facts"]))

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_invalid_json_falls_back_to_regex(self, mock_llm):
        """Invalid JSON response should fall back to regex."""
        mock_llm.return_value = {"content": "not valid json {{{", "tool_calls": [],
                                  "reasoning": "", "finish_reason": "stop", "raw": {}}
        result = self.extractor.extract_all("My name is Kenneth and I live in Norway")
        self.assertIn("facts", result)

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_empty_llm_response_falls_back(self, mock_llm):
        """Empty content from LLM should fall back to regex."""
        mock_llm.return_value = {"content": "", "tool_calls": [],
                                  "reasoning": "", "finish_reason": "stop", "raw": {}}
        result = self.extractor.extract_all("My name is Kenneth and I live in Norway")
        self.assertIn("facts", result)

    # --- Persona-specific suppression ---

    def test_psychiatrist_prompt_adds_suppression(self):
        """Psychiatrist persona should add conservative extraction rules."""
        prompt = _build_extraction_prompt(persona_slug="psychiatrist")
        self.assertIn("EXTRA conservative", prompt)

    def test_nsfw_mode_adds_suppression(self):
        """NSFW mode should add conservative extraction rules."""
        prompt = _build_extraction_prompt(nsfw_mode=True)
        self.assertIn("EXTRA conservative", prompt)

    def test_normal_prompt_no_extra_rules(self):
        """Normal mode should not have extra suppression."""
        prompt = _build_extraction_prompt(nsfw_mode=False, persona_slug="girlfriend")
        self.assertNotIn("EXTRA conservative", prompt)

    # --- Invalid tier handling ---

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_invalid_tier_defaults_to_identity(self, mock_llm):
        """An invalid tier from LLM should default to identity."""
        mock_llm.return_value = _make_llm_response(
            facts=[{"text": "Some fact", "tier": "private"}],
        )
        result = self.extractor.extract_all("Some message with enough words here please")
        if result["facts"]:
            self.assertEqual(result["facts"][0]["tier"], "identity")

    # --- Entity separation ---

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_entities_separated_from_facts(self, mock_llm):
        """Items with entity_type should go to entities, not facts."""
        mock_llm.return_value = _make_llm_response(
            facts=[
                {"text": "User likes hiking", "tier": "identity"},
                {"text": "User's dog Rex", "tier": "identity", "entity_type": "pet"},
            ],
        )
        result = self.extractor.extract_all("I like hiking and my dog Rex is great")
        self.assertEqual(len(result["facts"]), 1)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["entity_type"], "pet")

    # --- Topic merging ---

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_topics_merged_with_regex(self, mock_llm):
        """LLM topics and regex topics should be merged."""
        mock_llm.return_value = _make_llm_response(
            facts=[],
            topics=["custom_topic"],
        )
        result = self.extractor.extract_all(
            "I've been coding in python and learning machine learning")
        topic_names = [t["topic"] for t in result["topics"]]
        self.assertIn("custom_topic", topic_names)
        # Regex should also detect technology
        self.assertIn("technology", topic_names)

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_topics_capped_at_five(self, mock_llm):
        """Topics should be capped at 5."""
        mock_llm.return_value = _make_llm_response(
            facts=[],
            topics=["t1", "t2", "t3", "t4", "t5", "t6"],
        )
        result = self.extractor.extract_all("A message with enough words for LLM")
        self.assertLessEqual(len(result["topics"]), 5)

    # --- Valence passthrough ---

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_valence_included_in_output(self, mock_llm):
        """Valence from LLM should be passed through to the fact blob."""
        mock_llm.return_value = _make_llm_response(
            facts=[{"text": "User's friend Lars", "tier": "identity",
                     "entity_type": "person", "valence": "positive"}],
        )
        result = self.extractor.extract_all("My friend Lars is a great person and I like him")
        entity = result["entities"][0]
        self.assertEqual(entity["valence"], "positive")

    # --- Source metadata ---

    @patch("analysis.llm_knowledge_extractor.call_llm")
    def test_source_metadata_passed_through(self, mock_llm):
        """source_type and source_ref should be set on extracted items."""
        mock_llm.return_value = _make_llm_response(
            facts=[{"text": "User likes tea", "tier": "identity"}],
        )
        result = self.extractor.extract_all(
            "I really enjoy drinking tea every morning",
            source_type="document", source_ref="doc_42",
        )
        self.assertEqual(result["facts"][0]["source_type"], "document")
        self.assertEqual(result["facts"][0]["source_ref"], "doc_42")


if __name__ == "__main__":
    unittest.main()
