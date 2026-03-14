# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
from unittest.mock import patch
from echo.traits_extractor import extract_traits, _analyze_style, _analyze_vocabulary
from echo.echo_prompt_builder import build_echo_prompt
from echo.corpus_builder import build_corpus


class TestTraitsExtractor(unittest.TestCase):
    """Test the traits extractor with synthetic corpus data."""

    def _make_corpus(self, messages, facts=None, topics=None):
        return {
            "user_id": 9999,
            "messages": [{"role": "user", "content": m, "session_id": 1} for m in messages],
            "facts": facts or [],
            "topics": topics or [],
            "stats": {"total_messages": len(messages), "user_messages": len(messages)},
        }

    def test_extract_traits_basic(self):
        """Should return all expected keys."""
        corpus = self._make_corpus(["Hello there", "How are you?", "I like coding"])
        traits = extract_traits(corpus)
        self.assertIn("style", traits)
        self.assertIn("vocabulary", traits)
        self.assertIn("interests", traits)
        self.assertIn("values", traits)
        self.assertIn("message_count", traits)
        self.assertEqual(traits["message_count"], 3)

    def test_extract_traits_empty_corpus(self):
        """Empty corpus should return empty traits without error."""
        corpus = self._make_corpus([])
        traits = extract_traits(corpus)
        self.assertEqual(traits["message_count"], 0)
        self.assertEqual(traits["style"]["formality"], "unknown")

    def test_style_informal(self):
        """Informal markers should be detected."""
        messages = [
            "lol yeah that's cool",
            "gonna do it btw",
            "nah dude it's fine",
            "yo sup bruh",
        ]
        style = _analyze_style(messages)
        self.assertEqual(style["formality"], "informal")

    def test_style_formal(self):
        """Formal markers should be detected."""
        messages = [
            "Therefore, I believe this is the correct approach.",
            "Furthermore, the evidence suggests a different conclusion.",
            "Nevertheless, we must consider the consequences.",
            "Additionally, there are several important factors.",
            "Indeed, this is a fundamentally essential point.",
            "Moreover, the analysis certainly confirms this.",
        ]
        style = _analyze_style(messages)
        self.assertEqual(style["formality"], "formal")

    def test_style_short_messages(self):
        """Short messages should be detected."""
        messages = ["yes", "ok", "no", "sure", "fine", "yep"]
        style = _analyze_style(messages)
        self.assertLess(style["avg_words_per_message"], 5)

    def test_vocabulary_extraction(self):
        """Should extract characteristic words."""
        messages = [
            "python is amazing for data science",
            "python and javascript are my main languages",
            "data pipelines in python are great",
        ]
        vocab = _analyze_vocabulary(messages)
        self.assertIn("python", vocab["characteristic_words"])
        self.assertGreater(vocab["vocabulary_size"], 0)

    def test_interests_from_topics(self):
        """Should rank interests by weight."""
        corpus = self._make_corpus(
            ["test"],
            topics=[
                {"topic": "technology", "weight": 10},
                {"topic": "gaming", "weight": 5},
                {"topic": "cooking", "weight": 1},
            ],
        )
        traits = extract_traits(corpus)
        self.assertEqual(traits["interests"][0]["topic"], "technology")

    def test_values_from_facts(self):
        """Should extract identity facts as values."""
        corpus = self._make_corpus(
            ["test"],
            facts=[
                {"text": "User's name is Kenneth", "tier": "identity", "entity_type": "person"},
                {"text": "User loves hiking", "tier": "identity", "entity_type": None},
            ],
        )
        traits = extract_traits(corpus)
        self.assertGreater(len(traits["values"]), 0)


class TestEchoPromptBuilder(unittest.TestCase):
    """Test the echo prompt builder."""

    def test_build_echo_prompt_basic(self):
        """Should produce a non-empty prompt."""
        traits = {
            "style": {
                "avg_message_length": 50,
                "avg_words_per_message": 10,
                "avg_sentence_length": 8,
                "formality": "informal",
                "uses_questions": 0.3,
                "uses_exclamations": 0.4,
                "uses_emoji": 0.0,
                "uses_ellipsis": 0.0,
                "starts_lowercase": 0.6,
            },
            "vocabulary": {
                "characteristic_words": ["python", "coding", "server"],
                "common_phrases": ["i think", "let me"],
                "vocabulary_size": 200,
            },
            "interests": [
                {"topic": "technology", "weight": 10},
                {"topic": "gaming", "weight": 5},
            ],
            "values": ["User's name is Kenneth", "User loves hiking"],
            "message_count": 50,
        }
        prompt = build_echo_prompt(traits, user_name="Kenneth")
        self.assertIn("Kenneth", prompt)
        self.assertIn("python", prompt)
        self.assertIn("technology", prompt)
        self.assertIn("hiking", prompt)

    def test_build_echo_prompt_empty_traits(self):
        """Should handle empty traits gracefully."""
        traits = {"message_count": 0}
        prompt = build_echo_prompt(traits, user_name="Test")
        self.assertIn("not enough", prompt)

    def test_build_echo_prompt_informal_style(self):
        """Informal style should produce casual instructions."""
        traits = {
            "style": {"formality": "informal", "avg_words_per_message": 8,
                       "avg_message_length": 30, "avg_sentence_length": 6,
                       "uses_questions": 0, "uses_exclamations": 0,
                       "uses_emoji": 0, "uses_ellipsis": 0, "starts_lowercase": 0},
            "vocabulary": {"characteristic_words": [], "common_phrases": [], "vocabulary_size": 0},
            "interests": [],
            "values": [],
            "message_count": 10,
        }
        prompt = build_echo_prompt(traits)
        self.assertIn("casual", prompt.lower())

    def test_build_echo_prompt_no_break_character(self):
        """Prompt should instruct not to break character."""
        traits = {
            "style": {"formality": "neutral", "avg_words_per_message": 15,
                       "avg_message_length": 60, "avg_sentence_length": 10,
                       "uses_questions": 0, "uses_exclamations": 0,
                       "uses_emoji": 0, "uses_ellipsis": 0, "starts_lowercase": 0},
            "vocabulary": {"characteristic_words": [], "common_phrases": [], "vocabulary_size": 0},
            "interests": [],
            "values": [],
            "message_count": 5,
        }
        prompt = build_echo_prompt(traits)
        self.assertIn("Do NOT break character", prompt)


class TestCorpusBuilderTierFilter(unittest.TestCase):
    """Verify that corpus builder only includes identity-tier facts."""

    @patch("echo.corpus_builder._get_user_messages", return_value=[])
    @patch("echo.corpus_builder.get_user_topics", return_value=[])
    @patch("echo.corpus_builder.get_facts_by_tier")
    def test_corpus_builder_filters_to_identity_tier(self, mock_facts, mock_topics, mock_msgs):
        """build_corpus should call get_facts_by_tier with ['identity'] only."""
        mock_facts.return_value = [
            (1, "User's name is Kenneth", None, 0.8, "identity", "person"),
        ]
        corpus = build_corpus(user_id=9999)
        mock_facts.assert_called_once_with(9999, ["identity"])
        # Emotional facts should not be requested
        call_args = mock_facts.call_args
        self.assertEqual(call_args[0][1], ["identity"])

    @patch("echo.corpus_builder._get_user_messages", return_value=[])
    @patch("echo.corpus_builder.get_user_topics", return_value=[])
    @patch("echo.corpus_builder.get_facts_by_tier")
    def test_corpus_excludes_emotional_facts(self, mock_facts, mock_topics, mock_msgs):
        """Emotional-tier facts should never appear in Echo corpus."""
        # Simulate only identity facts being returned (as expected)
        mock_facts.return_value = [
            (1, "User likes hiking", None, 0.7, "identity", None),
        ]
        corpus = build_corpus(user_id=9999)
        for fact in corpus["facts"]:
            self.assertEqual(fact["tier"], "identity")


if __name__ == "__main__":
    unittest.main()
