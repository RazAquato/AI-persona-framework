import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from shared.scripts import agent_creator_with_emotions


class TestAgentCreatorWithEmotions(unittest.TestCase):

    @patch("psycopg2.connect")
    def test_insert_new_agent_with_emotions(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Simulate: no personality or emotional record exists
        mock_cursor.fetchone.side_effect = [None, None]

        user_id = 1
        name = "EmoBot"
        desc = "Emotionally aware bot"
        config = {
            "tone": "empathetic",
            "default_emotions": {"trust": 0.7, "joy": 0.6}
        }

        agent_creator_with_emotions.create_or_update_agent(user_id, name, desc, config)

        queries = [c[0][0] for c in mock_cursor.execute.call_args_list]

        # Confirm INSERT for personality and emotions
        assert any("INSERT INTO user_personalities" in q for q in queries), "Missing personality INSERT"
        assert any("INSERT INTO emotional_relationships" in q for q in queries), "Missing emotions INSERT"

    @patch("psycopg2.connect")
    def test_update_agent_and_emotions(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Simulate: personality exists, emotional relationship exists
        mock_cursor.fetchone.side_effect = [(101,), (202,)]

        user_id = 1
        name = "EmoBot"
        desc = "Updated emotional bot"
        config = {
            "tone": "gentle",
            "default_emotions": {"compassion": 0.9}
        }

        agent_creator_with_emotions.create_or_update_agent(user_id, name, desc, config)

        queries = [c[0][0] for c in mock_cursor.execute.call_args_list]

        assert any("UPDATE user_personalities" in q for q in queries), "Missing personality UPDATE"
        assert any("UPDATE emotional_relationships" in q for q in queries), "Missing emotions UPDATE"


if __name__ == "__main__":
    unittest.main()

