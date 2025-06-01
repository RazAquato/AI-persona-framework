import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call, ANY
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from shared.scripts import agent_creator


class TestAgentCreator(unittest.TestCase):

    @patch("psycopg2.connect")
    def test_insert_new_agent(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None  # Simulate: no existing agent

        user_id = 1
        name = "TestBot"
        desc = "Test insert path"
        config = {"tone": "curious"}

        agent_creator.create_or_update_agent(user_id, name, desc, config)

        # Collect all calls
        calls = [c for c in mock_cursor.execute.call_args_list]
        matched = any("INSERT INTO user_personalities" in c[0][0] for c in calls)

        self.assertTrue(matched, "INSERT INTO user_personalities not called")

    @patch("psycopg2.connect")
    def test_update_existing_agent(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (42,)  # Simulate: existing agent

        user_id = 1
        name = "TestBot"
        desc = "Updated description"
        config = {"tone": "analytical"}

        agent_creator.create_or_update_agent(user_id, name, desc, config)

        calls = [c for c in mock_cursor.execute.call_args_list]
        matched = any("UPDATE user_personalities" in c[0][0] for c in calls)

        self.assertTrue(matched, "UPDATE user_personalities not called")


if __name__ == "__main__":
    unittest.main()

