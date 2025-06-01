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
from unittest.mock import patch, MagicMock
from agents.loaders import agent_loader


class TestAgentLoader(unittest.TestCase):

    @patch("agents.loaders.agent_loader.get_db_connection")
    def test_load_user_agents_valid(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("FriendlyBot", "Helpful and kind", '{"tone": "friendly"}'),
            ("LogicBot", "Analytical personality", '{"tone": "analytical"}')
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        agents = agent_loader.load_user_agents(user_id=1)
        self.assertEqual(len(agents), 2)
        self.assertEqual(agents[0]["name"], "FriendlyBot")
        self.assertEqual(agents[0]["config"]["tone"], "friendly")
        self.assertEqual(agents[1]["config"]["tone"], "analytical")

    @patch("agents.loaders.agent_loader.get_db_connection")
    def test_load_user_agents_handles_dict(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("DirectBot", "JSON already parsed", {"tone": "direct"})
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        agents = agent_loader.load_user_agents(user_id=2)
        self.assertEqual(agents[0]["config"]["tone"], "direct")


if __name__ == "__main__":
    unittest.main()

