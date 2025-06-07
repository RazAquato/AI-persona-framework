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
from fastapi.testclient import TestClient
from interface.api.app import app

class TestChatAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_chat_endpoint_returns_response(self):
        payload = {
            "persona": "maya",
            "session_id": "unit-test-session",
            "message": "Hello Maya, how are you?"
        }
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())

    def test_chat_missing_fields_returns_422(self):
        response = self.client.post("/chat", json={"message": "Hello?"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

