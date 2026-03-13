# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from interface.api.app import app
        cls.client = TestClient(app)

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_index_returns_html(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("AI Persona Chat", resp.text)

    def test_personas_endpoint(self):
        resp = self.client.get("/personas")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("personas", data)
        self.assertIn("girlfriend", data["personas"])
        self.assertIn("debug", data["personas"])
        for pid, info in data["personas"].items():
            self.assertIn("name", info)
            self.assertIn("description", info)

    def test_tools_endpoint(self):
        resp = self.client.get("/tools")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("tools", resp.json())

    @patch("interface.api.app.requests.get")
    def test_model_endpoint_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{
                "id": "TestModel-7B-Q4.gguf",
                "meta": {"n_params": 7000000000}
            }]
        }
        mock_get.return_value = mock_resp

        resp = self.client.get("/model")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model"], "TestModel-7B-Q4")
        self.assertEqual(data["params"], 7000000000)

    @patch("interface.api.app.requests.get")
    def test_model_endpoint_offline(self, mock_get):
        mock_get.side_effect = Exception("connection refused")
        resp = self.client.get("/model")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["model"], "offline")

    @patch("interface.api.app.list_sessions")
    def test_sessions_endpoint(self, mock_list):
        mock_list.return_value = [
            {"id": 1, "personality_id": "girlfriend", "start_time": "2026-01-01T00:00:00",
             "message_count": 5, "last_user_msg": "hi", "last_time": "2026-01-01T01:00:00"},
            {"id": 2, "personality_id": "debug", "start_time": "2026-01-02T00:00:00",
             "message_count": 2, "last_user_msg": "test", "last_time": "2026-01-02T01:00:00"},
        ]
        resp = self.client.get("/sessions?user_id=9999")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("sessions", data)
        self.assertIn("girlfriend", data["sessions"])
        self.assertIn("debug", data["sessions"])
        self.assertEqual(len(data["sessions"]["girlfriend"]), 1)

    @patch("interface.api.app.start_chat_session")
    def test_new_session_endpoint(self, mock_start):
        mock_start.return_value = 42
        resp = self.client.post("/sessions/new", json={"user_id": 9999, "persona": "girlfriend"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], 42)
        self.assertEqual(data["persona"], "girlfriend")
        mock_start.assert_called_once_with(9999, "girlfriend")

    @patch("interface.api.app.get_chat_messages")
    def test_session_messages_endpoint(self, mock_msgs):
        mock_msgs.return_value = [
            (1, "user", "Hello", None, None),
            (2, "assistant", "Hi there", None, None),
        ]
        resp = self.client.get("/sessions/100/messages")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(data["messages"][0]["role"], "user")
        self.assertEqual(data["messages"][1]["content"], "Hi there")

    def test_chat_missing_message_returns_422(self):
        resp = self.client.post("/chat", json={"user_id": 9999})
        self.assertEqual(resp.status_code, 422)

    def test_html_contains_sidebar(self):
        resp = self.client.get("/")
        self.assertIn("sidebar", resp.text)
        self.assertIn("persona-select", resp.text)
        self.assertIn("new-chat-btn", resp.text)
        self.assertIn("session-list", resp.text)

    def test_html_contains_model_badge(self):
        resp = self.client.get("/")
        self.assertIn("header-model", resp.text)
        self.assertIn("loadModelInfo", resp.text)


if __name__ == "__main__":
    unittest.main()
