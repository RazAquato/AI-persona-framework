# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestAPIEndpoints(unittest.TestCase):
    """Tests for API endpoints — auth dependency overridden with test user."""

    @classmethod
    def setUpClass(cls):
        from interface.api.app import app, get_current_user
        cls.app = app
        # Override auth to always return test user_id
        cls.app.dependency_overrides[get_current_user] = lambda: 9999
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.app.dependency_overrides.clear()

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_personas_endpoint(self):
        resp = self.client.get("/personas")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("personas", data)
        self.assertIn("girlfriend", data["personas"])
        self.assertIn("debug", data["personas"])
        self.assertIn("trainer", data["personas"])
        self.assertIn("psychiatrist", data["personas"])
        self.assertEqual(len(data["personas"]), 4)
        for pid, info in data["personas"].items():
            self.assertIn("name", info)
            self.assertIn("description", info)
            self.assertIn("nsfw_capable", info)

    def test_personas_nsfw_capable_flag(self):
        resp = self.client.get("/personas")
        data = resp.json()["personas"]
        self.assertTrue(data["girlfriend"]["nsfw_capable"])
        self.assertFalse(data["debug"]["nsfw_capable"])
        self.assertFalse(data["trainer"]["nsfw_capable"])

    def test_tools_endpoint(self):
        resp = self.client.get("/tools")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("tools", resp.json())

    @patch("interface.api.app.http_requests.get")
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

    @patch("interface.api.app.http_requests.get")
    def test_model_endpoint_offline(self, mock_get):
        mock_get.side_effect = Exception("connection refused")
        resp = self.client.get("/model")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["model"], "offline")

    @patch("interface.api.app.list_sessions")
    def test_sessions_endpoint(self, mock_list):
        mock_list.return_value = [
            {"id": 1, "personality_id": "girlfriend", "start_time": "2026-01-01T00:00:00",
             "message_count": 5, "last_user_msg": "hi", "last_time": "2026-01-01T01:00:00",
             "incognito": False, "nsfw_mode": False},
            {"id": 2, "personality_id": "debug", "start_time": "2026-01-02T00:00:00",
             "message_count": 2, "last_user_msg": "test", "last_time": "2026-01-02T01:00:00",
             "incognito": True, "nsfw_mode": False},
        ]
        resp = self.client.get("/sessions")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("sessions", data)
        self.assertIn("girlfriend", data["sessions"])
        self.assertIn("debug", data["sessions"])
        # Verify list_sessions was called with the overridden user_id
        mock_list.assert_called_once_with(9999, limit=100)

    @patch("interface.api.app.start_chat_session")
    def test_new_session_endpoint(self, mock_start):
        mock_start.return_value = 42
        resp = self.client.post("/sessions/new", json={
            "persona": "girlfriend",
            "nsfw_mode": True, "incognito": False
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], 42)
        self.assertEqual(data["persona"], "girlfriend")
        self.assertTrue(data["nsfw_mode"])
        self.assertFalse(data["incognito"])
        mock_start.assert_called_once_with(9999, "girlfriend", incognito=False, nsfw_mode=True)

    @patch("interface.api.app.start_chat_session")
    def test_new_session_incognito(self, mock_start):
        mock_start.return_value = 99
        resp = self.client.post("/sessions/new", json={
            "persona": "debug", "incognito": True
        })
        data = resp.json()
        self.assertTrue(data["incognito"])
        mock_start.assert_called_once_with(9999, "debug", incognito=True, nsfw_mode=False)

    @patch("interface.api.app.get_session_owner")
    @patch("interface.api.app.get_chat_messages")
    def test_session_messages_endpoint(self, mock_msgs, mock_owner):
        mock_owner.return_value = 9999  # same as auth user
        mock_msgs.return_value = [
            (1, "user", "Hello", None, None),
            (2, "assistant", "Hi there", None, None),
        ]
        resp = self.client.get("/sessions/100/messages")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(data["messages"][0]["role"], "user")

    @patch("interface.api.app.get_session_owner")
    def test_session_messages_forbidden_for_other_user(self, mock_owner):
        mock_owner.return_value = 1234  # different user
        resp = self.client.get("/sessions/100/messages")
        self.assertEqual(resp.status_code, 403)

    def test_chat_missing_message_returns_422(self):
        resp = self.client.post("/chat", json={})
        self.assertEqual(resp.status_code, 422)


class TestAuthPages(unittest.TestCase):
    """Tests for login/register pages and auth redirects."""

    def setUp(self):
        """Fresh client with no auth overrides per test."""
        from interface.api.app import app
        app.dependency_overrides.clear()
        self.client = TestClient(app, follow_redirects=False)

    def test_login_page_renders(self):
        resp = self.client.get("/login")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("Login", resp.text)
        self.assertIn("Register", resp.text)

    def test_index_redirects_to_login_without_cookie(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["location"])

    def test_unauthenticated_api_returns_401(self):
        resp = self.client.get("/personas")
        self.assertEqual(resp.status_code, 401)

    def test_unauthenticated_sessions_returns_401(self):
        resp = self.client.get("/sessions")
        self.assertEqual(resp.status_code, 401)

    @patch("interface.api.app.get_user_by_name")
    def test_login_invalid_credentials(self, mock_get):
        mock_get.return_value = None
        resp = self.client.post("/api/login", json={
            "username": "nobody", "password": "wrong"
        })
        self.assertEqual(resp.status_code, 401)

    @patch("interface.api.app.create_user")
    @patch("interface.api.app.get_user_by_name")
    def test_register_success(self, mock_get, mock_create):
        mock_get.return_value = None  # username not taken
        mock_create.return_value = 100
        resp = self.client.post("/api/register", json={
            "username": "newuser", "password": "pass1234"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["user_id"], 100)
        self.assertEqual(data["username"], "newuser")
        # Cookie should be set
        self.assertIn("session", resp.cookies)

    @patch("interface.api.app.get_user_by_name")
    def test_register_duplicate_username(self, mock_get):
        mock_get.return_value = (1, "taken", "hash")
        resp = self.client.post("/api/register", json={
            "username": "taken", "password": "pass1234"
        })
        self.assertEqual(resp.status_code, 409)

    def test_register_short_password(self):
        resp = self.client.post("/api/register", json={
            "username": "user", "password": "ab"
        })
        self.assertEqual(resp.status_code, 400)

    def test_logout(self):
        resp = self.client.post("/api/logout")
        self.assertEqual(resp.status_code, 200)


class TestAuthChatEndpoint(unittest.TestCase):
    """Test /chat session ownership check."""

    @classmethod
    def setUpClass(cls):
        from interface.api.app import app, get_current_user
        cls.app = app
        cls.app.dependency_overrides[get_current_user] = lambda: 9999
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.app.dependency_overrides.clear()

    @patch("interface.api.app.get_session_owner")
    def test_chat_rejects_other_users_session(self, mock_owner):
        mock_owner.return_value = 1234
        resp = self.client.post("/chat", json={
            "message": "hello", "session_id": 5
        })
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
