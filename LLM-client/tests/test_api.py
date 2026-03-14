# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Fake persona list matching persona_store.list_personas() return format
FAKE_PERSONAS = [
    {"id": 6, "user_id": 9999, "slug": "girlfriend", "name": "Maya",
     "description": "Affectionate companion", "system_prompt": "You are Maya.",
     "nsfw_capable": True, "nsfw_prompt_addon": "NSFW addon text",
     "nsfw_system_prompt_addon": "NSFW addon text",
     "memory_scope": {"tier1": True, "tier2": "all", "tier3": "private"}, "is_public": False},
    {"id": 7, "user_id": 9999, "slug": "trainer", "name": "Coach",
     "description": "Fitness trainer", "system_prompt": "You are Coach.",
     "nsfw_capable": False, "nsfw_prompt_addon": None,
     "nsfw_system_prompt_addon": None,
     "memory_scope": {"tier1": True, "tier2": "all", "tier3": "private"}, "is_public": False},
    {"id": 8, "user_id": 9999, "slug": "psychiatrist", "name": "Dr. Lena",
     "description": "Psychiatrist", "system_prompt": "You are Dr. Lena.",
     "nsfw_capable": False, "nsfw_prompt_addon": None,
     "nsfw_system_prompt_addon": None,
     "memory_scope": {"tier1": True, "tier2": "all", "tier3": "private"}, "is_public": False},
    {"id": 9, "user_id": 9999, "slug": "debug", "name": "DebugBot",
     "description": "Debug assistant", "system_prompt": "You are DebugBot.",
     "nsfw_capable": False, "nsfw_prompt_addon": None,
     "nsfw_system_prompt_addon": None,
     "memory_scope": {"tier1": True, "tier2": "all", "tier3": "private"}, "is_public": False},
]

FAKE_GIRLFRIEND = FAKE_PERSONAS[0]


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

    @patch("interface.api.app.list_personas", return_value=FAKE_PERSONAS)
    def test_personas_endpoint(self, mock_list):
        resp = self.client.get("/personas")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("personas", data)
        personas = data["personas"]
        self.assertIsInstance(personas, list)
        self.assertEqual(len(personas), 4)
        slugs = [p["slug"] for p in personas]
        self.assertIn("girlfriend", slugs)
        self.assertIn("debug", slugs)
        self.assertIn("trainer", slugs)
        self.assertIn("psychiatrist", slugs)
        for p in personas:
            self.assertIn("id", p)
            self.assertIn("name", p)
            self.assertIn("nsfw_capable", p)

    @patch("interface.api.app.list_personas", return_value=FAKE_PERSONAS)
    def test_personas_nsfw_capable_flag(self, mock_list):
        resp = self.client.get("/personas")
        personas = resp.json()["personas"]
        by_slug = {p["slug"]: p for p in personas}
        self.assertTrue(by_slug["girlfriend"]["nsfw_capable"])
        self.assertFalse(by_slug["debug"]["nsfw_capable"])
        self.assertFalse(by_slug["trainer"]["nsfw_capable"])

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

    @patch("interface.api.app.list_session_groups", return_value=[])
    @patch("interface.api.app.list_sessions")
    def test_sessions_endpoint(self, mock_list, mock_groups):
        mock_list.return_value = [
            {"id": 1, "persona_id": 6, "persona_slug": "girlfriend", "persona_name": "Maya",
             "start_time": "2026-01-01T00:00:00",
             "message_count": 5, "last_user_msg": "hi", "last_time": "2026-01-01T01:00:00",
             "incognito": False, "nsfw_mode": False},
            {"id": 2, "persona_id": 9, "persona_slug": "debug", "persona_name": "DebugBot",
             "start_time": "2026-01-02T00:00:00",
             "message_count": 2, "last_user_msg": "test", "last_time": "2026-01-02T01:00:00",
             "incognito": True, "nsfw_mode": False},
        ]
        resp = self.client.get("/sessions")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("sessions", data)
        self.assertIn("groups", data)
        self.assertIn("girlfriend", data["sessions"])
        self.assertIn("debug", data["sessions"])
        mock_list.assert_called_once_with(9999, limit=100)

    @patch("interface.api.app.get_persona", return_value=FAKE_GIRLFRIEND)
    @patch("interface.api.app.start_chat_session")
    def test_new_session_endpoint(self, mock_start, mock_persona):
        mock_start.return_value = 42
        resp = self.client.post("/sessions/new", json={
            "persona_id": 6,
            "nsfw_mode": True, "incognito": False
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], 42)
        self.assertEqual(data["persona_id"], 6)
        self.assertTrue(data["nsfw_mode"])
        self.assertFalse(data["incognito"])
        mock_start.assert_called_once_with(9999, 6, incognito=False, nsfw_mode=True, group_id=None)

    @patch("interface.api.app.get_persona")
    @patch("interface.api.app.start_chat_session")
    def test_new_session_incognito(self, mock_start, mock_persona):
        mock_persona.return_value = FAKE_PERSONAS[3]  # debug
        mock_start.return_value = 99
        resp = self.client.post("/sessions/new", json={
            "persona_id": 9, "incognito": True
        })
        data = resp.json()
        self.assertTrue(data["incognito"])
        mock_start.assert_called_once_with(9999, 9, incognito=True, nsfw_mode=False, group_id=None)

    @patch("interface.api.app.get_persona", return_value=FAKE_GIRLFRIEND)
    @patch("interface.api.app.start_chat_session", return_value=50)
    def test_new_session_rejects_other_users_persona(self, mock_start, mock_persona):
        other_persona = dict(FAKE_GIRLFRIEND)
        other_persona["user_id"] = 1234
        mock_persona.return_value = other_persona
        resp = self.client.post("/sessions/new", json={"persona_id": 6})
        self.assertEqual(resp.status_code, 403)

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

    @patch("interface.api.app.get_persona", return_value=FAKE_GIRLFRIEND)
    def test_chat_requires_persona_id(self, mock_persona):
        """Chat request must include persona_id field."""
        resp = self.client.post("/chat", json={"message": "hi"})
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

    @patch("interface.api.app.seed_default_personas")
    @patch("interface.api.app.create_user")
    @patch("interface.api.app.get_user_by_name")
    def test_register_success(self, mock_get, mock_create, mock_seed):
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
        # Default personas should be seeded
        mock_seed.assert_called_once_with(100)

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

    @patch("interface.api.app.get_persona", return_value=FAKE_GIRLFRIEND)
    @patch("interface.api.app.get_session_owner")
    def test_chat_rejects_other_users_session(self, mock_owner, mock_persona):
        mock_owner.return_value = 1234
        resp = self.client.post("/chat", json={
            "message": "hello", "persona_id": 6, "session_id": 5
        })
        self.assertEqual(resp.status_code, 403)

    @patch("interface.api.app.get_persona")
    def test_chat_rejects_other_users_persona(self, mock_persona):
        other_persona = dict(FAKE_GIRLFRIEND)
        other_persona["user_id"] = 1234
        mock_persona.return_value = other_persona
        resp = self.client.post("/chat", json={
            "message": "hello", "persona_id": 6
        })
        self.assertEqual(resp.status_code, 403)


class TestModelEndpoints(unittest.TestCase):
    """Tests for model listing and switching endpoints."""

    @classmethod
    def setUpClass(cls):
        from interface.api.app import app, get_current_user
        cls.app = app
        cls.app.dependency_overrides[get_current_user] = lambda: 9999
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.app.dependency_overrides.clear()

    @patch("interface.api.app.list_available_models")
    def test_models_list(self, mock_list):
        mock_list.return_value = [
            {"key": "qwen9b", "name": "Qwen 9B", "vram_gb": 9.5, "ctx_size": 8192},
            {"key": "tinyllama", "name": "TinyLlama", "vram_gb": 1, "ctx_size": 2048},
        ]
        resp = self.client.get("/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("models", data)
        self.assertEqual(len(data["models"]), 2)
        self.assertEqual(data["models"][0]["key"], "qwen9b")
        self.assertEqual(data["models"][0]["vram_gb"], 9.5)

    @patch("interface.api.app.switch_model")
    def test_model_switch_success(self, mock_switch):
        mock_switch.return_value = {
            "success": True, "model_key": "tinyllama",
            "pid": 1234, "killed_previous": True,
        }
        resp = self.client.post("/model/switch", json={"model_key": "tinyllama"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        mock_switch.assert_called_once_with("tinyllama")

    @patch("interface.api.app.switch_model")
    def test_model_switch_failure(self, mock_switch):
        mock_switch.return_value = {
            "success": False, "model_key": "bad",
            "error": "Unknown model: bad",
        }
        resp = self.client.post("/model/switch", json={"model_key": "bad"})
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Unknown model", resp.json()["detail"])


class TestPersonaCRUD(unittest.TestCase):
    """Test persona CRUD endpoints."""

    @classmethod
    def setUpClass(cls):
        from interface.api.app import app, get_current_user
        cls.app = app
        cls.app.dependency_overrides[get_current_user] = lambda: 9999
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.app.dependency_overrides.clear()

    @patch("interface.api.app.get_persona_by_slug", return_value=None)
    @patch("interface.api.app.db_create_persona", return_value=20)
    def test_create_persona(self, mock_create, mock_slug):
        resp = self.client.post("/personas", json={
            "slug": "custom", "name": "CustomBot",
            "description": "Test persona", "system_prompt": "You are CustomBot.",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["persona_id"], 20)
        self.assertEqual(data["slug"], "custom")
        mock_create.assert_called_once()

    @patch("interface.api.app.get_persona_by_slug")
    def test_create_persona_duplicate_slug(self, mock_slug):
        mock_slug.return_value = {"id": 1, "slug": "taken"}
        resp = self.client.post("/personas", json={
            "slug": "taken", "name": "Dup",
        })
        self.assertEqual(resp.status_code, 409)

    @patch("interface.api.app.get_persona", return_value=FAKE_GIRLFRIEND)
    @patch("interface.api.app.db_update_persona")
    def test_update_persona(self, mock_update, mock_get):
        resp = self.client.put("/personas/6", json={
            "slug": "girlfriend", "name": "Maya Updated",
        })
        self.assertEqual(resp.status_code, 200)
        mock_update.assert_called_once()

    @patch("interface.api.app.get_persona")
    def test_update_persona_forbidden(self, mock_get):
        other = dict(FAKE_GIRLFRIEND)
        other["user_id"] = 1234
        mock_get.return_value = other
        resp = self.client.put("/personas/6", json={
            "slug": "girlfriend", "name": "Stolen",
        })
        self.assertEqual(resp.status_code, 403)

    @patch("interface.api.app.get_persona", return_value=FAKE_GIRLFRIEND)
    @patch("interface.api.app.db_delete_persona", return_value=True)
    def test_delete_persona(self, mock_delete, mock_get):
        resp = self.client.delete("/personas/6")
        self.assertEqual(resp.status_code, 200)
        mock_delete.assert_called_once_with(6)


if __name__ == "__main__":
    unittest.main()
