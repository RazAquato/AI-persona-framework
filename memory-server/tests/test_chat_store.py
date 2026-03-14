# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
import os
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from memory.chat_store import (
    start_chat_session,
    log_chat_message,
    get_chat_messages,
    get_last_session,
    get_last_session_for_persona,
    list_sessions
)


class TestChatStore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
        load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

        cls.conn = psycopg2.connect(
            dbname=os.getenv("PG_DATABASE"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )
        cls.cur = cls.conn.cursor()
        cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("TestChatUser",))
        cls.user_id = cls.cur.fetchone()[0]
        # Create a test persona for FK references
        cls.cur.execute("""
            INSERT INTO user_personalities (user_id, slug, name, system_prompt, memory_scope)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """, (cls.user_id, "test_chat", "TestChatPersona", "You are a test bot.", Json({"tier1": True, "tier2": "all", "tier3": "private"})))
        cls.persona_id = cls.cur.fetchone()[0]
        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        cls.cur.execute("DELETE FROM chat_messages WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM chat_sessions WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM user_personalities WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM users WHERE id = %s;", (cls.user_id,))
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    def test_start_chat_session(self):
        session_id = start_chat_session(self.user_id, persona_id=self.persona_id, context_summary="Test run.")
        self.assertIsInstance(session_id, int)

    def test_start_chat_session_no_persona(self):
        session_id = start_chat_session(self.user_id)
        self.assertIsInstance(session_id, int)

    def test_get_chat_messages_empty(self):
        session_id = start_chat_session(self.user_id)
        messages = get_chat_messages(session_id)
        self.assertEqual(messages, [])

    def test_log_and_get_chat_messages_ordered(self):
        session_id = start_chat_session(self.user_id)
        log_chat_message(session_id, self.user_id, "user", "Message one")
        log_chat_message(session_id, self.user_id, "assistant", "Message two")
        messages = get_chat_messages(session_id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0][2], "Message one")
        self.assertEqual(messages[1][2], "Message two")

    def test_store_chat_with_embedding(self):
        session_id = start_chat_session(self.user_id)
        embedding = [0.01] * 384
        message_id = log_chat_message(session_id, self.user_id, "user", "Embedded msg", embedding=embedding)
        self.assertIsInstance(message_id, int)

    def test_invalid_role_raises_error(self):
        session_id = start_chat_session(self.user_id)
        with self.assertRaises(Exception):
            log_chat_message(session_id, self.user_id, "alien", "Invalid role")

    def test_get_last_session(self):
        session_id = start_chat_session(self.user_id)
        last_id = get_last_session(self.user_id)
        self.assertEqual(session_id, last_id)

    def test_list_sessions_returns_list(self):
        sessions = list_sessions(self.user_id)
        self.assertIsInstance(sessions, list)

    def test_list_sessions_contains_expected_keys(self):
        start_chat_session(self.user_id, persona_id=self.persona_id)
        sessions = list_sessions(self.user_id)
        self.assertGreater(len(sessions), 0)
        s = sessions[0]
        for key in ("id", "persona_id", "start_time", "message_count", "last_user_msg", "last_time"):
            self.assertIn(key, s)

    def test_list_sessions_includes_persona_info(self):
        """Sessions linked to a persona should include slug and name."""
        sid = start_chat_session(self.user_id, persona_id=self.persona_id)
        sessions = list_sessions(self.user_id)
        found = [s for s in sessions if s["id"] == sid]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["persona_slug"], "test_chat")
        self.assertEqual(found[0]["persona_name"], "TestChatPersona")

    def test_list_sessions_with_messages(self):
        sid = start_chat_session(self.user_id, persona_id=self.persona_id)
        log_chat_message(sid, self.user_id, "user", "Hello from list_sessions test")
        log_chat_message(sid, self.user_id, "assistant", "Hi there")
        sessions = list_sessions(self.user_id)
        found = [s for s in sessions if s["id"] == sid]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["message_count"], 2)
        self.assertIn("Hello from list_sessions", found[0]["last_user_msg"])

    def test_list_sessions_respects_limit(self):
        for _ in range(3):
            start_chat_session(self.user_id, persona_id=self.persona_id)
        sessions = list_sessions(self.user_id, limit=2)
        self.assertLessEqual(len(sessions), 2)

    def test_list_sessions_empty_user(self):
        sessions = list_sessions(999999)
        self.assertEqual(sessions, [])

    def test_start_session_with_incognito(self):
        sid = start_chat_session(self.user_id, persona_id=self.persona_id, incognito=True)
        self.assertIsInstance(sid, int)
        sessions = list_sessions(self.user_id)
        found = [s for s in sessions if s["id"] == sid]
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]["incognito"])

    def test_start_session_with_nsfw_mode(self):
        sid = start_chat_session(self.user_id, persona_id=self.persona_id, nsfw_mode=True)
        self.assertIsInstance(sid, int)
        sessions = list_sessions(self.user_id)
        found = [s for s in sessions if s["id"] == sid]
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]["nsfw_mode"])

    def test_list_sessions_includes_new_flags(self):
        sid = start_chat_session(self.user_id, persona_id=self.persona_id,
                                 incognito=True, nsfw_mode=True)
        sessions = list_sessions(self.user_id)
        found = [s for s in sessions if s["id"] == sid]
        self.assertEqual(len(found), 1)
        self.assertIn("incognito", found[0])
        self.assertIn("nsfw_mode", found[0])
        self.assertTrue(found[0]["incognito"])
        self.assertTrue(found[0]["nsfw_mode"])

    def test_get_last_session_for_persona(self):
        """Should return the most recent session for a specific persona."""
        sid = start_chat_session(self.user_id, persona_id=self.persona_id)
        last = get_last_session_for_persona(self.user_id, self.persona_id)
        self.assertEqual(last, sid)

    def test_get_last_session_for_persona_isolation(self):
        """Sessions for persona A must not be returned when querying persona B."""
        # Create a second persona
        self.cur.execute("""
            INSERT INTO user_personalities (user_id, slug, name, system_prompt, memory_scope)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """, (self.user_id, "test_chat_other", "OtherPersona", "Other.", '{"tier1": true, "tier2": "all", "tier3": "private"}'))
        other_persona_id = self.cur.fetchone()[0]
        self.conn.commit()

        start_chat_session(self.user_id, persona_id=self.persona_id)
        last = get_last_session_for_persona(self.user_id, other_persona_id)
        self.assertIsNone(last)

        # Create a session for the other persona and confirm it's returned
        sid_other = start_chat_session(self.user_id, persona_id=other_persona_id)
        last = get_last_session_for_persona(self.user_id, other_persona_id)
        self.assertEqual(last, sid_other)

        # Cleanup
        self.cur.execute("DELETE FROM chat_sessions WHERE persona_id = %s;", (other_persona_id,))
        self.cur.execute("DELETE FROM user_personalities WHERE id = %s;", (other_persona_id,))
        self.conn.commit()

    def test_get_last_session_for_persona_skips_incognito(self):
        """Incognito sessions should not be returned by get_last_session_for_persona."""
        start_chat_session(self.user_id, persona_id=self.persona_id, incognito=True)
        last = get_last_session_for_persona(self.user_id, self.persona_id)
        # Should return an earlier non-incognito session or None
        if last is not None:
            # Verify it's not the incognito one
            sessions = list_sessions(self.user_id)
            found = [s for s in sessions if s["id"] == last]
            self.assertFalse(found[0]["incognito"])

    def test_get_last_session_for_persona_missing(self):
        """Should return None for a persona with no sessions."""
        last = get_last_session_for_persona(self.user_id, 999888)
        self.assertIsNone(last)


if __name__ == "__main__":
    unittest.main()
