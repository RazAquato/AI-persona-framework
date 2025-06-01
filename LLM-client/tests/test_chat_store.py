# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
import os
import psycopg2
from dotenv import load_dotenv
from memory.chat_store import (
    start_chat_session,
    log_chat_message,
    get_chat_messages,
    get_last_session
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
        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        cls.cur.execute("DELETE FROM chat_messages WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM chat_sessions WHERE user_id = %s;", (cls.user_id,))
        cls.cur.execute("DELETE FROM users WHERE id = %s;", (cls.user_id,))
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    def test_start_chat_session(self):
        session_id = start_chat_session(self.user_id, personality_id="TestPersona", context_summary="Test run.")
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


if __name__ == "__main__":
    unittest.main()

