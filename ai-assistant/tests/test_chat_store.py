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
        self.session_id = session_id  # Save for reuse

    def test_log_and_get_chat_message(self):
        session_id = start_chat_session(self.user_id)
        message_id = log_chat_message(
            session_id,
            self.user_id,
            role="user",
            content="Hello assistant!",
            sentiment=0.7,
            topics=["greeting", "init"]
        )
        self.assertIsInstance(message_id, int)

        messages = get_chat_messages(session_id)
        self.assertGreaterEqual(len(messages), 1)
        self.assertEqual(messages[0][2], "Hello assistant!")

    def test_get_last_session(self):
        session_id = start_chat_session(self.user_id, personality_id="echo")
        last_session = get_last_session(self.user_id)
        self.assertEqual(last_session, session_id)

if __name__ == "__main__":
    unittest.main()

