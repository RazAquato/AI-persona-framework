# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
import os
import psycopg2
from dotenv import load_dotenv
from memory.user_store import (
    create_user, get_user_by_name, get_user_by_id,
    set_password, get_session_owner,
)


class TestUserStore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import random
        cls._suffix = str(random.randint(100000, 999999))
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
        load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

        cls.conn = psycopg2.connect(
            dbname=os.getenv("PG_DATABASE"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
        )
        cls.created_user_ids = []

    @classmethod
    def tearDownClass(cls):
        cur = cls.conn.cursor()
        for uid in cls.created_user_ids:
            cur.execute("DELETE FROM chat_sessions WHERE user_id = %s;", (uid,))
            cur.execute("DELETE FROM users WHERE id = %s;", (uid,))
        cls.conn.commit()
        cur.close()
        cls.conn.close()

    def _name(self, base):
        return f"{base}_{self._suffix}"

    def test_create_user(self):
        uid = create_user(self._name("create_1"), "fakehash123")
        self.created_user_ids.append(uid)
        self.assertIsInstance(uid, int)

    def test_get_user_by_name(self):
        name = self._name("byname_2")
        uid = create_user(name, "fakehash456")
        self.created_user_ids.append(uid)
        row = get_user_by_name(name)
        self.assertIsNotNone(row)
        self.assertEqual(row[0], uid)
        self.assertEqual(row[1], name)
        self.assertEqual(row[2], "fakehash456")

    def test_get_user_by_name_case_insensitive(self):
        name = self._name("CaseTest")
        uid = create_user(name, "hash")
        self.created_user_ids.append(uid)
        row = get_user_by_name(name.lower())
        self.assertIsNotNone(row)
        self.assertEqual(row[0], uid)

    def test_get_user_by_name_missing(self):
        row = get_user_by_name("nonexistent_user_xyz_999")
        self.assertIsNone(row)

    def test_get_user_by_id(self):
        name = self._name("byid_3")
        uid = create_user(name, "hash")
        self.created_user_ids.append(uid)
        row = get_user_by_id(uid)
        self.assertIsNotNone(row)
        self.assertEqual(row[0], uid)
        self.assertEqual(row[1], name)

    def test_get_user_by_id_missing(self):
        row = get_user_by_id(999888777)
        self.assertIsNone(row)

    def test_set_password(self):
        name = self._name("setpw_4")
        uid = create_user(name, "oldhash")
        self.created_user_ids.append(uid)
        set_password(uid, "newhash")
        row = get_user_by_name(name)
        self.assertEqual(row[2], "newhash")

    def test_get_session_owner(self):
        uid = create_user(self._name("owner_5"), "hash")
        self.created_user_ids.append(uid)
        # Create a session (persona_id=NULL is OK)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO chat_sessions (user_id) VALUES (%s) RETURNING id;",
            (uid,),
        )
        sid = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        owner = get_session_owner(sid)
        self.assertEqual(owner, uid)

    def test_get_session_owner_missing(self):
        owner = get_session_owner(999888777)
        self.assertIsNone(owner)


if __name__ == "__main__":
    unittest.main()
