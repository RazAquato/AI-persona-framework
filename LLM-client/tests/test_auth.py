# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
import time
from unittest.mock import patch
from core.auth import hash_password, verify_password, create_auth_cookie, verify_auth_cookie


class TestPasswordHashing(unittest.TestCase):

    def test_hash_and_verify(self):
        pw = "test_password_123"
        hashed = hash_password(pw)
        self.assertTrue(verify_password(pw, hashed))

    def test_wrong_password_fails(self):
        hashed = hash_password("correct")
        self.assertFalse(verify_password("wrong", hashed))

    def test_hash_format(self):
        hashed = hash_password("test")
        self.assertIn("$", hashed)
        salt, dk = hashed.split("$", 1)
        self.assertEqual(len(salt), 32)  # 16 bytes hex

    def test_different_hashes_for_same_password(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        self.assertNotEqual(h1, h2)  # different salts

    def test_verify_empty_hash(self):
        self.assertFalse(verify_password("pw", ""))
        self.assertFalse(verify_password("pw", None))

    def test_verify_malformed_hash(self):
        self.assertFalse(verify_password("pw", "no_dollar_sign"))


class TestAuthCookie(unittest.TestCase):

    def test_create_and_verify(self):
        cookie = create_auth_cookie(42)
        uid = verify_auth_cookie(cookie)
        self.assertEqual(uid, 42)

    def test_verify_empty(self):
        self.assertIsNone(verify_auth_cookie(""))
        self.assertIsNone(verify_auth_cookie(None))

    def test_verify_malformed(self):
        self.assertIsNone(verify_auth_cookie("garbage"))
        self.assertIsNone(verify_auth_cookie("a:b"))
        self.assertIsNone(verify_auth_cookie("a:b:c:d"))

    def test_verify_tampered_signature(self):
        cookie = create_auth_cookie(42)
        parts = cookie.split(":")
        parts[2] = "0" * len(parts[2])  # tamper signature
        self.assertIsNone(verify_auth_cookie(":".join(parts)))

    def test_verify_tampered_user_id(self):
        cookie = create_auth_cookie(42)
        parts = cookie.split(":")
        parts[0] = "999"  # change user_id
        self.assertIsNone(verify_auth_cookie(":".join(parts)))

    @patch("core.auth.time")
    def test_expired_cookie(self, mock_time):
        # Create cookie at time 1000
        mock_time.time.return_value = 1000
        cookie = create_auth_cookie(42)
        # Verify at time way in the future (expired)
        mock_time.time.return_value = 1000 + 31 * 24 * 3600  # 31 days later
        self.assertIsNone(verify_auth_cookie(cookie))


if __name__ == "__main__":
    unittest.main()
