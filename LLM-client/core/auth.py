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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Authentication utilities — password hashing and signed session cookies.

Passwords: PBKDF2-HMAC-SHA256 with random salt (hashlib built-in).
Cookies: HMAC-SHA256 signed, containing user_id + timestamp, 30-day expiry.
"""

import hashlib
import hmac
import os
import secrets
import time

PBKDF2_ITERATIONS = 260_000
COOKIE_MAX_AGE = 30 * 24 * 3600  # 30 days in seconds

# Auth secret loaded from env, or generated per-process (won't survive restarts)
_AUTH_SECRET = os.getenv("AUTH_SECRET", "")
if not _AUTH_SECRET:
    _AUTH_SECRET = secrets.token_hex(32)


def hash_password(password: str) -> str:
    """Hash a password with a random salt. Returns 'salt$hash' hex string."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    )
    return salt + "$" + dk.hex()


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored 'salt$hash' string."""
    if not stored or "$" not in stored:
        return False
    salt, expected_hash = stored.split("$", 1)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(dk.hex(), expected_hash)


def create_auth_cookie(user_id: int) -> str:
    """Create a signed cookie value: 'user_id:timestamp:signature'."""
    ts = str(int(time.time()))
    payload = f"{user_id}:{ts}"
    sig = hmac.new(
        _AUTH_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{payload}:{sig}"


def verify_auth_cookie(cookie_value: str) -> int | None:
    """Verify a signed cookie and return user_id, or None if invalid/expired."""
    if not cookie_value:
        return None
    parts = cookie_value.split(":")
    if len(parts) != 3:
        return None
    user_id_str, ts_str, sig = parts

    # Verify signature
    payload = f"{user_id_str}:{ts_str}"
    expected_sig = hmac.new(
        _AUTH_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None

    # Check expiry
    try:
        ts = int(ts_str)
        if time.time() - ts > COOKIE_MAX_AGE:
            return None
        return int(user_id_str)
    except (ValueError, TypeError):
        return None
