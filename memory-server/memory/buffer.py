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

"""
Short-Term Conversation Buffer
-------------------------------
In-memory sliding window of recent conversation turns.
One buffer per (user_id, session_id) pair.

This avoids a DB round-trip for the most recent messages and ensures
the LLM always has immediate context even before messages are persisted.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import threading


class ConversationBuffer:
    """
    Thread-safe in-memory buffer for recent conversation turns.
    Each buffer is keyed by (user_id, session_id).
    """

    def __init__(self, max_turns: int = 20):
        """
        Args:
            max_turns: maximum number of messages (not pairs) to keep per session.
        """
        self.max_turns = max_turns
        self._buffers: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_message(self, user_id: int, session_id: int, role: str, content: str):
        """Add a message to the buffer. Evicts oldest if over max_turns."""
        key = (user_id, session_id)
        msg = {"role": role, "content": content}

        with self._lock:
            buf = self._buffers[key]
            buf.append(msg)
            # Evict oldest messages beyond max_turns
            if len(buf) > self.max_turns:
                self._buffers[key] = buf[-self.max_turns:]

    def get_messages(self, user_id: int, session_id: int, limit: int = None) -> List[dict]:
        """
        Get recent messages from the buffer.

        Args:
            user_id: user ID
            session_id: session ID
            limit: if set, return only the last N messages

        Returns:
            List of {"role": ..., "content": ...} dicts, oldest first.
        """
        key = (user_id, session_id)
        with self._lock:
            buf = self._buffers.get(key, [])
            if limit:
                return list(buf[-limit:])
            return list(buf)

    def clear(self, user_id: int, session_id: int):
        """Clear the buffer for a specific session."""
        key = (user_id, session_id)
        with self._lock:
            self._buffers.pop(key, None)

    def clear_all(self):
        """Clear all buffers (e.g., on server restart)."""
        with self._lock:
            self._buffers.clear()

    def session_count(self) -> int:
        """Number of active buffered sessions."""
        with self._lock:
            return len(self._buffers)


# Global singleton buffer — imported by engine.py
conversation_buffer = ConversationBuffer()
