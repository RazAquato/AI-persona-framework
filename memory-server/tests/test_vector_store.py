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
import random

from memory.vector_store import store_embedding, search_similar_vectors

class TestVectorStore(unittest.TestCase):

    def setUp(self):
        # Create a random but deterministic fake vector (384 dims)
        self.vector = [round(random.uniform(-1, 1), 5) for _ in range(384)]
        self.user_id = 9999
        self.metadata = {
            "user_id": self.user_id,
            "role": "user",
            "agent": "test-agent",
            "topics": ["vector_test"],
            "memory_class": "session_memory",
        }

    def test_store_and_search_embedding(self):
        point_id = store_embedding(self.vector, self.metadata)
        self.assertIsInstance(point_id, str)

        results = search_similar_vectors(self.vector, top_k=1, user_id=self.user_id)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].payload["agent"], "test-agent")

    def test_search_requires_user_id(self):
        """search_similar_vectors must reject calls without user_id."""
        with self.assertRaises(ValueError):
            search_similar_vectors(self.vector, top_k=1)

    def test_search_isolation_between_users(self):
        """Vectors stored for user A must not appear in user B's search."""
        vec_a = [round(random.uniform(-1, 1), 5) for _ in range(384)]
        store_embedding(vec_a, {"user_id": 11111, "role": "user", "text": "user A data"})

        results_b = search_similar_vectors(vec_a, top_k=5, user_id=22222)
        user_ids = [r.payload.get("user_id") for r in results_b]
        self.assertNotIn(11111, user_ids)


if __name__ == "__main__":
    unittest.main()
