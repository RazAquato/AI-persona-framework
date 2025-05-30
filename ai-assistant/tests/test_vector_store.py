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
        self.metadata = {
            "role": "user",
            "agent": "test-agent",
            "topics": ["vector_test"],
            "joy_level": 0.8
        }

    def test_store_and_search_embedding(self):
        point_id = store_embedding(self.vector, self.metadata)
        self.assertIsInstance(point_id, str)
        
        results = search_similar_vectors(self.vector, top_k=1)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].payload["agent"], "test-agent")

if __name__ == "__main__":
    unittest.main()
