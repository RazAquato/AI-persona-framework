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
from dotenv import load_dotenv

from memory.embedding import embed_text

class TestEmbedding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
        load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

    def test_embed_text_returns_vector(self):
        text = "The quick brown fox jumps over the lazy dog."
        vector = embed_text(text)
        self.assertIsInstance(vector, list, "Embedding should return a list")
        self.assertGreater(len(vector), 0, "Embedding vector should not be empty")
        self.assertTrue(all(isinstance(x, float) for x in vector), "All elements in vector should be floats")

if __name__ == '__main__':
    unittest.main()

