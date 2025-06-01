import os
import unittest
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

class TestQdrantInit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
        load_dotenv(dotenv_path=config_path)

        cls.collection_name = os.getenv("QDRANT_COLLECTION", "chat_memory")
        cls.vector_size = int(os.getenv("VECTOR_SIZE", 384))

        cls.qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST"),
            port=int(os.getenv("QDRANT_PORT"))
        )

    def test_collection_exists(self):
        self.assertTrue(
            self.qdrant.collection_exists(self.collection_name),
            f"Collection '{self.collection_name}' does not exist in Qdrant."
        )

    def test_collection_vector_config(self):
        info = self.qdrant.get_collection(self.collection_name)
        vectors = info.config.params.vectors

        self.assertEqual(vectors.size, self.vector_size)
        self.assertEqual(vectors.distance, models.Distance.COSINE)

    def test_no_unexpected_collections(self):
        expected = {self.collection_name}
        actual = {col.name for col in self.qdrant.get_collections().collections}
        extra = actual - expected
        self.assertFalse(extra, f"Unexpected Qdrant collections: {extra}")

if __name__ == "__main__":
    unittest.main()

