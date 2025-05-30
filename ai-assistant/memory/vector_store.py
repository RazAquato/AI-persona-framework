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

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

# Load environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

# Qdrant connection parameters
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chat_memory")

# Initialize Qdrant client
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def store_embedding(vector: list, metadata: dict) -> str:
    """
    Stores a single vector with metadata into the Qdrant collection.

    Args:
        vector (list): Embedding vector (float list).
        metadata (dict): Associated payload metadata.

    Returns:
        str: UUID of the inserted point.
    """
    point_id = str(uuid.uuid4())

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata
            )
        ]
    )

    return point_id

def search_similar_vectors(query_vector: list, top_k: int = 5, filters: dict = None):
    """
    Searches for the top-K most similar vectors to the query vector.

    Args:
        query_vector (list): Embedding vector to search with.
        top_k (int): Number of top matches to return.
        filters (dict, optional): Metadata filters for narrowing results.

    Returns:
        list: List of search results with score and payload.
    """
    query_filter = None
    if filters:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
                for key, value in filters.items()
            ]
        )

    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter
    )

    return results
