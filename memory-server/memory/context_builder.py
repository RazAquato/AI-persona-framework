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

from memory.embedding import embed_text
from memory.vector_store import search_similar_vectors
from memory.fact_store import get_facts
from memory.topic_graph import get_related_topics

def build_context(user_id: int, input_text: str, top_k: int = 5) -> dict:
    """
    Constructs a context dictionary containing facts, vector-based matches,
    and topic relationships relevant to the input text.

    Args:
        user_id (int): User ID for personalized memory
        input_text (str): The current message from user or assistant
        top_k (int): Number of top vector matches to retrieve

    Returns:
        dict: {
            'facts': [...],
            'vectors': [...],
            'topics': [...],
            'raw_input': str,
            'embedded_input': list[float]
        }
    """
    embedded = embed_text(input_text)

    # Retrieve top vector matches from Qdrant
    vector_hits = search_similar_vectors(embedded, top_k=top_k)
    vector_results = [
        {
            "payload": hit.payload,
            "score": hit.score
        }
        for hit in vector_hits
    ]

    # Fetch user-specific facts
    user_facts = get_facts(user_id)

    # Use topic links if vector matches have topics
    related_topics = set()
    for hit in vector_hits:
        for topic in hit.payload.get("topics", []):
            related_topics.update(get_related_topics(topic))

    return {
        "facts": user_facts,
        "vectors": vector_results,
        "topics": list(related_topics),
        "raw_input": input_text,
        "embedded_input": embedded
    }

