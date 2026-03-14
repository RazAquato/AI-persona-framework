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
Context Builder
---------------
Aggregates all memory layers into a context dict for the engine.

Two-tier fact model:
  - identity: Echo-safe facts (name, preferences, positive people mentions)
  - emotional: chatbot-only facts (negative people mentions, struggles)

All chatbot personas see both tiers. Echo sees identity only (filtered in corpus_builder).
"""

from memory.embedding import embed_text
from memory.vector_store import search_similar_vectors
from memory.fact_store import get_facts, get_facts_by_tier
from memory.topic_graph import get_related_topics, get_user_topics


def build_context(user_id: int, input_text: str, top_k: int = 5,
                  memory_scope: dict = None) -> dict:
    """
    Constructs a context dictionary containing facts, vector-based matches,
    and topic relationships relevant to the input text.

    Args:
        user_id: User ID for personalized memory
        input_text: The current message from user or assistant
        top_k: Number of top vector matches to retrieve
        memory_scope: Optional persona memory scope config for tier filtering

    Returns:
        dict with keys: facts, vectors, topics, user_topics, raw_input, embedded_input
    """
    embedded = embed_text(input_text)

    # Retrieve top vector matches from Qdrant (filtered to this user)
    vector_hits = search_similar_vectors(embedded, top_k=top_k, user_id=user_id)
    vector_results = [
        {
            "payload": hit.payload,
            "score": hit.score
        }
        for hit in vector_hits
    ]

    # Fetch user facts with tier filtering
    user_facts = _get_scoped_facts(user_id, memory_scope)

    # Get user's top topics for context
    user_topics = get_user_topics(user_id, limit=10)

    # Use topic links if vector matches have topics (scoped to this user)
    related_topics = set()
    for hit in vector_hits:
        for topic in hit.payload.get("topics", []):
            related_topics.update(get_related_topics(topic, user_id=user_id))

    return {
        "facts": user_facts,
        "vectors": vector_results,
        "topics": list(related_topics),
        "user_topics": user_topics,
        "raw_input": input_text,
        "embedded_input": embedded,
    }


def _get_scoped_facts(user_id: int, memory_scope: dict = None) -> list:
    """
    Get facts for chatbot persona context.

    All chatbot personas see both identity and emotional facts.
    The memory_scope parameter is accepted for backwards compatibility but
    no longer affects tier filtering — the two-tier model gives all personas
    full access. Echo privacy filtering happens in corpus_builder instead.
    """
    return get_facts(user_id)
