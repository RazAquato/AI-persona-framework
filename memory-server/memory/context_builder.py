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
Supports tier-based fact filtering based on persona memory_scope config.

Memory scope config (from persona template):
    memory_scope:
        tier1: true        # always included
        tier2: "all" | ["gaming", "technology"]  # topic filter
        tier3: "private"   # always private to this persona
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
    Get facts filtered by the persona's memory scope.

    If no memory_scope is provided, returns all facts (backwards compatible).

    Tier filtering:
    - Tier 1 (identity): always included
    - Tier 2 (knowledge): included if scope says "all" or matches topic domains
    - Tier 3 (relationship): never included here (handled by chat history per-session)
    """
    if memory_scope is None:
        # No scope = return everything (backwards compatible)
        return get_facts(user_id)

    # Always include identity facts
    tiers = ["identity"]

    # Include knowledge tier based on scope config
    tier2_config = memory_scope.get("tier2", "all")
    if tier2_config == "all" or tier2_config is True:
        tiers.append("knowledge")
    elif isinstance(tier2_config, list) and len(tier2_config) > 0:
        # Filter knowledge facts by topic domains
        # For now, include all knowledge facts — topic filtering
        # will be refined when facts get topic tags
        tiers.append("knowledge")

    # Tier 3 (relationship) is intentionally excluded from context_builder.
    # Relationship data comes from the chat history which is already per-session/persona.

    return get_facts_by_tier(user_id, tiers)
