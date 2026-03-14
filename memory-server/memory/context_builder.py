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

Domain-based access control:
  - Each persona has a domain_access list (e.g., ["physical", "hobbies", "work"])
  - Facts with a domain require the persona to have access to that domain
  - Facts with NULL domain are always visible (backward compat for uncategorized facts)
  - Persona-private facts (persona_id set) only visible to the owning persona

Echo sees identity only (filtered in corpus_builder).
"""

from memory.embedding import embed_text
from memory.vector_store import search_similar_vectors
from memory.fact_store import get_accessible_facts
from memory.topic_graph import get_related_topics, get_user_topics
from memory.topic_store import get_salient_fact_ids, get_persona_salience
from memory.topic_emotion_store import get_user_topic_emotions

SALIENCE_THRESHOLD = 0.2


def build_context(user_id: int, input_text: str, top_k: int = 5,
                  memory_scope: dict = None, persona_id: int = None,
                  domain_access: list = None) -> dict:
    """
    Constructs a context dictionary containing facts, vector-based matches,
    and topic relationships relevant to the input text.

    Salience filtering:
      - Facts linked to topics with salience >= threshold are included
      - Facts with no linked topics are always included (backward compat)
      - Facts linked to topics all below threshold are excluded
      - Facts are ordered by max topic salience (most relevant first)

    Args:
        user_id: User ID for personalized memory
        input_text: The current message from user or assistant
        top_k: Number of top vector matches to retrieve
        memory_scope: Legacy persona memory scope config (ignored)
        persona_id: Persona ID for persona-private fact filtering
        domain_access: List of domain names this persona can see

    Returns:
        dict with keys: facts, vectors, topics, user_topics, topic_emotions,
                        raw_input, embedded_input
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

    # Fetch user facts with domain + persona filtering
    all_facts = get_accessible_facts(user_id, persona_id=persona_id,
                                     domain_access=domain_access)

    # Apply salience filtering if persona_id is available
    if persona_id and all_facts:
        salient_ids = get_salient_fact_ids(user_id, persona_id,
                                           min_salience=SALIENCE_THRESHOLD)
        # Get IDs of all facts that have any topic links (to distinguish
        # "no topics" from "topics below threshold")
        from memory.topic_store import get_connection
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                fact_ids_list = [f[0] for f in all_facts]
                if fact_ids_list:
                    cur.execute("""
                        SELECT DISTINCT fact_id FROM fact_topics
                        WHERE fact_id = ANY(%s);
                    """, (fact_ids_list,))
                    linked_fact_ids = {r[0] for r in cur.fetchall()}
                else:
                    linked_fact_ids = set()
        finally:
            conn.close()

        # Filter: keep facts with no topics (always visible) or salient topics
        user_facts = [
            f for f in all_facts
            if f[0] not in linked_fact_ids or f[0] in salient_ids
        ]

        # Order by salience: salient facts first, then unlinked facts
        user_facts.sort(key=lambda f: (
            0 if f[0] in salient_ids else 1,  # salient first
            f[0] not in linked_fact_ids,       # linked but salient before unlinked
        ))
    else:
        user_facts = all_facts

    # Get user's top topics for context
    user_topics = get_user_topics(user_id, limit=10)

    # Use topic links if vector matches have topics (scoped to this user)
    related_topics = set()
    for hit in vector_hits:
        for topic in hit.payload.get("topics", []):
            related_topics.update(get_related_topics(topic, user_id=user_id))

    # Get user's emotional profile per topic (from nightly reflection)
    topic_emotions = get_user_topic_emotions(user_id, min_intensity=0.2)

    return {
        "facts": user_facts,
        "vectors": vector_results,
        "topics": list(related_topics),
        "user_topics": user_topics,
        "topic_emotions": topic_emotions,
        "raw_input": input_text,
        "embedded_input": embedded,
    }
