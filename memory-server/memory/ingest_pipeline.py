# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Unified Ingestion Pipeline
--------------------------
Single entry point for all fact ingestion regardless of source.

All sources (conversation, documents, Immich, Mealie, future imports)
should flow through ingest() to guarantee consistent post-storage steps:

    1. Store fact blobs (PostgreSQL) — dedup, tags, topic linking
    2. Populate knowledge graph (Neo4j) — topics, entities, cross-links
    3. Bump topic salience (PostgreSQL) — per-persona relevance tracking
    4. Store vectors (Qdrant) — optional semantic search embeddings

Usage:

    from memory.ingest_pipeline import ingest

    # Conversation extraction (engine.py)
    ingest(user_id, extracted, persona_id=pid)

    # Document ingestion
    ingest(user_id, extracted, source_type="document", source_ref="file.txt")

    # External adapter (Immich, Mealie, etc.)
    ingest(user_id, {"facts": blobs, "entities": [], "topics": []},
           source_type="immich", snapshot=True)

The extracted dict follows the schema returned by KnowledgeExtractor:
    {
        "facts": [{"text", "tier", "tags", "domain", "confidence", ...}],
        "entities": [{"text", "entity_type", "tags", ...}],
        "topics": [{"topic": str, "confidence": float}],
    }
"""

import logging

from memory.fact_store import store_fact_blobs, delete_facts_by_source
from memory.topic_graph import ingest_extracted_knowledge
from memory.topic_store import bump_salience

log = logging.getLogger(__name__)


def ingest(user_id: int, extracted: dict,
           persona_id: int = None,
           source_type: str = None,
           source_ref: str = None,
           snapshot: bool = False) -> dict:
    """
    Canonical entry point for all fact ingestion.

    Args:
        user_id: who owns this data
        extracted: dict with keys {facts, entities, topics}
            - facts: list of fact blob dicts (from make_fact_blob or extractor)
            - entities: list of entity blob dicts (same schema as facts)
            - topics: list of {"topic": str, "confidence": float}
        persona_id: optional — bumps salience for this persona's topic tracking
        source_type: fallback source_type for blobs that don't specify one
        source_ref: fallback source_ref for blobs that don't specify one
        snapshot: if True, delete all existing facts with this source_type
                  before inserting (used by Immich/Mealie full-sync adapters)

    Returns:
        dict with keys:
            facts_stored: number of new facts inserted
            facts_skipped: number of deduped/skipped blobs
            topics_found: number of topics detected
            topic_names: list of topic name strings
    """
    facts = extracted.get("facts", [])
    entities = extracted.get("entities", [])
    topics = extracted.get("topics", [])
    all_blobs = facts + entities

    # 0. Snapshot mode: delete old facts before inserting fresh set
    if snapshot and source_type:
        try:
            deleted = delete_facts_by_source(user_id, source_type)
            log.info("Snapshot delete: removed %d old '%s' facts", deleted, source_type)
        except Exception as e:
            log.warning("Snapshot delete failed for '%s': %s", source_type, e)

    # 1. Store fact blobs — PostgreSQL insert + dedup + topic linking via tags
    stored_ids = []
    if all_blobs:
        stored_ids = store_fact_blobs(user_id, all_blobs,
                                      source_type=source_type,
                                      source_ref=source_ref)

    facts_stored = sum(1 for i in stored_ids if i is not None)
    facts_skipped = len(stored_ids) - facts_stored

    # 2. Populate knowledge graph — Neo4j topics, entities, cross-links
    if topics or entities:
        try:
            ingest_extracted_knowledge(user_id, extracted)
        except Exception as e:
            log.warning("Neo4j ingestion failed: %s", e)

    # 3. Bump topic salience — per-persona relevance tracking
    topic_names = [t["topic"] for t in topics if t.get("topic")]
    if topic_names and persona_id:
        try:
            bump_salience(user_id, persona_id, topic_names)
        except Exception as e:
            log.warning("Salience bump failed: %s", e)

    result = {
        "facts_stored": facts_stored,
        "facts_skipped": facts_skipped,
        "topics_found": len(topic_names),
        "topic_names": topic_names,
    }

    log.info("Ingested %d facts (%d skipped), %d topics for user %d [source=%s]",
             facts_stored, facts_skipped, len(topic_names), user_id,
             source_type or "default")

    return result
