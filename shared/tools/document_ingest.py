# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Document Ingestion Tool
-----------------------
Reads a text file (txt, md, json), runs it through the knowledge extractor,
and stores extracted facts/entities/topics via the ingestion pipeline.

Callable as a tool from the LLM or as a /command from the user.
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MEMORY_PATH = os.path.join(ROOT, "memory-server")
SHARED_PATH = os.path.join(ROOT, "shared")

for p in [MEMORY_PATH, SHARED_PATH]:
    if p not in sys.path:
        sys.path.append(p)

from analysis.knowledge_extractor import KnowledgeExtractor
from memory.fact_store import store_fact_blobs
from memory.topic_graph import ingest_extracted_knowledge

_extractor = KnowledgeExtractor()

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonl"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


def ingest_document(file_path: str, user_id: int, **kwargs) -> dict:
    """
    Ingest a document file into the knowledge pipeline.

    Args:
        file_path: path to the file to ingest
        user_id: the user who owns the extracted knowledge
        **kwargs: additional args (user_permission injected by engine)

    Returns:
        dict with keys: success, file, facts_stored, topics_found, error
    """
    file_path = os.path.expanduser(file_path)

    if not os.path.isfile(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return {"success": False, "error": f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"}

    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE:
        return {"success": False, "error": f"File too large ({size} bytes). Max: {MAX_FILE_SIZE}"}

    try:
        text = _read_file(file_path, ext)
    except Exception as e:
        return {"success": False, "error": f"Failed to read file: {e}"}

    if not text.strip():
        return {"success": False, "error": "File is empty"}

    # Split into chunks for extraction (one chunk per paragraph or JSON entry)
    chunks = _split_into_chunks(text, ext)
    source_ref = os.path.basename(file_path)

    all_facts = []
    all_topics = []
    total_extracted = {"facts": [], "entities": [], "topics": []}

    for chunk in chunks:
        extracted = _extractor.extract_all(
            chunk, role="user",
            source_type="document", source_ref=source_ref,
        )
        total_extracted["facts"].extend(extracted["facts"])
        total_extracted["entities"].extend(extracted["entities"])
        # Merge topics by name, keeping highest confidence
        for topic in extracted["topics"]:
            existing = next((t for t in total_extracted["topics"] if t["topic"] == topic["topic"]), None)
            if existing:
                existing["confidence"] = max(existing["confidence"], topic["confidence"])
            else:
                total_extracted["topics"].append(topic)

    # Store facts and entities in bulk
    all_blobs = total_extracted["facts"] + total_extracted["entities"]
    stored_ids = store_fact_blobs(user_id, all_blobs,
                                  source_type="document", source_ref=source_ref)
    facts_stored = sum(1 for i in stored_ids if i is not None)

    # Store in Neo4j
    ingest_extracted_knowledge(user_id, total_extracted)
    topics_found = len(total_extracted["topics"])

    return {
        "success": True,
        "file": source_ref,
        "facts_stored": facts_stored,
        "topics_found": topics_found,
        "total_extracted": len(all_blobs),
    }


def _read_file(file_path: str, ext: str) -> str:
    """Read file contents based on extension."""
    with open(file_path, "r", encoding="utf-8") as f:
        if ext == ".json":
            data = json.load(f)
            if isinstance(data, dict):
                return json.dumps(data, indent=2)
            elif isinstance(data, list):
                return "\n".join(json.dumps(item) for item in data)
            return str(data)
        elif ext == ".jsonl":
            lines = f.readlines()
            return "\n".join(lines)
        else:
            return f.read()


def _split_into_chunks(text: str, ext: str) -> list:
    """Split text into processable chunks."""
    if ext in (".json", ".jsonl"):
        # Each line is a chunk
        return [line.strip() for line in text.split("\n") if line.strip()]
    else:
        # Split by double newlines (paragraphs), fall back to single chunks
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text]
        return paragraphs
