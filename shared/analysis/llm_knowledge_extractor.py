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
LLM Knowledge Extractor
------------------------
Uses the local LLM to extract structured facts from conversation text,
classifying each fact into a privacy tier (identity vs emotional).

Falls back to the regex-based KnowledgeExtractor when:
- The message is too short (<4 words)
- The role is 'assistant' (only topics are useful)
- The LLM call fails or returns invalid JSON

Tier definitions:
  identity  — Echo-safe facts: name, age, location, family, preferences,
               positive/neutral mentions of people
  emotional — chatbot-only facts: negative mentions of people, struggles,
               complaints, emotional confessions
  private   — NOT extracted (NSFW content, therapy confessions, trauma)
"""

import json
import re
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, ".."))
LLM_CLIENT_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "LLM-client"))
for p in [SHARED_PATH, LLM_CLIENT_PATH]:
    if p not in sys.path:
        sys.path.append(p)

from analysis.knowledge_extractor import KnowledgeExtractor
from core.llm_client import call_llm

# Keywords that indicate private content — facts containing these are dropped
PRIVATE_KEYWORDS = [
    "self-harm", "suicide", "suicidal", "kill myself", "cutting myself",
    "sexual", "orgasm", "masturbat", "porn", "fetish", "kink",
    "molest", "rape", "assault", "abuse",
]

# JSON schema for grammar-constrained output
EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "knowledge_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "tier": {"type": "string", "enum": ["identity", "emotional"]},
                            "entity_type": {
                                "type": ["string", "null"],
                                "enum": ["person", "pet", "place", "event", "thing", None],
                            },
                            "valence": {
                                "type": ["string", "null"],
                                "enum": ["positive", "neutral", "negative", None],
                            },
                        },
                        "required": ["text", "tier"],
                        "additionalProperties": False,
                    },
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["facts", "topics"],
            "additionalProperties": False,
        },
    },
}

EXTRACTION_PROMPT = """You are a knowledge extractor. Analyze the user message and extract structured facts.

RULES:
1. Extract facts about the USER (who they are, what they like, people they mention).
2. Each fact gets a tier:
   - "identity": name, age, location, family, hobbies, preferences, positive/neutral mentions of named people
   - "emotional": negative mentions of people, complaints about relationships, personal struggles, frustrations
3. Do NOT extract private content (sexual, self-harm, therapy confessions, trauma details). Return empty facts for those.
4. For named people:
   - Positive or neutral mention → tier "identity", valence "positive" or "neutral"
   - Negative mention (frustration, anger, conflict) → tier "emotional", valence "negative"
5. entity_type: "person", "pet", "place", "event", "thing", or null
6. topics: broad topic categories (e.g. "technology", "family", "fitness", "gaming", "cooking", "music")
7. Keep fact text concise (one sentence max).
8. If the message contains no extractable facts, return empty facts list.

{extra_rules}
Respond with JSON only."""


def _build_extraction_prompt(nsfw_mode=False, persona_slug=None):
    """Build the extraction prompt with optional persona-specific rules."""
    extra = ""
    if persona_slug == "psychiatrist" or nsfw_mode:
        extra = "9. Be EXTRA conservative: do NOT extract emotional confessions, therapy disclosures, or intimate details. When in doubt, skip the fact.\n"
    return EXTRACTION_PROMPT.format(extra_rules=extra)


class LLMKnowledgeExtractor:
    """
    LLM-based knowledge extractor with regex fallback.
    Returns the same schema as KnowledgeExtractor.extract_all().
    """

    def __init__(self):
        self._regex_fallback = KnowledgeExtractor()

    def extract_all(self, text, role="user", source_type="conversation",
                    source_ref=None, context_messages=None,
                    nsfw_mode=False, persona_slug=None) -> dict:
        """
        Extract knowledge from a message.

        Args:
            text: the message content
            role: 'user' or 'assistant'
            source_type: origin of the text
            source_ref: reference ID (message_id, etc.)
            context_messages: last few buffer messages for context
            nsfw_mode: if True, apply extra suppression
            persona_slug: persona identifier for persona-specific rules

        Returns:
            dict with keys: facts, entities, topics, classification, raw_text
        """
        # Assistant messages: only extract topics via regex
        if role != "user":
            return self._regex_fallback.extract_all(
                text, role=role, source_type=source_type, source_ref=source_ref)

        # Short messages: regex fallback
        if len(text.split()) < 4:
            return self._regex_fallback.extract_all(
                text, role=role, source_type=source_type, source_ref=source_ref)

        # Try LLM extraction
        llm_result = self._llm_extract(text, context_messages,
                                       nsfw_mode=nsfw_mode,
                                       persona_slug=persona_slug)

        if llm_result is None:
            # LLM failed — fall back to regex
            return self._regex_fallback.extract_all(
                text, role=role, source_type=source_type, source_ref=source_ref)

        # Build result in the same schema as regex extractor
        facts = []
        entities = []

        for item in llm_result.get("facts", []):
            fact_text = item.get("text", "").strip()
            if not fact_text:
                continue

            # Safety net: drop facts with private keywords
            if self._contains_private_keywords(fact_text):
                continue

            tier = item.get("tier", "identity")
            if tier not in ("identity", "emotional"):
                tier = "identity"

            entity_type = item.get("entity_type")
            valence = item.get("valence")

            blob = {
                "text": fact_text,
                "tier": tier,
                "entity_type": entity_type,
                "confidence": 0.75,
                "tags": ["llm_extracted"],
                "source_type": source_type,
                "source_ref": source_ref,
                "valence": valence,
            }

            if entity_type:
                entities.append(blob)
            else:
                facts.append(blob)

        # Topics from LLM
        llm_topics = llm_result.get("topics", [])
        topics = [{"topic": t, "confidence": 0.7} for t in llm_topics if isinstance(t, str)]

        # Supplement with regex topic detection (catches things LLM might miss)
        regex_result = self._regex_fallback.extract_all(text, role=role)
        regex_topic_names = {t["topic"] for t in regex_result.get("topics", [])}
        llm_topic_names = {t["topic"] for t in topics}
        for rt in regex_result.get("topics", []):
            if rt["topic"] not in llm_topic_names:
                topics.append(rt)

        classification = regex_result.get("classification", ["statement"])

        return {
            "facts": facts,
            "entities": entities,
            "topics": topics[:5],
            "classification": classification,
            "raw_text": text,
        }

    def _llm_extract(self, text, context_messages=None,
                     nsfw_mode=False, persona_slug=None):
        """
        Call the LLM to extract facts. Returns parsed JSON dict or None on failure.
        """
        system_prompt = _build_extraction_prompt(
            nsfw_mode=nsfw_mode, persona_slug=persona_slug)

        messages = [{"role": "system", "content": system_prompt}]

        # Add context messages if available (helps LLM understand references)
        if context_messages:
            for msg in context_messages[-3:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        messages.append({"role": "user", "content": text})

        try:
            response = call_llm(
                messages,
                temperature=0.3,
                max_tokens=256,
                response_format=EXTRACTION_SCHEMA,
            )

            content = response.get("content", "")
            if not content:
                return None

            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                return None
            if "facts" not in parsed:
                return None

            return parsed

        except (json.JSONDecodeError, Exception):
            return None

    def _contains_private_keywords(self, text):
        """Check if text contains any private content keywords."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in PRIVATE_KEYWORDS)
