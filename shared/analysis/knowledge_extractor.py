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
Knowledge Extractor
-------------------
Extracts structured knowledge from conversation text:
- Facts (statements about the user or the world)
- Entities (people, pets, places, events, things)
- Topics (subject areas for Neo4j graph)
- Message classification (fact, opinion, memory, preference, question, greeting)

Each extracted item includes:
- text: the extracted knowledge
- tier: identity / emotional
- entity_type: person / pet / place / event / thing / None
- topics: list of topic strings
- confidence: 0.0-1.0 extraction confidence
- tags: list of classification tags

This module uses pattern matching and keyword analysis. It does NOT require
an LLM call — it runs synchronously on every message for real-time extraction.
A future LLM-based extractor can supplement this for deeper understanding.
"""

import re
from typing import List, Dict, Optional


# --- Identity patterns (Tier 1) ---
# These extract core facts about who the user IS.

IDENTITY_PATTERNS = [
    # Name patterns
    (r"\bmy name is (\w+)", "identity", "person", "User's name is {0}"),
    (r"\bi(?:'?m| am) (\w+)", "identity", "person", "User may be named {0}"),
    (r"\bcall me (\w+)", "identity", "person", "User wants to be called {0}"),
    # Age
    (r"\bi(?:'?m| am) (\d{1,3}) years? old", "identity", None, "User is {0} years old"),
    (r"\bmy age is (\d{1,3})", "identity", None, "User is {0} years old"),
    # Family
    (r"\bmy (wife|husband|partner|spouse)(?:'?s)? (?:name is |is named |is called |is )(\w+)", "identity", "person", "User's {0} is named {1}"),
    (r"\bmy (son|daughter|child|kid)(?:'?s)? (?:name is |is named |is called |is )(\w+)", "identity", "person", "User's {0} is named {1}"),
    (r"\bmy (mother|father|mom|dad|parent)(?:'?s)? (?:name is |is named |is called |is )(\w+)", "identity", "person", "User's {0} is named {1}"),
    (r"\bmy (brother|sister|sibling)(?:'?s)? (?:name is |is named |is called |is )(\w+)", "identity", "person", "User's {0} is named {1}"),
    (r"\bi have (\d+) (kids?|children|sons?|daughters?)", "identity", None, "User has {0} {1}"),
    # Location
    (r"\bi live in (.+?)(?:\.|,|$)", "identity", "place", "User lives in {0}"),
    (r"\bi(?:'?m| am) from (.+?)(?:\.|,|$)", "identity", "place", "User is from {0}"),
    # Occupation
    (r"\bi work (?:as|at|in|for) (.+?)(?:\.|,|$)", "identity", None, "User works {0}"),
    (r"\bi(?:'?m| am) an? (.+?(?:engineer|developer|teacher|doctor|nurse|designer|manager|student|scientist|artist|writer|chef|driver|pilot|lawyer|accountant|programmer))", "identity", None, "User is a {0}"),
    # Core traits
    (r"\bi have (autism|adhd|add|dyslexia|anxiety|depression)", "identity", None, "User has {0}"),
]

# --- Entity patterns ---
# These extract references to named entities.

ENTITY_PATTERNS = [
    # Pets
    (r"\bmy (?:dog|cat|pet|bird|fish|hamster|rabbit|parrot)(?:'?s)? (?:name is |is named |is called |is |named |called )(\w+)", "pet", "User has a pet named {0}"),
    (r"\b(\w+) (?:is |was )my (?:dog|cat|pet|bird)", "pet", "User has a pet named {0}"),
    # People (generic)
    (r"\bmy (?:friend|best friend|buddy|colleague|boss|mentor)(?:'?s)? (?:name is |is named |is called |is )(\w+)", "person", "User's friend/colleague is named {0}"),
    # Places
    (r"\bi (?:went|traveled|visited|been) to (.+?)(?:\.|,|$)", "place", "User visited {0}"),
    (r"\bwe (?:went|traveled|visited) to (.+?)(?:\.|,|$)", "place", "User visited {0}"),
    # Events
    (r"\b(?:when|last|during) (?:my|our) (wedding|birthday|graduation|vacation|trip|holiday|christmas|anniversary)", "event", "User references their {0}"),
]

# --- Preference patterns (Tier 1 or 2) ---

PREFERENCE_PATTERNS = [
    (r"\bi (?:love|really like|adore|enjoy) (.+?)(?:\.|,|!|$)", "User loves {0}"),
    (r"\bi (?:hate|dislike|can't stand|despise) (.+?)(?:\.|,|!|$)", "User dislikes {0}"),
    (r"\bmy favorite (.+?) is (.+?)(?:\.|,|$)", "User's favorite {0} is {1}"),
    (r"\bi prefer (.+?) (?:over|to|instead of) (.+?)(?:\.|,|$)", "User prefers {0} over {1}"),
    (r"\bi(?:'?m| am) (?:really )?into (.+?)(?:\.|,|$)", "User is into {0}"),
    (r"\bi(?:'?m| am) interested in (.+?)(?:\.|,|$)", "User is interested in {0}"),
]

# --- Topic detection ---
# Broad topic categories for Neo4j graph building.

TOPIC_KEYWORDS = {
    "technology": ["computer", "software", "hardware", "programming", "code", "tech", "ai",
                    "machine learning", "server", "linux", "windows", "python", "javascript",
                    "database", "network", "gpu", "cpu", "raspberry pi", "arduino"],
    "gaming": ["game", "gaming", "playstation", "xbox", "nintendo", "steam", "pc game",
                "rpg", "fps", "mmorpg", "minecraft", "fortnite", "valorant", "esport"],
    "fitness": ["workout", "exercise", "gym", "running", "lifting", "weight", "muscle",
                 "cardio", "protein", "diet", "health", "training", "yoga", "swimming"],
    "cooking": ["cook", "recipe", "food", "baking", "kitchen", "meal", "ingredient",
                 "dinner", "lunch", "breakfast", "restaurant", "chef"],
    "music": ["music", "song", "band", "guitar", "piano", "drum", "concert", "album",
               "spotify", "playlist", "singing", "instrument"],
    "nature": ["hiking", "camping", "mountain", "forest", "nature", "outdoor", "trail",
                "wildlife", "garden", "plant", "tree", "flower", "bird watching"],
    "science": ["science", "physics", "chemistry", "biology", "research", "experiment",
                 "theory", "hypothesis", "discovery", "quantum", "space", "astronomy"],
    "art": ["drawing", "painting", "art", "sketch", "canvas", "sculpture", "design",
             "illustration", "creative", "photography", "camera"],
    "family": ["family", "kids", "children", "parent", "wife", "husband", "son", "daughter",
                "brother", "sister", "mother", "father", "grandparent"],
    "work": ["work", "job", "career", "office", "meeting", "project", "deadline",
              "client", "colleague", "boss", "salary", "promotion", "business"],
    "mental_health": ["anxiety", "depression", "stress", "therapy", "counseling", "mental health",
                       "mindfulness", "meditation", "self-care", "burnout", "overwhelmed"],
    "home_automation": ["home assistant", "smart home", "iot", "mqtt", "sensor", "automation",
                         "zigbee", "z-wave", "esphome", "tasmota", "homekit"],
    "woodworking": ["woodworking", "scroll saw", "lathe", "carpentry", "wood", "lumber",
                     "joinery", "dovetail", "plywood", "furniture", "carving"],
    "vehicles": ["car", "motorcycle", "bike", "driving", "engine", "electric vehicle",
                  "tesla", "mechanic", "repair", "road trip"],
    "education": ["learning", "study", "course", "school", "university", "degree",
                   "tutorial", "lecture", "exam", "homework", "teaching"],
    "movies_tv": ["movie", "film", "series", "tv show", "netflix", "streaming",
                   "cinema", "actor", "director", "season", "episode"],
    "pets": ["dog", "cat", "pet", "puppy", "kitten", "vet", "walk", "breed",
              "animal", "aquarium", "terrarium"],
    "finance": ["money", "budget", "invest", "stock", "crypto", "savings", "bank",
                 "mortgage", "tax", "income", "expense", "retirement"],
    "travel": ["travel", "vacation", "flight", "hotel", "destination", "passport",
                "tourist", "backpack", "country", "city", "abroad"],
}

# --- Message classification ---

CLASSIFICATION_PATTERNS = {
    "question": [r"\?$", r"^(?:what|where|when|who|why|how|can|could|would|should|is|are|do|does|did)\b"],
    "greeting": [r"^(?:hi|hello|hey|good morning|good evening|good afternoon|howdy|sup|yo)\b"],
    "farewell": [r"^(?:bye|goodbye|good night|see you|later|cya|gotta go|take care)\b"],
    "memory": [r"\bremember when\b", r"\blast (?:time|year|month|week)\b", r"\bback in\b",
                r"\byears? ago\b", r"\bused to\b", r"\bwhen i was\b"],
    "opinion": [r"\bi think\b", r"\bi believe\b", r"\bin my opinion\b", r"\bi feel like\b",
                 r"\bpersonally\b", r"\bif you ask me\b"],
    "preference": [r"\bi (?:love|like|prefer|enjoy|hate|dislike)\b", r"\bmy favorite\b"],
    "fact": [r"\bi (?:am|have|work|live|was born)\b", r"\bmy (?:name|age|job)\b"],
}


# --- Domain classification by keyword heuristic ---
# Maps TOPIC_KEYWORDS categories to knowledge domains.

TOPIC_TO_DOMAIN = {
    "family": "family",
    "fitness": "physical",
    "cooking": "hobbies",
    "gaming": "hobbies",
    "music": "hobbies",
    "art": "hobbies",
    "woodworking": "hobbies",
    "nature": "hobbies",
    "pets": "hobbies",
    "work": "work",
    "technology": "work",
    "education": "work",
    "mental_health": "emotional",
    "travel": "memories",
    "movies_tv": "hobbies",
    "vehicles": "hobbies",
    "finance": "work",
    "science": "work",
    "home_automation": "work",
}

# Keyword sets for direct domain classification on fact text
DOMAIN_KEYWORDS = {
    "family": ["wife", "husband", "partner", "spouse", "son", "daughter", "child",
               "kids", "children", "mother", "father", "mom", "dad", "parent",
               "brother", "sister", "sibling", "family", "grandparent", "grandmother",
               "grandfather", "uncle", "aunt", "cousin", "nephew", "niece"],
    "physical": ["workout", "exercise", "gym", "running", "lifting", "weight",
                 "muscle", "cardio", "protein", "diet", "health", "training",
                 "yoga", "swimming", "fitness", "medical", "doctor", "hospital",
                 "injury", "pain", "sick", "illness"],
    "hobbies": ["cook", "recipe", "food", "baking", "game", "gaming", "guitar",
                "piano", "music", "song", "band", "concert", "painting", "drawing",
                "art", "photography", "camera", "football", "soccer", "basketball",
                "tennis", "hiking", "camping", "fishing"],
    "work": ["job", "career", "office", "project", "deadline", "client",
             "colleague", "boss", "salary", "promotion", "business", "company",
             "programming", "code", "software", "meeting"],
    "emotional": ["depressed", "anxious", "stress", "therapy", "counseling",
                  "overwhelmed", "burnout", "struggle", "frustrated", "lonely",
                  "angry", "sad", "worried", "scared"],
    "memories": ["remember", "years ago", "back in", "used to", "when i was",
                 "vacation", "trip", "travel", "visited", "moved", "graduated",
                 "wedding", "birthday", "anniversary", "milestone"],
}


def classify_domain(text: str) -> str | None:
    """Classify a fact's knowledge domain from its text content. Returns domain name or None."""
    text_lower = text.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            scores[domain] = matches
    if scores:
        return max(scores, key=scores.get)
    return None


class KnowledgeExtractor:
    """
    Extracts structured knowledge from conversation text.
    Runs synchronously — no LLM needed.
    """

    def extract_all(self, text: str, role: str = "user",
                    source_type: str = "conversation",
                    source_ref: str = None) -> dict:
        """
        Run all extraction passes on a message.

        Args:
            text: the message content
            role: 'user' or 'assistant'
            source_type: origin of the text (conversation, document, image_analysis, etc.)
            source_ref: reference ID (message_id, filename, etc.)

        Returns:
            dict with keys: facts, entities, topics, classification, raw_text
        """
        facts = []
        entities = []

        # Only extract identity/preference facts from user messages
        if role == "user":
            facts.extend(self._extract_identity_facts(text))
            facts.extend(self._extract_preference_facts(text))
            entities.extend(self._extract_entities(text))

        # Stamp source metadata onto all extracted items
        for item in facts + entities:
            item["source_type"] = source_type
            item["source_ref"] = source_ref

        topics = self._detect_topics(text)
        classification = self._classify_message(text)

        return {
            "facts": facts,
            "entities": entities,
            "topics": topics,
            "classification": classification,
            "raw_text": text,
        }

    def _extract_identity_facts(self, text: str) -> List[dict]:
        """Extract identity-level facts (Tier 1)."""
        results = []
        text_lower = text.lower()

        for pattern, tier, entity_type, template in IDENTITY_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                # Clean up captured groups
                groups = tuple(g.strip() for g in groups if g)
                if groups:
                    fact_text = template.format(*groups)
                    # Capitalize proper nouns
                    if entity_type in ("person", "place"):
                        fact_text = self._capitalize_names(fact_text)
                    results.append({
                        "text": fact_text,
                        "tier": tier,
                        "entity_type": entity_type,
                        "confidence": 0.8,
                        "tags": ["auto_extracted", "identity"],
                        "source_pattern": pattern,
                        "domain": classify_domain(fact_text),
                    })

        return results

    def _extract_entities(self, text: str) -> List[dict]:
        """Extract entity references (people, pets, places, events)."""
        results = []
        text_lower = text.lower()

        for pattern, entity_type, template in ENTITY_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                groups = tuple(g.strip() for g in match.groups() if g)
                if groups:
                    entity_text = template.format(*groups)
                    entity_text = self._capitalize_names(entity_text)
                    results.append({
                        "text": entity_text,
                        "tier": "identity",
                        "entity_type": entity_type,
                        "confidence": 0.7,
                        "tags": ["auto_extracted", "entity", entity_type],
                        "domain": classify_domain(entity_text),
                    })

        return results

    def _extract_preference_facts(self, text: str) -> List[dict]:
        """Extract user preferences and interests."""
        results = []
        text_lower = text.lower()

        for pattern, template in PREFERENCE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                groups = tuple(g.strip() for g in match.groups() if g)
                if groups:
                    fact_text = template.format(*groups)
                    results.append({
                        "text": fact_text,
                        "tier": "identity",
                        "entity_type": None,
                        "confidence": 0.7,
                        "tags": ["auto_extracted", "preference"],
                        "domain": classify_domain(fact_text),
                    })

        return results

    def _detect_topics(self, text: str) -> List[dict]:
        """
        Detect topic areas from text.
        Returns list of {topic, confidence} dicts.
        """
        text_lower = text.lower()
        detected = []

        for topic, keywords in TOPIC_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                # Confidence based on number of keyword matches
                confidence = min(1.0, 0.4 + (matches * 0.2))
                detected.append({
                    "topic": topic,
                    "confidence": round(confidence, 2),
                })

        # Sort by confidence descending
        detected.sort(key=lambda x: x["confidence"], reverse=True)
        return detected[:5]  # Cap at 5 topics per message

    def _classify_message(self, text: str) -> List[str]:
        """
        Classify the message type(s).
        A message can have multiple classifications (e.g., question + memory).
        """
        text_lower = text.lower().strip()
        classifications = []

        for cls, patterns in CLASSIFICATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    if cls not in classifications:
                        classifications.append(cls)
                    break

        if not classifications:
            classifications.append("statement")

        return classifications

    def _capitalize_names(self, text: str) -> str:
        """Capitalize likely proper nouns in extracted text."""
        def cap_last_word(match):
            """Capitalize only the final word (the proper noun) in the match."""
            full = match.group(0)
            name = match.group(1)
            return full[:-len(name)] + name.capitalize()

        # Capitalize the name after "is named X", "named X", "called X", "is X"
        # Use longest-first matching to avoid "is " eating "named"
        text = re.sub(r"(?:is named |is called |named |called |is )(\w+)$", cap_last_word, text)
        # Also handle mid-sentence names
        text = re.sub(r"(?:is named |is called |named |called )(\w+)", cap_last_word, text)
        # Capitalize locations — title-case each word (proper nouns in place names)
        text = re.sub(r"(?:lives in |is from )(.+)",
                       lambda m: m.group(0)[:-len(m.group(1))] + m.group(1).title(), text)
        # For "visited X", only capitalize the place name (first word)
        text = re.sub(r"(visited )(\w+)",
                       lambda m: m.group(1) + m.group(2).capitalize(), text)
        return text
