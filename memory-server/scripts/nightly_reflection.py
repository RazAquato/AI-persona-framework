# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Nightly Reflection Job
----------------------
Analyzes today's conversations to classify user emotions per topic.
Runs as a cron job alongside salience decay.

Pipeline:
1. Fetch today's chat messages grouped by topic
2. For each topic with enough context, classify user emotions via LLM
3. Merge classified emotions into user_topic_emotions (running average)
4. Decay old topic emotions that haven't been refreshed

The LLM prompt design is the critical piece — see classify_topic_emotions()
for the prompt template. This is where we decide how the 9B model interprets
emotional nuance from conversation fragments.

Usage:
    python3 scripts/nightly_reflection.py [--dry-run] [--user USER_ID]
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, ".."))
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared"))
LLM_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "LLM-client"))

for p in [MEMORY_PATH, SHARED_PATH, LLM_PATH]:
    if p not in sys.path:
        sys.path.append(p)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.abspath(os.path.join(BASE_DIR, "..", "config", ".env")))

import psycopg2
from memory.topic_store import get_or_create_topic, decay_all_salience
from memory.topic_emotion_store import (
    set_topic_emotions, decay_topic_emotions, VALID_EMOTIONS,
)

PG_CONFIG = {
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
    "database": os.getenv("PG_DATABASE"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
}


def get_connection():
    return psycopg2.connect(**PG_CONFIG)


def get_todays_messages_by_topic(user_id: int = None,
                                  lookback_hours: int = 24) -> dict:
    """
    Fetch recent chat messages grouped by topic.

    Returns: {
        (user_id, topic_name): [
            {"role": "user", "content": "...", "session_id": N},
            ...
        ]
    }
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT m.user_id, m.session_id, m.role, m.content, unnest(m.topics) AS topic
                FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE m.timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s hours'
                  AND s.incognito = FALSE
                  AND m.topics IS NOT NULL
                  AND array_length(m.topics, 1) > 0
            """
            params = [lookback_hours]
            if user_id:
                query += " AND m.user_id = %s"
                params.append(user_id)
            query += " ORDER BY m.timestamp;"

            cur.execute(query, params)
            grouped = {}
            for row in cur.fetchall():
                uid, sid, role, content, topic = row
                key = (uid, topic.lower().strip())
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append({
                    "role": role,
                    "content": content,
                    "session_id": sid,
                })
            return grouped
    finally:
        conn.close()


def classify_topic_emotions(topic_name: str, messages: list) -> list:
    """
    Classify user emotions about a topic from conversation messages.

    Uses the LLM with structured JSON output to classify emotions.
    Falls back to regex-based emotion detection if LLM is unavailable.

    Args:
        topic_name: the topic being analyzed
        messages: list of {"role", "content"} dicts about this topic

    Returns:
        list of {"emotion": str, "intensity": float} dicts

    --- PROMPT DESIGN NOTE ---
    The prompt template below is a skeleton. The actual prompt wording
    needs Kenneth's input to tune for the 9B model's capabilities.
    Key decisions:
    - How much conversation context to include (token budget)
    - Whether to include assistant messages for context or just user messages
    - How to handle topics with very few messages (skip? lower confidence?)
    - Whether to use few-shot examples in the prompt
    - Temperature and max_tokens for classification
    """
    # Filter to user messages only (we're classifying the USER's emotions)
    user_msgs = [m for m in messages if m["role"] == "user"]
    if len(user_msgs) < 2:
        # Too few messages to classify — use regex fallback
        return _regex_fallback(topic_name, user_msgs)

    # Build context string from messages (cap to avoid token overflow)
    context_lines = []
    for msg in messages[-10:]:  # last 10 messages max
        prefix = "User" if msg["role"] == "user" else "AI"
        context_lines.append(f"{prefix}: {msg['content']}")
    context_str = "\n".join(context_lines)

    # --- LLM classification ---
    try:
        from core.llm_client import call_llm

        # JSON schema for grammar-constrained output
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "topic_emotions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "emotions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "emotion": {
                                        "type": "string",
                                        "enum": sorted(VALID_EMOTIONS),
                                    },
                                    "intensity": {
                                        "type": "number",
                                    },
                                },
                                "required": ["emotion", "intensity"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["emotions"],
                    "additionalProperties": False,
                },
            },
        }

        # TODO: Refine this prompt with Kenneth's input
        # This is the critical piece that determines classification quality
        system_prompt = (
            "You are an emotion analyst. Given conversation excerpts about a topic, "
            "classify the USER's emotions toward that topic.\n\n"
            "Rules:\n"
            "1. Only classify emotions the user clearly expresses or implies\n"
            "2. Intensity is 0.0-1.0 (0.3=mild, 0.5=moderate, 0.7=strong, 0.9=intense)\n"
            "3. Return 1-4 emotions maximum — only the most prominent\n"
            "4. A topic can have mixed emotions (love AND frustration)\n"
            "5. If unsure, prefer lower intensity over higher\n"
            "6. Ignore the AI's emotions — only classify the user's\n"
        )

        user_prompt = (
            f"Topic: {topic_name}\n\n"
            f"Conversation excerpts:\n{context_str}\n\n"
            "What emotions does the user express about this topic?"
        )

        response = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=256,
            response_format=response_format,
        )

        raw = response.get("content", "")
        parsed = json.loads(raw)
        emotions = parsed.get("emotions", [])

        # Validate and filter
        result = []
        for em in emotions:
            name = em.get("emotion", "").lower().strip()
            intensity = em.get("intensity", 0.0)
            if name in VALID_EMOTIONS and 0.0 < intensity <= 1.0:
                result.append({"emotion": name, "intensity": round(intensity, 2)})
        return result[:4]  # cap at 4

    except Exception as e:
        print(f"[reflection] LLM classification failed for '{topic_name}': {e}")
        return _regex_fallback(topic_name, user_msgs)


def _regex_fallback(topic_name: str, user_msgs: list) -> list:
    """
    Fallback emotion classification using the existing regex-based
    EmotionVectorGenerator when the LLM is unavailable.
    """
    try:
        from analysis.emotion_handler import EmotionVectorGenerator
        gen = EmotionVectorGenerator()
        combined_text = " ".join(m["content"] for m in user_msgs)
        vector = gen.analyze(combined_text)

        # Convert 18-dim vector to top emotions
        result = []
        for emotion, intensity in sorted(vector.items(), key=lambda x: -x[1]):
            if intensity >= 0.2 and emotion in VALID_EMOTIONS:
                result.append({"emotion": emotion, "intensity": round(intensity, 2)})
            if len(result) >= 4:
                break
        return result
    except Exception:
        return []


def run_reflection(user_id: int = None, dry_run: bool = False,
                   lookback_hours: int = 24):
    """
    Run the full nightly reflection pipeline.

    1. Fetch today's messages by topic
    2. Classify emotions per topic
    3. Store/merge into user_topic_emotions
    4. Decay old emotions
    5. Decay salience (Phase 2)
    """
    print(f"[reflection] Starting reflection (lookback={lookback_hours}h, "
          f"user={user_id or 'all'}, dry_run={dry_run})")

    # 1. Get messages grouped by topic
    grouped = get_todays_messages_by_topic(user_id, lookback_hours)
    print(f"[reflection] Found {len(grouped)} user-topic pairs to analyze")

    classified_count = 0
    emotion_count = 0

    # 2-3. Classify and store for each user-topic pair
    for (uid, topic_name), messages in grouped.items():
        emotions = classify_topic_emotions(topic_name, messages)
        if not emotions:
            continue

        classified_count += 1
        emotion_count += len(emotions)

        if dry_run:
            print(f"  [{uid}] {topic_name}: "
                  + ", ".join(f"{e['emotion']}={e['intensity']}" for e in emotions))
            continue

        # Get topic_id (already exists from salience tracking)
        topic_id = get_or_create_topic(uid, topic_name)
        if topic_id:
            set_topic_emotions(uid, topic_id, emotions, source="reflection")

    print(f"[reflection] Classified {classified_count} topics, "
          f"{emotion_count} emotions total")

    # 4. Decay old topic emotions
    if not dry_run:
        decayed, removed = decay_topic_emotions()
        print(f"[reflection] Emotion decay: {decayed} decayed, {removed} removed")

    # 5. Decay salience (piggyback on nightly job)
    if not dry_run:
        sal_updated = decay_all_salience()
        print(f"[reflection] Salience decay: {sal_updated} records updated")

    print("[reflection] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nightly reflection job")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print classifications without storing")
    parser.add_argument("--user", type=int, default=None,
                        help="Run for a specific user ID only")
    parser.add_argument("--lookback", type=int, default=24,
                        help="Hours of messages to analyze (default: 24)")
    args = parser.parse_args()

    run_reflection(user_id=args.user, dry_run=args.dry_run,
                   lookback_hours=args.lookback)
