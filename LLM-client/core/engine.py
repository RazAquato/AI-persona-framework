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
Conversation Engine
-------------------
Orchestrates a full conversation turn:
1. Session management
2. Build memory context (facts, vectors, topics)
3. Analyze user emotions
4. Load + update persona emotion state
5. Assemble prompt via prompt_builder
6. Call LLM
7. Persist everything (chat, embeddings, emotions)
8. Update buffer
"""

import sys
import os
from dotenv import load_dotenv

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_CLIENT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "memory-server"))
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared"))

for p in [LLM_CLIENT_ROOT, MEMORY_PATH, SHARED_PATH]:
    if p not in sys.path:
        sys.path.append(p)

# Load .env configs
SHARED_ENV = os.path.abspath(os.path.join(SHARED_PATH, "config", ".env"))
LOCAL_ENV = os.path.abspath(os.path.join(LLM_CLIENT_ROOT, "config", ".env"))
load_dotenv(dotenv_path=SHARED_ENV)
load_dotenv(dotenv_path=LOCAL_ENV, override=True)

# Imports — after path setup
from memory.chat_store import get_last_session, start_chat_session, log_chat_message, get_chat_messages
from memory.context_builder import build_context
from memory.vector_store import store_embedding
from memory.emotion_store import store_emotion_vector
from memory.persona_emotion_store import load_persona_emotion, save_persona_emotion
from memory.buffer import conversation_buffer
from core.llm_client import call_llm
from core.prompt_builder import build_system_prompt, build_message_list
from agents.loader import load_persona_config
from analysis.emotion_handler import EmotionVectorGenerator, PersonaEmotionEngine


# Module-level singletons
_emotion_gen = EmotionVectorGenerator()
_persona_engine = PersonaEmotionEngine()


def run_conversation_turn(
    user_id: int,
    user_input: str,
    personality_id: str = "default",
    session_id: int = None,
) -> dict:
    """
    Execute one full conversation turn.

    Args:
        user_id: the user's ID
        user_input: what the user said
        personality_id: which persona to use
        session_id: optional existing session ID (will create/resume if None)

    Returns:
        dict with keys: session_id, user_input, assistant_reply, persona_emotions, user_emotions, llm_raw
    """

    # 1. Session management
    if session_id is None:
        session_id = get_last_session(user_id)
    if session_id is None:
        session_id = start_chat_session(user_id, personality_id)

    # 2. Build memory context (facts, vector matches, topic graph)
    context = build_context(user_id, user_input)
    embedded_input = context["embedded_input"]

    # 3. Analyze user emotions
    user_emotions = _emotion_gen.analyze(user_input)
    user_emotion_tone = max(user_emotions, key=user_emotions.get, default="neutral")

    # 4. Load persona emotion state and update it
    persona_state = load_persona_emotion(user_id, personality_id)
    current_persona_emotions = persona_state["emotions"]
    last_interaction = persona_state["last_updated"]

    new_persona_emotions = _persona_engine.update_persona_emotions(
        current_emotions=current_persona_emotions,
        user_text=user_input,
        user_emotions=user_emotions,
        last_interaction=last_interaction,
    )

    # Generate natural-language emotion description for the prompt
    emotion_description = _persona_engine.describe_emotional_state(new_persona_emotions)

    # 5. Load persona config
    persona = load_persona_config(personality_id)

    # 6. Assemble system prompt with all context
    system_prompt = build_system_prompt(
        persona=persona,
        persona_emotion_desc=emotion_description,
        facts=context["facts"],
        similar_memories=context["vectors"],
        related_topics=context["topics"],
    )

    # 7. Get chat history — use buffer for recent, DB for older
    buffer_limit = int(os.getenv("CHAT_BUFFER_LIMIT", 10))
    buffer_msgs = conversation_buffer.get_messages(user_id, session_id, limit=buffer_limit)

    # If buffer is empty (first turn or restart), fall back to DB history
    if not buffer_msgs:
        db_history = get_chat_messages(session_id)[-buffer_limit:]
    else:
        db_history = []

    # 8. Build final message list
    messages = build_message_list(
        system_prompt=system_prompt,
        chat_history=db_history,
        user_input=user_input,
        buffer_messages=buffer_msgs,
    )

    # 9. Call LLM
    response = call_llm(messages)
    assistant_reply = response["content"]

    # 10. Persist user message
    message_id = log_chat_message(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=user_input,
        embedding=embedded_input,
        sentiment=None,
        topics=[],
    )

    # 11. Store user emotion vector
    store_emotion_vector(message_id, user_emotions, tone=user_emotion_tone)

    # 12. Store embedding in Qdrant for future semantic search
    store_embedding(
        vector=embedded_input,
        metadata={
            "user_id": user_id,
            "session_id": session_id,
            "text": user_input,
            "role": "user",
        },
    )

    # 13. Persist assistant message
    log_chat_message(
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=assistant_reply,
        embedding=None,
        sentiment=None,
        topics=[],
    )

    # 14. Save updated persona emotions
    save_persona_emotion(user_id, personality_id, new_persona_emotions)

    # 15. Update conversation buffer
    conversation_buffer.add_message(user_id, session_id, "user", user_input)
    conversation_buffer.add_message(user_id, session_id, "assistant", assistant_reply)

    # 16. Return full result
    return {
        "session_id": session_id,
        "user_input": user_input,
        "assistant_reply": assistant_reply,
        "persona_emotions": new_persona_emotions,
        "user_emotions": user_emotions,
        "emotion_description": emotion_description,
        "llm_raw": response.get("raw"),
    }


def process_input(user_input: str, session_id: str = None, user_id: int = 9999,
                   personality_id: str = "default") -> str:
    """
    Wrapper for router integration.
    """
    result = run_conversation_turn(
        user_id=user_id,
        user_input=user_input,
        personality_id=personality_id,
        session_id=session_id,
    )
    return result.get("assistant_reply", "[No reply generated]")
