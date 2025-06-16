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

import sys
import os
from dotenv import load_dotenv

# Add memory-server to path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_CLIENT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(LLM_CLIENT_ROOT)
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "memory-server"))
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared"))
sys.path.append(MEMORY_PATH)
sys.path.append(SHARED_PATH)

# Load .env configs
SHARED_ENV = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared", "config", ".env"))
LOCAL_ENV = os.path.abspath(os.path.join(BASE_DIR, "..", "config", ".env"))
load_dotenv(dotenv_path=SHARED_ENV)
load_dotenv(dotenv_path=LOCAL_ENV, override=True)

# Paths and imports
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from memory.chat_store import get_last_session, start_chat_session, log_chat_message, get_chat_messages
from memory.context_builder import build_context
from core.llm_client import call_llm
from agents.loader import load_persona_config
from memory.vector_store import store_embedding
from memory.emotion_store import store_emotion_vector
from analysis.emotion_handler import EmotionVectorGenerator



def run_conversation_turn(user_id: int, user_input: str, personality_id: str = "default") -> dict:
    # 1. Get or create chat session
    session_id = get_last_session(user_id)
    if session_id is None:
        session_id = start_chat_session(user_id, personality_id)

    # 2. Build context
    context = build_context(user_id, user_input)
    embedded_input = context["embedded_input"]

    # 3. Analyze emotion
    emotion_gen = EmotionVectorGenerator()
    emotion_vector = emotion_gen.analyze(user_input)
    emotion_tone = max(emotion_vector, key=emotion_vector.get, default="neutral")

    # 4. Store user message
    message_id = log_chat_message(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=user_input,
        embedding=embedded_input,
        sentiment=None,
        topics=[]
    )

    # 5. Store emotion vector in metadata
    store_emotion_vector(message_id, emotion_vector, tone=emotion_tone)

    # 6. Load persona and prompt
    persona = load_persona_config(personality_id)
    system_prompt = persona.get("system_prompt", "You are a helpful assistant.")

    # 7. Load chat history
    past_messages = get_chat_messages(session_id)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in past_messages:
        messages.append({"role": msg[1], "content": msg[2]})  # role, content

    messages.append({"role": "user", "content": user_input})

    # 8. Call LLM
    response = call_llm(messages)
    assistant_reply = response["content"]

    # 9. Store assistant message (no emotion tagging for now)
    log_chat_message(
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=assistant_reply,
        embedding=None,
        sentiment=None,
        topics=[]
    )

    # 10. Return response
    return {
        "session_id": session_id,
        "user_input": user_input,
        "assistant_reply": assistant_reply,
        "llm_raw": response.get("raw")
    }

def process_input(user_input: str, session_id: str = None, user_id: int = 9999) -> str:
    """
    Wrapper for router integration. Looks like 'engine.process_input'.
    """
    result = run_conversation_turn(user_id=user_id, user_input=user_input)
    return result.get("assistant_reply", "[No reply generated]")
