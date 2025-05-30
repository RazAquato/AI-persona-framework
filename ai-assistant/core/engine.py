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

from memory.chat_store import get_last_session, start_chat_session, log_chat_message, get_chat_messages
from memory.context_builder import build_context
from core.llm_client import call_llm
from agents.loader import load_persona_config
from memory.vector_store import store_embedding

def run_conversation_turn(user_id: int, user_input: str, personality_id: str = "default") -> dict:
    # 1. Get or create chat session
    session_id = get_last_session(user_id)
    if session_id is None:
        session_id = start_chat_session(user_id, personality_id)

    # 2. Build context
    context = build_context(user_id, user_input)
    embedded_input = context["embedded_input"]

    # 3. Store user message + embedding
    log_chat_message(
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=user_input,
        embedding=embedded_input,
        sentiment=None,  # Optional NLP step later
        topics=[]  # Optional NLP step later
    )

    # 4. Load persona prompt
    persona = load_persona_config(personality_id)
    system_prompt = persona.get("system_prompt", "You are a helpful assistant.")

    # 5. Load chat history
    past_messages = get_chat_messages(session_id)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in past_messages:
        messages.append({"role": msg[1], "content": msg[2]})  # role, content

    messages.append({"role": "user", "content": user_input})

    # 6. Call LLM
    response = call_llm(messages)

    # 7. Store assistant response
    assistant_reply = response["content"]
    log_chat_message(
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=assistant_reply,
        embedding=None,
        sentiment=None,
        topics=[]
    )

    # 8. Return full response
    return {
        "session_id": session_id,
        "user_input": user_input,
        "assistant_reply": assistant_reply,
        "llm_raw": response.get("raw")
    }

