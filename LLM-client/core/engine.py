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
2. Build memory context (facts, vectors, topics) with tier filtering
3. Analyze user emotions
4. Load + update persona emotion state
5. Assemble prompt via prompt_builder
6. Call LLM
7. Persist everything (chat, embeddings, emotions)
8. Extract knowledge (facts, entities, topics) from both user input and reply
9. Populate Neo4j topic graph
10. Update buffer
"""

import json
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
from memory.fact_store import store_fact
from memory.topic_graph import create_topic_relation, link_all_topics, create_entity, link_entity_to_topic
from memory.buffer import conversation_buffer
from core.llm_client import call_llm
from core.prompt_builder import build_system_prompt, build_message_list
from agents.loader import load_persona_config
from analysis.emotion_handler import EmotionVectorGenerator, PersonaEmotionEngine
from analysis.knowledge_extractor import KnowledgeExtractor
from tools.tool_call_parser import has_tool_calls, execute_tool_calls, ToolCallResult
from tools.tool_registry import get_tool, get_tool_definitions


# Module-level singletons
_emotion_gen = EmotionVectorGenerator()
_persona_engine = PersonaEmotionEngine()
_knowledge_extractor = KnowledgeExtractor()


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
        dict with keys: session_id, user_input, assistant_reply, persona_emotions,
                        user_emotions, emotion_description, extracted_knowledge, llm_raw
    """

    # 1. Session management
    if session_id is None:
        session_id = get_last_session(user_id)
    if session_id is None:
        session_id = start_chat_session(user_id, personality_id)

    # 2. Load persona config (needed for memory_scope)
    persona = load_persona_config(personality_id)
    memory_scope = persona.get("memory_scope", None)

    # 3. Build memory context with tier filtering
    context = build_context(user_id, user_input, memory_scope=memory_scope)
    embedded_input = context["embedded_input"]

    # 4. Analyze user emotions
    user_emotions = _emotion_gen.analyze(user_input)
    user_emotion_tone = max(user_emotions, key=user_emotions.get, default="neutral")

    # 5. Load persona emotion state and update it
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

    # 9. Call LLM (with tool definitions so it can autonomously use tools)
    tool_defs = get_tool_definitions()
    response = call_llm(messages, tools=tool_defs)
    assistant_reply = response["content"]

    # 9a. Handle tool calls from the LLM response
    tool_results = []
    user_permission = persona.get("user_permission", "adult")

    # Native tool calls (from /v1/chat/completions API)
    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            func_info = tc.get("function", {})
            tool_name = func_info.get("name", "")
            try:
                arguments = json.loads(func_info.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                arguments = {}

            tool_func = get_tool(tool_name)
            if tool_func is None:
                tool_results.append(ToolCallResult(
                    tool_name=tool_name, arguments=arguments,
                    success=False, error="Unknown tool: {}".format(tool_name),
                ))
                continue

            # Inject execution context
            arguments["user_id"] = user_id
            arguments["user_permission"] = user_permission

            try:
                result = tool_func(**arguments)
                display = ""
                if tool_name == "generate_image" and isinstance(result, dict):
                    if result.get("success") and result.get("images"):
                        display = "[Image: {}]".format(", ".join(result["images"]))
                tool_results.append(ToolCallResult(
                    tool_name=tool_name, arguments=arguments,
                    success=True, result=result, display_text=display,
                ))
            except Exception as e:
                tool_results.append(ToolCallResult(
                    tool_name=tool_name, arguments=arguments,
                    success=False, error=str(e),
                ))

        # Feed tool results back to the model for a natural text response
        # Build tool result messages in OpenAI format
        tool_result_msgs = []
        # Add the assistant's tool-call message
        tool_call_msg = {
            "role": "assistant",
            "content": assistant_reply or None,
            "tool_calls": response["tool_calls"],
        }
        tool_result_msgs.append(tool_call_msg)

        # Add tool results
        for tc, tr in zip(response["tool_calls"], tool_results):
            if tr.success:
                result_text = tr.display_text or "Tool executed successfully."
            else:
                result_text = "Tool failed: {}".format(tr.error)
            tool_result_msgs.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result_text,
            })

        # Second LLM call: model sees the tool results and produces a natural reply
        followup_messages = messages + tool_result_msgs
        followup = call_llm(followup_messages, tools=tool_defs)
        assistant_reply = followup["content"].strip()

        # Append image paths so the interface can display them
        tool_display = " ".join(tr.display_text for tr in tool_results if tr.display_text)
        if tool_display:
            assistant_reply = (assistant_reply + "\n" + tool_display).strip()

    # Fallback: parse <tool_call> tags from text (for models that use text-based tool calls)
    elif has_tool_calls(assistant_reply):
        assistant_reply, tool_results = execute_tool_calls(
            response_text=assistant_reply,
            tool_getter=get_tool,
            user_id=user_id,
            user_permission=user_permission,
        )

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

    # 15. Knowledge extraction — extract from user input
    extracted = _knowledge_extractor.extract_all(user_input, role="user")

    # 15a. Store extracted facts
    for fact in extracted["facts"]:
        store_fact(
            user_id=user_id,
            fact=fact["text"],
            tags=fact.get("tags", []),
            relevance_score=fact.get("confidence", 0.5),
            source_type="conversation",
            source_ref=str(message_id),
            tier=fact.get("tier", "knowledge"),
            entity_type=fact.get("entity_type"),
        )

    # 15b. Store extracted entities as facts AND in Neo4j
    for entity in extracted["entities"]:
        store_fact(
            user_id=user_id,
            fact=entity["text"],
            tags=entity.get("tags", []),
            relevance_score=entity.get("confidence", 0.5),
            source_type="conversation",
            source_ref=str(message_id),
            tier=entity.get("tier", "knowledge"),
            entity_type=entity.get("entity_type"),
        )
        # Extract entity name from the fact text for Neo4j
        _store_entity_in_graph(user_id, entity)

    # 15c. Populate topic graph
    topic_names = [t["topic"] for t in extracted["topics"]]
    for topic_info in extracted["topics"]:
        create_topic_relation(user_id, topic_info["topic"])

    # 15d. Cross-link topics that appeared in the same message
    if len(topic_names) >= 2:
        link_all_topics(topic_names)

    # 16. Update conversation buffer
    conversation_buffer.add_message(user_id, session_id, "user", user_input)
    conversation_buffer.add_message(user_id, session_id, "assistant", assistant_reply)

    # 17. Return full result
    return {
        "session_id": session_id,
        "user_input": user_input,
        "assistant_reply": assistant_reply,
        "persona_emotions": new_persona_emotions,
        "user_emotions": user_emotions,
        "emotion_description": emotion_description,
        "extracted_knowledge": extracted,
        "tool_results": tool_results,
        "llm_raw": response.get("raw"),
    }


def _store_entity_in_graph(user_id: int, entity: dict):
    """Helper to store an extracted entity in Neo4j."""
    entity_type = entity.get("entity_type")
    text = entity.get("text", "")

    # Try to extract the entity name from the fact text
    # Pattern: "User has a pet named X" or "User's wife is named X"
    import re
    name_match = re.search(r"named (\w+)", text)
    if name_match:
        entity_name = name_match.group(1)
        create_entity(user_id, entity_name, entity_type or "thing")
    elif entity_type:
        # Use first capitalized word as entity name fallback
        words = text.split()
        for w in reversed(words):
            if w[0].isupper() and len(w) > 1:
                create_entity(user_id, w, entity_type)
                break


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
