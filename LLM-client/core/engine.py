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

import logging
log = logging.getLogger(__name__)

# Load .env configs
SHARED_ENV = os.path.abspath(os.path.join(SHARED_PATH, "config", ".env"))
LOCAL_ENV = os.path.abspath(os.path.join(LLM_CLIENT_ROOT, "config", ".env"))
load_dotenv(dotenv_path=SHARED_ENV)
load_dotenv(dotenv_path=LOCAL_ENV, override=True)

# Imports — after path setup
from memory.chat_store import get_last_session, get_last_session_for_persona, start_chat_session, log_chat_message, get_chat_messages
from memory.context_builder import build_context
from memory.vector_store import store_embedding
from memory.emotion_store import store_emotion_vector
from memory.persona_emotion_store import load_persona_emotion, save_persona_emotion
from memory.persona_store import get_persona
from memory.fact_store import store_fact, store_fact_blobs
from memory.topic_graph import create_topic_relation, link_all_topics, create_entity, link_entity_to_topic, ingest_extracted_knowledge
from memory.topic_store import bump_salience, boost_group_salience
from memory.ingest_pipeline import ingest as ingest_facts
from memory.buffer import conversation_buffer
from echo.corpus_builder import build_corpus
from echo.traits_extractor import extract_traits
from echo.echo_prompt_builder import build_echo_prompt
from core.llm_client import call_llm
from core.prompt_builder import build_system_prompt, build_message_list
from analysis.emotion_handler import EmotionVectorGenerator, PersonaEmotionEngine
from analysis.llm_knowledge_extractor import LLMKnowledgeExtractor
from tools.tool_call_parser import has_tool_calls, execute_tool_calls, ToolCallResult
from tools.tool_registry import get_tool, get_tool_definitions


# Module-level singletons
_emotion_gen = EmotionVectorGenerator()
_persona_engine = PersonaEmotionEngine()
_knowledge_extractor = LLMKnowledgeExtractor()


def run_conversation_turn(
    user_id: int,
    user_input: str,
    persona_id: int = None,
    session_id: int = None,
    nsfw_mode: bool = False,
    incognito: bool = False,
    echo_mode: bool = False,
) -> dict:
    """
    Execute one full conversation turn.

    Args:
        user_id: the user's ID
        user_input: what the user said
        persona_id: numeric persona ID (from user_personalities table)
        session_id: optional existing session ID (will create/resume if None)

    Returns:
        dict with keys: session_id, user_input, assistant_reply, persona_emotions,
                        user_emotions, emotion_description, extracted_knowledge, llm_raw
    """

    # 1. Load persona config from DB
    persona = get_persona(persona_id) if persona_id else None
    if not persona:
        raise ValueError(f"Persona {persona_id} not found")

    # 2. Session management (persona-aware: never resume another persona's session)
    if session_id is None and persona_id is not None:
        session_id = get_last_session_for_persona(user_id, persona_id)
    elif session_id is None:
        session_id = get_last_session(user_id)
    if session_id is None:
        session_id = start_chat_session(user_id, persona_id,
                                        incognito=incognito, nsfw_mode=nsfw_mode)

    memory_scope = persona.get("memory_scope", None)
    domain_access = persona.get("domain_access", None)

    # 2b. Boost group salience if session is in a folder
    if persona_id and not incognito:
        try:
            from memory.chat_store import get_connection as _get_chat_conn
            _conn = _get_chat_conn()
            with _conn.cursor() as _cur:
                _cur.execute("SELECT group_id FROM chat_sessions WHERE id = %s;", (session_id,))
                _row = _cur.fetchone()
                group_id = _row[0] if _row else None
            _conn.close()
            if group_id:
                boost_group_salience(user_id, persona_id, group_id)
        except Exception as e:
            log.debug("Group salience boost skipped: %s", e)

    # 3. Build memory context with domain + persona filtering
    context = build_context(user_id, user_input, memory_scope=memory_scope,
                            persona_id=persona_id, domain_access=domain_access)
    embedded_input = context["embedded_input"]

    # 4. Analyze user emotions
    user_emotions = _emotion_gen.analyze(user_input)
    user_emotion_tone = max(user_emotions, key=user_emotions.get, default="neutral")

    # 5. Load persona emotion state and update it
    persona_state = load_persona_emotion(user_id, persona_id)
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

    # 6. Build echo prompt if in echo mode
    echo_prompt = None
    if echo_mode:
        corpus = build_corpus(user_id, max_messages=300)
        traits = extract_traits(corpus)
        user_name = persona.get("name", "the user")
        echo_prompt = build_echo_prompt(traits, user_name=user_name)

    # 7. Assemble system prompt with all context
    system_prompt = build_system_prompt(
        persona=persona,
        persona_emotion_desc=emotion_description,
        facts=context["facts"],
        similar_memories=context["vectors"],
        related_topics=context["topics"],
        topic_emotions=context.get("topic_emotions", []),
        nsfw_mode=nsfw_mode,
        echo_prompt=echo_prompt,
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
    if nsfw_mode and persona.get("nsfw_capable"):
        user_permission = "adult"
    else:
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
            arguments["persona_id"] = persona_id

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

    # 10-15. Persist everything (skipped in incognito mode)
    message_id = None
    extracted = {"facts": [], "entities": [], "topics": []}

    if not incognito:
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
                "persona_id": persona_id,
                "session_id": session_id,
                "text": user_input,
                "role": "user",
                "memory_class": "session_memory",
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
        save_persona_emotion(user_id, persona_id, new_persona_emotions)

        # 15. Knowledge extraction — extract from user input
        context_msgs = conversation_buffer.get_messages(user_id, session_id, limit=3)
        extracted = _knowledge_extractor.extract_all(
            user_input, role="user",
            source_type="conversation", source_ref=str(message_id),
            context_messages=context_msgs,
            nsfw_mode=nsfw_mode, persona_slug=persona.get("slug"),
        )

        # 15a. Stamp persona_id on emotional-tier facts (conversation-private)
        for blob in extracted["facts"] + extracted["entities"]:
            if blob.get("tier") == "emotional":
                blob["persona_id"] = persona_id

        # 15b-d. Unified ingestion: store facts, populate Neo4j, bump salience
        ingest_facts(user_id, extracted, persona_id=persona_id)

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
        "incognito": incognito,
        "nsfw_mode": nsfw_mode,
        "echo_mode": echo_mode,
        "llm_raw": response.get("raw"),
    }


def process_input(user_input: str, session_id: str = None, user_id: int = 9999,
                   persona_id: int = None, nsfw_mode: bool = False,
                   incognito: bool = False, echo_mode: bool = False) -> str:
    """
    Wrapper for router integration.
    """
    result = run_conversation_turn(
        user_id=user_id,
        user_input=user_input,
        persona_id=persona_id,
        session_id=session_id,
        nsfw_mode=nsfw_mode,
        incognito=incognito,
        echo_mode=echo_mode,
    )
    return result.get("assistant_reply", "[No reply generated]")
