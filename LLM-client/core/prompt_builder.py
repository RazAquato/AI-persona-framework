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
Prompt Builder
--------------
Single source of truth for assembling the full prompt.
Takes persona config + memory context + emotion state + chat history
and produces the message list that gets sent to the LLM.
"""

from typing import Dict, List, Optional


def build_system_prompt(
    persona: dict,
    persona_emotion_desc: str = "",
    facts: list = None,
    similar_memories: list = None,
    related_topics: list = None,
    nsfw_mode: bool = False,
) -> str:
    """
    Assemble the system prompt from persona config + memory layers + emotion state.

    Args:
        persona: dict with at least "system_prompt" key
        persona_emotion_desc: natural-language emotion description from PersonaEmotionEngine
        facts: list of user fact tuples from fact_store [(id, text, tags, score), ...]
        similar_memories: list of dicts with payload/score from vector search
        related_topics: list of topic name strings from Neo4j

    Returns:
        Complete system prompt string.
    """
    parts = []

    # 1. Core persona identity
    base_prompt = persona.get("system_prompt", "You are a helpful assistant.")
    if nsfw_mode and persona.get("nsfw_system_prompt_addon"):
        base_prompt += "\n" + persona["nsfw_system_prompt_addon"]
    parts.append(base_prompt)

    # 2. Persona's current emotional state
    if persona_emotion_desc:
        parts.append(f"\n## Your Current Emotional State\n{persona_emotion_desc}")
        parts.append(
            "Let this emotional state subtly influence your tone, word choice, and warmth — "
            "but don't explicitly state your emotions unless it feels natural in conversation."
        )

    # 3. Known facts about the user
    if facts:
        fact_lines = []
        for fact in facts:
            # fact_store returns (id, text, tags, relevance_score) or (id, text, relevance_score)
            if len(fact) >= 2:
                fact_lines.append(f"- {fact[1]}")
        if fact_lines:
            parts.append(
                "\n## What You Know About This User\n"
                + "\n".join(fact_lines)
            )
            parts.append(
                "Use these facts naturally in conversation when relevant. "
                "Don't list them back — weave them into your responses."
            )

    # 4. Relevant memories from past conversations
    if similar_memories:
        memory_lines = []
        for mem in similar_memories:
            payload = mem.get("payload", {})
            text = payload.get("text", payload.get("content", ""))
            score = mem.get("score", 0)
            if text and score > 0.3:
                memory_lines.append(f"- {text}")
        if memory_lines:
            parts.append(
                "\n## Relevant Past Conversations\n"
                + "\n".join(memory_lines[:5])  # cap to avoid prompt bloat
            )

    # 5. Related topics the user has discussed
    if related_topics:
        topic_str = ", ".join(related_topics[:10])
        parts.append(
            f"\n## Topics This User Is Interested In\n{topic_str}"
        )

    return "\n\n".join(parts)


def build_message_list(
    system_prompt: str,
    chat_history: list,
    user_input: str,
    buffer_messages: list = None,
) -> List[Dict[str, str]]:
    """
    Build the final message list for the LLM.

    Args:
        system_prompt: assembled system prompt from build_system_prompt()
        chat_history: past messages from DB [(id, role, content, ...), ...]
        user_input: current user message
        buffer_messages: recent messages from in-memory buffer [{"role": ..., "content": ...}, ...]

    Returns:
        List of {"role": ..., "content": ...} dicts ready for llm_client.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Add DB chat history (older messages)
    if chat_history:
        for msg in chat_history:
            # chat_store returns (id, role, content, sentiment, topics)
            role = msg[1]
            content = msg[2]
            messages.append({"role": role, "content": content})

    # Add buffer messages (most recent, may overlap with DB — engine handles dedup)
    if buffer_messages:
        for msg in buffer_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current user input
    messages.append({"role": "user", "content": user_input})

    return messages
