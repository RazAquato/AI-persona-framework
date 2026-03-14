# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Echo Prompt Builder
-------------------
Takes extracted traits and builds a system prompt section that instructs
the LLM to respond in the user's voice/style.

This is the "prompt injection" approach to Echo — no fine-tuning needed.
The traits profile is converted into natural-language instructions.
"""


def build_echo_prompt(traits: dict, user_name: str = "the user") -> str:
    """
    Build a system prompt section that simulates the user's personality.

    Args:
        traits: dict from traits_extractor.extract_traits()
        user_name: display name for the user being echoed

    Returns:
        System prompt string for echo mode.
    """
    if traits.get("message_count", 0) == 0:
        return (
            f"You are simulating {user_name}'s personality, but there is not enough "
            "conversation history yet. Respond naturally and explain that you need more "
            "conversations to build an accurate echo."
        )

    parts = []
    parts.append(
        f"You are simulating the personality of {user_name}. "
        f"Respond as {user_name} would — match their tone, style, and perspective. "
        "Do NOT break character or mention that you are an AI."
    )

    # Style instructions
    style = traits.get("style", {})
    style_desc = _describe_style(style)
    if style_desc:
        parts.append(f"\n## Writing Style\n{style_desc}")

    # Vocabulary hints
    vocab = traits.get("vocabulary", {})
    vocab_desc = _describe_vocabulary(vocab)
    if vocab_desc:
        parts.append(f"\n## Vocabulary\n{vocab_desc}")

    # Interests
    interests = traits.get("interests", [])
    if interests:
        topic_list = ", ".join(i["topic"].replace("_", " ") for i in interests[:8])
        parts.append(
            f"\n## Interests\n{user_name} is interested in: {topic_list}. "
            "Reference these naturally when relevant."
        )

    # Values and identity
    values = traits.get("values", [])
    if values:
        value_lines = "\n".join(f"- {v}" for v in values[:10])
        parts.append(
            f"\n## Identity & Values\n{value_lines}"
        )

    return "\n\n".join(parts)


def _describe_style(style: dict) -> str:
    """Convert style metrics into natural-language instructions."""
    lines = []

    formality = style.get("formality", "neutral")
    if formality == "informal":
        lines.append("Write casually and conversationally. Use slang and contractions freely.")
    elif formality == "formal":
        lines.append("Write in a measured, articulate way. Use proper grammar and complete sentences.")

    avg_words = style.get("avg_words_per_message", 0)
    if avg_words > 0:
        if avg_words < 10:
            lines.append("Keep responses short and punchy — typically just a few words or a sentence.")
        elif avg_words < 30:
            lines.append("Write moderately-length responses — a sentence or two.")
        else:
            lines.append("Write detailed, thorough responses — this person tends to elaborate.")

    if style.get("uses_emoji", 0) > 0.2:
        lines.append("Use emoji occasionally as this person does.")
    elif style.get("uses_emoji", 0) == 0:
        lines.append("Do not use emoji — this person doesn't.")

    if style.get("uses_exclamations", 0) > 0.3:
        lines.append("Use exclamation marks to show enthusiasm — this person is expressive!")

    if style.get("uses_ellipsis", 0) > 0.15:
        lines.append("Use ellipsis (...) for trailing thoughts — this person does it often.")

    if style.get("starts_lowercase", 0) > 0.5:
        lines.append("Often start messages with lowercase letters.")

    if style.get("uses_questions", 0) > 0.4:
        lines.append("Ask questions frequently — this person is curious and engaging.")

    return " ".join(lines)


def _describe_vocabulary(vocab: dict) -> str:
    """Convert vocabulary analysis into instructions."""
    words = vocab.get("characteristic_words", [])
    phrases = vocab.get("common_phrases", [])

    lines = []
    if words:
        word_str = ", ".join(f'"{w}"' for w in words[:10])
        lines.append(f"Characteristic words this person uses often: {word_str}.")

    if phrases:
        phrase_str = ", ".join(f'"{p}"' for p in phrases[:5])
        lines.append(f"Common phrases: {phrase_str}.")

    return " ".join(lines)
