# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Echo Traits Extractor
---------------------
Analyzes a user corpus to extract personality and style signals:
- Writing style (sentence length, formality, punctuation patterns)
- Vocabulary preferences (common words, unique phrases)
- Emotional patterns (dominant emotions from conversation history)
- Topic interests (weighted by frequency)
- Values and opinions (from extracted facts)

No LLM call needed — pure pattern analysis on the corpus.
"""

import re
from collections import Counter
from typing import List, Dict


# Words that suggest informal style
INFORMAL_MARKERS = {
    "lol", "haha", "yeah", "yep", "nah", "gonna", "wanna", "kinda",
    "sorta", "btw", "tbh", "imo", "ngl", "bruh", "dude", "bro",
    "ok", "okay", "cool", "awesome", "chill", "yo", "sup",
}

# Words that suggest formal style
FORMAL_MARKERS = {
    "therefore", "however", "furthermore", "nevertheless", "consequently",
    "regarding", "concerning", "additionally", "moreover", "accordingly",
    "indeed", "perhaps", "certainly", "essentially", "fundamentally",
}


def extract_traits(corpus: dict) -> dict:
    """
    Analyze a corpus and return a traits profile.

    Args:
        corpus: dict from corpus_builder.build_corpus()

    Returns:
        dict with keys: style, vocabulary, emotions, interests, values, summary
    """
    user_messages = [
        m["content"] for m in corpus.get("messages", [])
        if m["role"] == "user" and m["content"]
    ]

    if not user_messages:
        return _empty_traits()

    style = _analyze_style(user_messages)
    vocabulary = _analyze_vocabulary(user_messages)
    interests = _analyze_interests(corpus.get("topics", []))
    values = _analyze_values(corpus.get("facts", []))

    return {
        "style": style,
        "vocabulary": vocabulary,
        "interests": interests,
        "values": values,
        "message_count": len(user_messages),
    }


def _analyze_style(messages: List[str]) -> dict:
    """Analyze writing style patterns."""
    total_chars = sum(len(m) for m in messages)
    total_words = sum(len(m.split()) for m in messages)
    total_sentences = sum(len(re.split(r'[.!?]+', m)) for m in messages)

    avg_msg_length = total_chars / len(messages)
    avg_words_per_msg = total_words / len(messages)
    avg_sentence_length = total_words / max(total_sentences, 1)

    # Check formality
    all_text = " ".join(messages).lower()
    words = re.findall(r'\b\w+\b', all_text)
    word_set = set(words)

    informal_count = len(word_set & INFORMAL_MARKERS)
    formal_count = len(word_set & FORMAL_MARKERS)

    if informal_count > formal_count + 2:
        formality = "informal"
    elif formal_count > informal_count + 2:
        formality = "formal"
    else:
        formality = "neutral"

    # Punctuation habits
    question_ratio = sum(1 for m in messages if "?" in m) / len(messages)
    exclamation_ratio = sum(1 for m in messages if "!" in m) / len(messages)
    emoji_ratio = sum(1 for m in messages if re.search(r'[\U0001f600-\U0001f9ff]', m)) / len(messages)
    ellipsis_ratio = sum(1 for m in messages if "..." in m) / len(messages)

    # Capitalization pattern
    starts_lowercase = sum(1 for m in messages if m and m[0].islower()) / len(messages)

    return {
        "avg_message_length": round(avg_msg_length, 1),
        "avg_words_per_message": round(avg_words_per_msg, 1),
        "avg_sentence_length": round(avg_sentence_length, 1),
        "formality": formality,
        "uses_questions": round(question_ratio, 2),
        "uses_exclamations": round(exclamation_ratio, 2),
        "uses_emoji": round(emoji_ratio, 2),
        "uses_ellipsis": round(ellipsis_ratio, 2),
        "starts_lowercase": round(starts_lowercase, 2),
    }


def _analyze_vocabulary(messages: List[str]) -> dict:
    """Identify characteristic vocabulary patterns."""
    all_words = []
    for m in messages:
        words = re.findall(r'\b\w+\b', m.lower())
        all_words.extend(words)

    # Filter out very common words
    stopwords = {
        "i", "me", "my", "we", "you", "your", "it", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "can", "may", "might",
        "shall", "the", "a", "an", "and", "but", "or", "so", "if", "then",
        "that", "this", "what", "which", "who", "how", "when", "where",
        "not", "no", "yes", "to", "of", "in", "on", "at", "for", "with",
        "from", "by", "as", "up", "out", "about", "into", "just", "also",
        "than", "its", "all", "there", "their", "them", "they", "he", "she",
        "his", "her", "him", "some", "any", "more", "very", "too", "much",
        "many", "most", "other", "only", "own", "same", "get", "got",
        "like", "know", "think", "want", "going", "go", "make", "see",
        "one", "two", "new", "now", "way", "time", "thing", "things",
        "really", "well", "still", "even", "back", "here", "over",
        "don", "doesn", "didn", "won", "isn", "aren", "wasn", "weren",
        "haven", "hasn", "hadn", "couldn", "wouldn", "shouldn", "ll", "ve",
        "re", "im", "been",
    }

    meaningful = [w for w in all_words if w not in stopwords and len(w) > 2]
    word_freq = Counter(meaningful)

    # Top characteristic words
    top_words = [w for w, _ in word_freq.most_common(20)]

    # Phrases (bigrams)
    bigrams = []
    for m in messages:
        words = re.findall(r'\b\w+\b', m.lower())
        for i in range(len(words) - 1):
            if words[i] not in stopwords or words[i+1] not in stopwords:
                bigrams.append(f"{words[i]} {words[i+1]}")

    bigram_freq = Counter(bigrams)
    top_phrases = [p for p, c in bigram_freq.most_common(10) if c >= 2]

    return {
        "characteristic_words": top_words,
        "common_phrases": top_phrases,
        "vocabulary_size": len(set(meaningful)),
    }


def _analyze_interests(topics: list) -> list:
    """Extract top interests from topic data."""
    sorted_topics = sorted(topics, key=lambda t: t.get("weight", 0), reverse=True)
    return [
        {"topic": t["topic"], "weight": t["weight"]}
        for t in sorted_topics[:10]
    ]


def _analyze_values(facts: list) -> list:
    """Extract values and preferences from identity/knowledge facts."""
    values = []
    for fact in facts:
        text = fact.get("text", "")
        tier = fact.get("tier", "")
        if tier == "identity" or any(kw in text.lower() for kw in
                                      ["love", "hate", "prefer", "value",
                                       "believe", "important", "favorite"]):
            values.append(text)
    return values[:15]


def _empty_traits() -> dict:
    """Return empty traits structure when no data is available."""
    return {
        "style": {
            "avg_message_length": 0,
            "avg_words_per_message": 0,
            "avg_sentence_length": 0,
            "formality": "unknown",
            "uses_questions": 0,
            "uses_exclamations": 0,
            "uses_emoji": 0,
            "uses_ellipsis": 0,
            "starts_lowercase": 0,
        },
        "vocabulary": {
            "characteristic_words": [],
            "common_phrases": [],
            "vocabulary_size": 0,
        },
        "interests": [],
        "values": [],
        "message_count": 0,
    }
