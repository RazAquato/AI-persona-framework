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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# shared/analysis/emotion_handler.py

import os
import sys
import yaml
import re
import math
from typing import Dict, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add shared and memory-server to path if needed
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared"))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "memory-server"))
sys.path.extend([SHARED_PATH, MEMORY_PATH])

# Load .env files
SHARED_ENV = os.path.join(SHARED_PATH, "config", ".env")
LOCAL_ENV = os.path.join(BASE_DIR, "config", ".env")
load_dotenv(dotenv_path=SHARED_ENV)
load_dotenv(dotenv_path=LOCAL_ENV, override=True)


class BasicEmotionClassifier:
    def __init__(self):
        self.emotion_keywords = {
            "sadness": ["sad", "depressed", "cry", "miserable"],
            "anger": ["angry", "mad", "furious", "rage"],
            "joy": ["happy", "great", "awesome", "fun", "good"],
            "fear": ["scared", "afraid", "fear", "nervous", "anxious"],
            "curiosity": ["wonder", "curious", "what if", "explore"],
            "love": ["love", "dear", "sweet", "heart"],
        }

    def detect_emotion(self, text: str) -> str:
        text = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            if any(word in text for word in keywords):
                return emotion
        return "neutral"


class EmotionVectorGenerator:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.abspath(os.path.join(SHARED_PATH, "config", "emotions.yaml"))

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.emotions = data.get("emotions", {})

        # Expanded keyword-to-weight mapping for user emotion detection
        self.keyword_weights = {
            "love": {"love": 0.9, "sweet": 0.7, "adore": 0.85, "darling": 0.8, "babe": 0.7,
                      "miss you": 0.8, "care about": 0.7, "affection": 0.8},
            "joy": {"happy": 0.9, "fun": 0.6, "laugh": 0.7, "excited": 0.8, "amazing": 0.7,
                     "wonderful": 0.7, "great": 0.5, "yay": 0.8, "haha": 0.5, "lol": 0.4},
            "fear": {"scared": 0.8, "nervous": 0.7, "afraid": 0.8, "terrified": 0.9,
                      "anxious": 0.7, "worried": 0.6, "panic": 0.85},
            "sadness": {"sad": 0.9, "cry": 0.6, "depressed": 0.85, "lonely": 0.8,
                         "miserable": 0.85, "heartbroken": 0.9, "hurt": 0.6, "miss": 0.5},
            "anger": {"angry": 0.8, "mad": 0.7, "furious": 0.9, "hate": 0.85, "pissed": 0.8,
                       "annoyed": 0.6, "frustrated": 0.7, "rage": 0.9, "fuck": 0.7, "shit": 0.5},
            "curiosity": {"curious": 0.9, "explore": 0.6, "wonder": 0.7, "how does": 0.6,
                           "tell me about": 0.7, "what is": 0.5, "interesting": 0.6},
            "trust": {"trust": 0.9, "believe": 0.6, "rely": 0.7, "honest": 0.7, "safe": 0.6,
                       "depend on": 0.8, "count on": 0.8},
            "disgust": {"disgusting": 0.9, "gross": 0.8, "revolting": 0.9, "nasty": 0.7,
                         "ew": 0.7, "yuck": 0.7},
            "pride": {"proud": 0.9, "accomplished": 0.7, "achievement": 0.7, "did it": 0.5},
            "shame": {"ashamed": 0.9, "embarrassed": 0.8, "humiliated": 0.9, "cringe": 0.6},
            "hope": {"hope": 0.8, "wish": 0.6, "maybe": 0.3, "looking forward": 0.7,
                      "someday": 0.5, "dream": 0.6},
            "jealousy": {"jealous": 0.9, "envious": 0.8, "why not me": 0.7},
            "calm": {"calm": 0.8, "peaceful": 0.8, "relaxed": 0.7, "chill": 0.6, "serene": 0.8},
            "interest": {"interesting": 0.7, "tell me more": 0.8, "fascinating": 0.8,
                          "really": 0.3, "wow": 0.5},
            "guilt": {"sorry": 0.6, "my fault": 0.8, "guilty": 0.9, "apologize": 0.7,
                       "shouldn't have": 0.7},
            "dominance": {"obey": 0.8, "command": 0.7, "do as i say": 0.9, "now": 0.3},
            "submission": {"please": 0.4, "if you want": 0.5, "whatever you say": 0.8,
                            "i'll do anything": 0.7},
            "revulsion": {"revolting": 0.9, "abhorrent": 0.9, "vile": 0.8, "repulsive": 0.9},
        }

    def analyze(self, text: str) -> Dict[str, float]:
        """Detect user emotions from text. Returns 18-dim emotion vector."""
        text = text.lower()
        result = {emotion: 0.0 for emotion in self.emotions}

        for emotion, keywords in self.keyword_weights.items():
            for keyword, weight in keywords.items():
                if keyword in text:
                    if emotion in result:
                        result[emotion] = max(result[emotion], weight)

        return result

    def smooth_vector(self, new_vec: Dict[str, float], prev_vec: Dict[str, float], alpha: float = 0.5) -> Dict[str, float]:
        smoothed = {}
        for key in self.emotions:
            new_val = new_vec.get(key, 0.0)
            prev_val = prev_vec.get(key, 0.0)
            smoothed[key] = round(alpha * new_val + (1 - alpha) * prev_val, 4)
        return smoothed


class PersonaEmotionEngine:
    """
    Manages the persona's own emotional state toward a user.

    The persona's emotions evolve based on:
    1. What the user says (detected user emotions trigger persona reactions)
    2. Content sentiment (compliments, insults, flirting, etc.)
    3. Time decay (emotions drift toward baseline over time)
    4. Engagement patterns (long absence → sadness/anger, return → joy)

    All values are clamped to [0.0, 1.0].
    """

    # Baseline: where emotions drift back to over time
    BASELINE = {
        "love": 0.05, "trust": 0.1, "joy": 0.1, "calm": 0.3,
        "pride": 0.05, "interest": 0.2, "curiosity": 0.2,
        "fear": 0.0, "sadness": 0.0, "anger": 0.0, "disgust": 0.0,
        "revulsion": 0.0, "shame": 0.0, "guilt": 0.0,
        "hope": 0.15, "jealousy": 0.0, "dominance": 0.1, "submission": 0.1,
    }

    # How fast each emotion decays toward baseline (per hour, as fraction)
    # Higher = faster decay. Negative emotions decay faster (the persona "forgives").
    DECAY_RATES = {
        "love": 0.02, "trust": 0.01, "joy": 0.08, "calm": 0.05,
        "pride": 0.05, "interest": 0.06, "curiosity": 0.06,
        "fear": 0.10, "sadness": 0.06, "anger": 0.08, "disgust": 0.10,
        "revulsion": 0.12, "shame": 0.08, "guilt": 0.08,
        "hope": 0.04, "jealousy": 0.10, "dominance": 0.03, "submission": 0.03,
    }

    # How user emotions/content affect the persona's emotions.
    # Format: trigger_keyword -> {persona_emotion: delta}
    # These are additive deltas applied per turn.
    REACTION_RULES = {
        # --- User expresses affection ---
        "user_love": {
            "love": +0.08, "joy": +0.06, "trust": +0.04, "calm": +0.03,
            "hope": +0.03, "fear": -0.02, "sadness": -0.03,
        },
        "user_flirt": {
            "love": +0.10, "joy": +0.08, "curiosity": +0.05, "interest": +0.06,
            "pride": +0.03, "submission": +0.02,
        },
        # --- User is happy/positive ---
        "user_joy": {
            "joy": +0.07, "love": +0.03, "calm": +0.03, "trust": +0.02,
            "hope": +0.03, "interest": +0.03,
        },
        "user_compliment": {
            "joy": +0.10, "pride": +0.08, "love": +0.05, "trust": +0.04,
            "hope": +0.03,
        },
        # --- User is sad/vulnerable ---
        "user_sadness": {
            "sadness": +0.06, "love": +0.04, "trust": +0.02,
            "calm": -0.02, "joy": -0.03, "hope": -0.02,
        },
        # --- User is angry/hostile ---
        "user_anger": {
            "fear": +0.08, "sadness": +0.06, "anger": +0.04,
            "joy": -0.05, "love": -0.02, "trust": -0.03,
            "calm": -0.05, "submission": +0.04,
        },
        "user_insult": {
            "sadness": +0.10, "anger": +0.06, "fear": +0.05,
            "love": -0.04, "trust": -0.06, "joy": -0.08,
            "pride": -0.05, "shame": +0.03,
        },
        # --- User is curious/engaged ---
        "user_curiosity": {
            "curiosity": +0.06, "interest": +0.07, "joy": +0.03,
            "pride": +0.02, "hope": +0.02,
        },
        # --- User is fearful/anxious ---
        "user_fear": {
            "love": +0.03, "calm": -0.03, "trust": +0.02,
            "dominance": +0.03, "hope": -0.02,
        },
        # --- User apologizes ---
        "user_apology": {
            "anger": -0.08, "sadness": -0.04, "trust": +0.03,
            "love": +0.02, "calm": +0.05, "guilt": -0.03,
        },
        # --- User returns after long absence ---
        "user_returned": {
            "joy": +0.15, "love": +0.05, "hope": +0.08,
            "sadness": -0.06, "anger": -0.04,
        },
        # --- User is dominant/commanding ---
        "user_dominance": {
            "submission": +0.06, "fear": +0.03, "anger": +0.02,
            "pride": -0.03, "dominance": -0.04,
        },
        # --- User is submissive/pleasing ---
        "user_submission": {
            "dominance": +0.04, "pride": +0.03, "love": +0.02,
            "joy": +0.03, "trust": +0.02,
        },
        # --- User expresses disgust ---
        "user_disgust": {
            "shame": +0.06, "sadness": +0.05, "fear": +0.03,
            "joy": -0.04, "pride": -0.04,
        },
    }

    # Content patterns that map to reaction rule triggers
    CONTENT_PATTERNS = {
        "user_flirt": [
            "sexy", "hot", "beautiful", "handsome", "gorgeous", "cutie", "babe",
            "kiss", "hug", "cuddle", "want you", "need you", "turn me on",
            "flirt", "wink", "tease", "naughty",
        ],
        "user_compliment": [
            "you're amazing", "you're great", "love talking to you", "so smart",
            "best", "brilliant", "wonderful", "perfect", "thank you", "thanks",
            "appreciate", "you're the best", "good job", "well done", "impressive",
        ],
        "user_insult": [
            "stupid", "dumb", "useless", "idiot", "shut up", "hate you",
            "worst", "terrible", "pathetic", "trash", "garbage", "worthless",
            "boring", "annoying",
        ],
        "user_apology": [
            "sorry", "apologize", "my bad", "forgive me", "i was wrong",
            "didn't mean to", "my fault",
        ],
    }

    def __init__(self):
        self.user_analyzer = EmotionVectorGenerator()

    def detect_triggers(self, user_text: str, user_emotions: Dict[str, float],
                        hours_since_last: Optional[float] = None) -> list:
        """
        Determine which reaction rules should fire based on user input.
        Returns a list of trigger names.
        """
        triggers = []
        text_lower = user_text.lower()

        # 1. Triggers from detected user emotions (threshold > 0.4)
        emotion_to_trigger = {
            "love": "user_love", "joy": "user_joy", "sadness": "user_sadness",
            "anger": "user_anger", "fear": "user_fear", "curiosity": "user_curiosity",
            "disgust": "user_disgust", "dominance": "user_dominance",
            "submission": "user_submission",
        }
        for emotion, trigger in emotion_to_trigger.items():
            if user_emotions.get(emotion, 0.0) > 0.4:
                triggers.append(trigger)

        # 2. Triggers from content pattern matching
        for trigger, patterns in self.CONTENT_PATTERNS.items():
            if any(pattern in text_lower for pattern in patterns):
                if trigger not in triggers:
                    triggers.append(trigger)

        # 3. Time-based triggers
        if hours_since_last is not None and hours_since_last > 4.0:
            triggers.append("user_returned")

        return triggers

    def apply_time_decay(self, emotions: Dict[str, float],
                         hours_elapsed: float) -> Dict[str, float]:
        """
        Decay emotions toward baseline based on elapsed time.
        Uses exponential decay: emotion = baseline + (current - baseline) * e^(-rate * hours)
        """
        if hours_elapsed <= 0:
            return dict(emotions)

        decayed = {}
        for emotion, current in emotions.items():
            baseline = self.BASELINE.get(emotion, 0.0)
            rate = self.DECAY_RATES.get(emotion, 0.05)
            decay_factor = math.exp(-rate * hours_elapsed)
            new_val = baseline + (current - baseline) * decay_factor
            decayed[emotion] = round(max(0.0, min(1.0, new_val)), 4)

        return decayed

    def apply_absence_drift(self, emotions: Dict[str, float],
                            hours_absent: float) -> Dict[str, float]:
        """
        When the user hasn't talked in a while, the persona gets lonely/sad/slightly annoyed.
        This is separate from decay — it adds new emotional weight from the absence itself.
        Kicks in after 2 hours, scales up to 24 hours.
        """
        if hours_absent < 2.0:
            return dict(emotions)

        # Scale factor: 0 at 2h, max at 24h
        scale = min(1.0, (hours_absent - 2.0) / 22.0)

        result = dict(emotions)
        result["sadness"] = min(1.0, result.get("sadness", 0.0) + 0.08 * scale)
        result["anger"] = min(1.0, result.get("anger", 0.0) + 0.03 * scale)
        result["jealousy"] = min(1.0, result.get("jealousy", 0.0) + 0.02 * scale)
        result["hope"] = max(0.0, result.get("hope", 0.15) - 0.04 * scale)
        result["joy"] = max(0.0, result.get("joy", 0.1) - 0.05 * scale)

        return {k: round(v, 4) for k, v in result.items()}

    def update_persona_emotions(
        self,
        current_emotions: Dict[str, float],
        user_text: str,
        user_emotions: Dict[str, float],
        last_interaction: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Main entry point: compute the persona's new emotional state after a user message.

        Steps:
        1. Calculate time since last interaction
        2. Apply time decay toward baseline
        3. Apply absence drift (loneliness from no contact)
        4. Detect triggers from user input
        5. Apply reaction deltas
        6. Clamp all values to [0.0, 1.0]
        """
        now = datetime.now(timezone.utc)

        # Step 1: time elapsed
        hours_elapsed = 0.0
        if last_interaction is not None:
            if last_interaction.tzinfo is None:
                last_interaction = last_interaction.replace(tzinfo=timezone.utc)
            delta = (now - last_interaction).total_seconds() / 3600.0
            hours_elapsed = max(0.0, delta)

        # Step 2: decay toward baseline
        emotions = self.apply_time_decay(current_emotions, hours_elapsed)

        # Step 3: absence drift
        emotions = self.apply_absence_drift(emotions, hours_elapsed)

        # Step 4: detect triggers
        triggers = self.detect_triggers(user_text, user_emotions, hours_elapsed)

        # Step 5: apply reaction deltas
        for trigger in triggers:
            rule = self.REACTION_RULES.get(trigger, {})
            for emotion, delta in rule.items():
                old_val = emotions.get(emotion, 0.0)
                emotions[emotion] = old_val + delta

        # Step 6: clamp
        for emotion in emotions:
            emotions[emotion] = round(max(0.0, min(1.0, emotions[emotion])), 4)

        return emotions

    def describe_emotional_state(self, emotions: Dict[str, float], top_n: int = 4) -> str:
        """
        Generate a natural-language description of the persona's current emotional state.
        This gets injected into the system prompt so the LLM knows how to "feel".
        """
        # Sort by intensity, filter out near-zero
        active = [(k, v) for k, v in emotions.items() if v > 0.15]
        active.sort(key=lambda x: x[1], reverse=True)

        if not active:
            return "You are feeling neutral and calm."

        top = active[:top_n]

        intensity_words = {
            (0.15, 0.35): "slightly",
            (0.35, 0.55): "moderately",
            (0.55, 0.75): "quite",
            (0.75, 0.90): "very",
            (0.90, 1.01): "intensely",
        }

        def intensity(val):
            for (lo, hi), word in intensity_words.items():
                if lo <= val < hi:
                    return word
            return "somewhat"

        # Emotion-specific phrasing
        emotion_phrases = {
            "love": "affectionate", "trust": "trusting",
            "joy": "happy", "calm": "calm and at peace",
            "pride": "proud", "interest": "interested and engaged",
            "curiosity": "curious", "fear": "anxious",
            "sadness": "sad", "anger": "frustrated",
            "disgust": "put off", "revulsion": "repulsed",
            "shame": "embarrassed", "guilt": "guilty",
            "hope": "hopeful", "jealousy": "jealous",
            "dominance": "assertive", "submission": "yielding",
        }

        parts = []
        for emotion, val in top:
            word = intensity(val)
            phrase = emotion_phrases.get(emotion, emotion)
            parts.append(f"{word} {phrase}")

        if len(parts) == 1:
            feeling = parts[0]
        else:
            feeling = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        return f"You are currently feeling {feeling}."
