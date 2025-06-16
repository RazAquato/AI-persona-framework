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
from typing import Dict
from dotenv import load_dotenv

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add shared and memory-server to path if needed
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared"))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "memory-server"))
sys.path.extend([SHARED_PATH, MEMORY_PATH])

# Load .env files
SHARED_ENV = os.path.join(SHARED_PATH, "config", ".env")
LOCAL_ENV = os.path.join(BASE_DIR, "config", ".env")  # Optional: for future per-module configs
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
        print(SHARED_PATH)
        if config_path is None:
            # shared/config/emotions.yaml
            config_path = os.path.abspath(os.path.join(SHARED_PATH, "config", "emotions.yaml"))

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.emotions = data.get("emotions", {})

        # Simple emotion-word mapping (can be expanded)
        self.keyword_weights = {
            "love": {"love": 0.9, "sweet": 0.7},
            "joy": {"happy": 0.9, "fun": 0.6},
            "fear": {"scared": 0.8, "nervous": 0.7},
            "sadness": {"sad": 0.9, "cry": 0.6},
            "anger": {"angry": 0.8, "mad": 0.7},
            "curiosity": {"curious": 0.9, "explore": 0.6},
            # Extendable from config later
        }

    def analyze(self, text: str) -> Dict[str, float]:
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
