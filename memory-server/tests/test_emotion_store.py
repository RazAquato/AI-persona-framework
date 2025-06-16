# AI-persona-framework - Emotion DB Store Unit Test
# Copyright (C) 2025 Kenneth Haider

import unittest
from memory.emotion_store import store_emotion_vector, load_latest_emotion_state

class TestEmotionStore(unittest.TestCase):

    def test_store_and_load(self):
        fake_vector = {
            "joy": 0.9,
            "sadness": 0.1,
            "anger": 0.0
        }

        # Replace with a known message ID from a seeded test DB
        test_message_id = 1 
        test_user_id = 9999

        # Store emotion
        store_emotion_vector(test_message_id, fake_vector, tone="joy")

        # Load emotion (assumes message 1 belongs to user 9999)
        result = load_latest_emotion_state(user_id=test_user_id, role="user")
        print(result)
        self.assertIsInstance(result, dict)
        emotions = result.get("emotions", {})
        self.assertGreater(emotions.get("joy", 0), 0.5)

if __name__ == "__main__":
    unittest.main()
