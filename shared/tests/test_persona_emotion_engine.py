# AI-persona-framework - PersonaEmotionEngine Unit Tests
# Copyright (C) 2025 Kenneth Haider

import unittest
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from shared.analysis.emotion_handler import PersonaEmotionEngine, EmotionVectorGenerator


class TestPersonaEmotionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = PersonaEmotionEngine()
        self.baseline = dict(PersonaEmotionEngine.BASELINE)

    def test_baseline_is_complete(self):
        """Baseline should have all 18 emotions."""
        self.assertEqual(len(self.baseline), 18)
        for val in self.baseline.values():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_detect_triggers_love(self):
        """User expressing love should fire user_love trigger."""
        user_emo = {"love": 0.9, "joy": 0.0, "anger": 0.0, "sadness": 0.0,
                     "fear": 0.0, "curiosity": 0.0, "disgust": 0.0,
                     "dominance": 0.0, "submission": 0.0}
        triggers = self.engine.detect_triggers("I love you", user_emo)
        self.assertIn("user_love", triggers)

    def test_detect_triggers_flirt_pattern(self):
        """Content patterns should detect flirting."""
        user_emo = {k: 0.0 for k in self.baseline}
        triggers = self.engine.detect_triggers("you're so sexy", user_emo)
        self.assertIn("user_flirt", triggers)

    def test_detect_triggers_insult_pattern(self):
        """Content patterns should detect insults."""
        user_emo = {k: 0.0 for k in self.baseline}
        triggers = self.engine.detect_triggers("you're so stupid", user_emo)
        self.assertIn("user_insult", triggers)

    def test_detect_triggers_user_returned(self):
        """Long absence should trigger user_returned."""
        user_emo = {k: 0.0 for k in self.baseline}
        triggers = self.engine.detect_triggers("hey", user_emo, hours_since_last=8.0)
        self.assertIn("user_returned", triggers)

    def test_detect_triggers_no_return_for_short_absence(self):
        """Short absence should NOT trigger user_returned."""
        user_emo = {k: 0.0 for k in self.baseline}
        triggers = self.engine.detect_triggers("hey", user_emo, hours_since_last=1.0)
        self.assertNotIn("user_returned", triggers)

    def test_time_decay_toward_baseline(self):
        """Emotions should decay toward baseline over time."""
        elevated = dict(self.baseline)
        elevated["joy"] = 0.9
        elevated["anger"] = 0.8

        decayed = self.engine.apply_time_decay(elevated, hours_elapsed=10.0)

        # Joy and anger should have moved toward their baselines
        self.assertLess(decayed["joy"], 0.9)
        self.assertGreater(decayed["joy"], self.baseline["joy"])
        self.assertLess(decayed["anger"], 0.8)

    def test_no_decay_at_zero_hours(self):
        """Zero time elapsed should not change emotions."""
        elevated = dict(self.baseline)
        elevated["joy"] = 0.9
        decayed = self.engine.apply_time_decay(elevated, hours_elapsed=0.0)
        self.assertAlmostEqual(decayed["joy"], 0.9, places=3)

    def test_absence_drift_after_long_gap(self):
        """Long absence should increase sadness."""
        emotions = dict(self.baseline)
        drifted = self.engine.apply_absence_drift(emotions, hours_absent=12.0)
        self.assertGreater(drifted["sadness"], emotions["sadness"])

    def test_absence_drift_no_effect_short_gap(self):
        """Short absence should not trigger drift."""
        emotions = dict(self.baseline)
        drifted = self.engine.apply_absence_drift(emotions, hours_absent=1.0)
        self.assertAlmostEqual(drifted["sadness"], emotions["sadness"], places=4)

    def test_update_love_increases_on_affection(self):
        """Persona love should increase when user expresses love."""
        gen = EmotionVectorGenerator()
        user_emo = gen.analyze("I love you so much darling")
        new = self.engine.update_persona_emotions(
            dict(self.baseline), "I love you so much darling", user_emo
        )
        self.assertGreater(new["love"], self.baseline["love"])
        self.assertGreater(new["joy"], self.baseline["joy"])

    def test_update_sadness_increases_on_insult(self):
        """Persona sadness should increase when user insults."""
        user_emo = {k: 0.0 for k in self.baseline}
        user_emo["anger"] = 0.8
        new = self.engine.update_persona_emotions(
            dict(self.baseline), "you're stupid and useless", user_emo
        )
        self.assertGreater(new["sadness"], self.baseline["sadness"])

    def test_clamping(self):
        """All emotions should stay in [0.0, 1.0] even with extreme input."""
        maxed = {k: 1.0 for k in self.baseline}
        user_emo = {k: 1.0 for k in self.baseline}
        new = self.engine.update_persona_emotions(
            maxed, "I love you you're amazing kiss hug", user_emo
        )
        for val in new.values():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_clamping_floor(self):
        """Emotions should not go below 0.0."""
        zeroed = {k: 0.0 for k in self.baseline}
        user_emo = {k: 0.0 for k in self.baseline}
        user_emo["anger"] = 1.0
        new = self.engine.update_persona_emotions(
            zeroed, "you're stupid trash garbage", user_emo
        )
        for val in new.values():
            self.assertGreaterEqual(val, 0.0)


class TestDescribeEmotionalState(unittest.TestCase):

    def setUp(self):
        self.engine = PersonaEmotionEngine()

    def test_neutral_state(self):
        """Near-zero emotions should produce neutral description."""
        emotions = {k: 0.0 for k in PersonaEmotionEngine.BASELINE}
        desc = self.engine.describe_emotional_state(emotions)
        self.assertIn("neutral", desc.lower())

    def test_high_love_description(self):
        """High love should mention affectionate."""
        emotions = {k: 0.0 for k in PersonaEmotionEngine.BASELINE}
        emotions["love"] = 0.8
        desc = self.engine.describe_emotional_state(emotions)
        self.assertIn("affectionate", desc.lower())

    def test_moderate_anger_description(self):
        """Moderate anger should mention frustrated."""
        emotions = {k: 0.0 for k in PersonaEmotionEngine.BASELINE}
        emotions["anger"] = 0.5
        desc = self.engine.describe_emotional_state(emotions)
        self.assertIn("frustrated", desc.lower())

    def test_intensity_words(self):
        """Different intensities should use different words."""
        emotions = {k: 0.0 for k in PersonaEmotionEngine.BASELINE}
        emotions["joy"] = 0.25
        desc_low = self.engine.describe_emotional_state(emotions)
        self.assertIn("slightly", desc_low.lower())

        emotions["joy"] = 0.85
        desc_high = self.engine.describe_emotional_state(emotions)
        self.assertIn("very", desc_high.lower())


if __name__ == "__main__":
    unittest.main()
