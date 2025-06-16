# AI-persona-framework - Emotion Handler Unit Test
# Copyright (C) 2025 Kenneth Haider

import unittest, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from shared.analysis.emotion_handler import EmotionVectorGenerator

class TestEmotionVectorGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = EmotionVectorGenerator()

    def test_analyze_emotions(self):
        text = "I'm very happy and curious today!"
        result = self.generator.analyze(text)
        self.assertIsInstance(result, dict)
        self.assertGreater(result.get("joy", 0), 0)
        self.assertGreater(result.get("curiosity", 0), 0)

    def test_smooth_vector(self):
        old = {k: 0.0 for k in self.generator.emotions}
        new = {k: 1.0 if k == "joy" else 0.0 for k in self.generator.emotions}
        smoothed = self.generator.smooth_vector(new, old, alpha=0.5)
        self.assertAlmostEqual(smoothed["joy"], 0.5, places=2)

if __name__ == "__main__":
    unittest.main()
