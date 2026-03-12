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

import unittest
from analysis.knowledge_extractor import KnowledgeExtractor


class TestKnowledgeExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = KnowledgeExtractor()

    # --- extract_all structure ---

    def test_extract_all_returns_expected_keys(self):
        result = self.extractor.extract_all("Hello there", role="user")
        self.assertIn("facts", result)
        self.assertIn("entities", result)
        self.assertIn("topics", result)
        self.assertIn("classification", result)
        self.assertIn("raw_text", result)

    def test_extract_all_preserves_raw_text(self):
        result = self.extractor.extract_all("My name is Kenneth")
        self.assertEqual(result["raw_text"], "My name is Kenneth")

    def test_extract_all_assistant_skips_facts(self):
        result = self.extractor.extract_all("My name is Kenneth", role="assistant")
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["entities"], [])

    # --- Identity extraction ---

    def test_extract_name(self):
        result = self.extractor.extract_all("My name is Kenneth")
        facts = result["facts"]
        self.assertTrue(any("Kenneth" in f["text"] for f in facts))
        name_fact = next(f for f in facts if "Kenneth" in f["text"])
        self.assertEqual(name_fact["tier"], "identity")
        self.assertEqual(name_fact["entity_type"], "person")

    def test_extract_age(self):
        result = self.extractor.extract_all("I am 35 years old")
        facts = result["facts"]
        self.assertTrue(any("35" in f["text"] for f in facts))

    def test_extract_location(self):
        result = self.extractor.extract_all("I live in Norway")
        facts = result["facts"]
        self.assertTrue(any("Norway" in f["text"] for f in facts))
        loc_fact = next(f for f in facts if "Norway" in f["text"])
        self.assertEqual(loc_fact["entity_type"], "place")

    def test_extract_occupation(self):
        result = self.extractor.extract_all("I work as a software engineer")
        facts = result["facts"]
        self.assertTrue(any("software" in f["text"].lower() for f in facts))

    def test_extract_family_member(self):
        result = self.extractor.extract_all("My wife is named Maria")
        facts = result["facts"]
        self.assertTrue(any("Maria" in f["text"] for f in facts))

    def test_extract_children(self):
        result = self.extractor.extract_all("I have 2 kids")
        facts = result["facts"]
        self.assertTrue(any("2" in f["text"] and "kids" in f["text"] for f in facts))

    # --- Entity extraction ---

    def test_extract_pet_entity(self):
        result = self.extractor.extract_all("My dog is named Arix")
        entities = result["entities"]
        self.assertTrue(any("Arix" in e["text"] for e in entities))
        pet = next(e for e in entities if "Arix" in e["text"])
        self.assertEqual(pet["entity_type"], "pet")

    def test_extract_place_entity(self):
        result = self.extractor.extract_all("I went to Paris last summer")
        entities = result["entities"]
        self.assertTrue(any("Paris" in e["text"] for e in entities))

    # --- Preference extraction ---

    def test_extract_preference_love(self):
        result = self.extractor.extract_all("I love woodworking")
        facts = result["facts"]
        self.assertTrue(any("woodworking" in f["text"].lower() for f in facts))

    def test_extract_preference_hate(self):
        result = self.extractor.extract_all("I hate cold weather")
        facts = result["facts"]
        self.assertTrue(any("cold weather" in f["text"].lower() for f in facts))

    def test_extract_favorite(self):
        result = self.extractor.extract_all("My favorite color is blue")
        facts = result["facts"]
        self.assertTrue(any("blue" in f["text"].lower() for f in facts))

    # --- Topic detection ---

    def test_detect_technology_topic(self):
        result = self.extractor.extract_all("I've been learning python and machine learning")
        topics = result["topics"]
        topic_names = [t["topic"] for t in topics]
        self.assertIn("technology", topic_names)

    def test_detect_gaming_topic(self):
        result = self.extractor.extract_all("I played some Minecraft on my Xbox last night")
        topics = result["topics"]
        topic_names = [t["topic"] for t in topics]
        self.assertIn("gaming", topic_names)

    def test_detect_multiple_topics(self):
        result = self.extractor.extract_all("After my workout at the gym, I cooked dinner")
        topics = result["topics"]
        topic_names = [t["topic"] for t in topics]
        self.assertIn("fitness", topic_names)
        self.assertIn("cooking", topic_names)

    def test_topic_confidence_increases_with_matches(self):
        result = self.extractor.extract_all("I use python for machine learning and AI on my linux server")
        topics = result["topics"]
        tech = next(t for t in topics if t["topic"] == "technology")
        self.assertGreater(tech["confidence"], 0.6)

    def test_topics_capped_at_five(self):
        result = self.extractor.extract_all(
            "After hiking in the forest I cooked dinner while listening to music "
            "and coding a python server for my home assistant smart home project "
            "then worked on my woodworking lathe before playing a game"
        )
        self.assertLessEqual(len(result["topics"]), 5)

    # --- Message classification ---

    def test_classify_question(self):
        result = self.extractor.extract_all("What is your name?")
        self.assertIn("question", result["classification"])

    def test_classify_greeting(self):
        result = self.extractor.extract_all("Hello there!")
        self.assertIn("greeting", result["classification"])

    def test_classify_farewell(self):
        result = self.extractor.extract_all("Goodbye, see you later")
        self.assertIn("farewell", result["classification"])

    def test_classify_memory(self):
        result = self.extractor.extract_all("Remember when we talked about that?")
        self.assertIn("memory", result["classification"])

    def test_classify_opinion(self):
        result = self.extractor.extract_all("I think that's a great idea")
        self.assertIn("opinion", result["classification"])

    def test_classify_default_statement(self):
        result = self.extractor.extract_all("The sky is clear today")
        self.assertIn("statement", result["classification"])

    # --- Fact structure ---

    def test_fact_has_required_fields(self):
        result = self.extractor.extract_all("My name is Kenneth")
        for fact in result["facts"]:
            self.assertIn("text", fact)
            self.assertIn("tier", fact)
            self.assertIn("confidence", fact)
            self.assertIn("tags", fact)

    def test_entity_has_required_fields(self):
        result = self.extractor.extract_all("My dog is named Arix")
        for entity in result["entities"]:
            self.assertIn("text", entity)
            self.assertIn("tier", entity)
            self.assertIn("entity_type", entity)
            self.assertIn("confidence", entity)

    # --- Edge cases ---

    def test_empty_input(self):
        result = self.extractor.extract_all("")
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["entities"], [])

    def test_no_matches(self):
        result = self.extractor.extract_all("Hmm okay sure")
        self.assertEqual(result["facts"], [])


if __name__ == "__main__":
    unittest.main()
