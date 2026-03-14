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
import os
import psycopg2
from dotenv import load_dotenv

from memory.fact_store import (
    store_fact,
    store_fact_blobs,
    make_fact_blob,
    get_facts,
    get_facts_by_tag,
    get_facts_by_tier,
    get_facts_by_entity_type,
    delete_fact,
    get_top_facts,
    update_fact
)

class TestFactStore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
        load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

        cls.conn = psycopg2.connect(
            dbname=os.getenv("PG_DATABASE"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )
        cls.cur = cls.conn.cursor()
        cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("TestUser",))
        cls.test_user_id = cls.cur.fetchone()[0]
        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        cls.cur.execute("DELETE FROM facts WHERE user_id = %s;", (cls.test_user_id,))
        cls.cur.execute("DELETE FROM users WHERE id = %s;", (cls.test_user_id,))
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    def test_store_and_get_fact_with_tags(self):
        store_fact(self.test_user_id, "User likes sci-fi", tags=["books", "genre"])
        results = get_facts(self.test_user_id)
        self.assertTrue(any("sci-fi" in f[1] for f in results))

    def test_store_fact_with_none_tags(self):
        store_fact(self.test_user_id, "Fact with no tags", tags=None)
        results = get_facts(self.test_user_id)
        self.assertTrue(any("no tags" in f[1] for f in results))

    def test_store_duplicate_fact_is_deduped(self):
        unique = "Dedup old test fact ZZZ"
        # Clean up
        for f in get_facts(self.test_user_id):
            if f[1] == unique:
                delete_fact(f[0])
        store_fact(self.test_user_id, unique, tags=["test"])
        store_fact(self.test_user_id, unique, tags=["test"])
        results = [fact[1] for fact in get_facts(self.test_user_id)]
        self.assertEqual(results.count(unique), 1, "Dedup should prevent duplicate facts")

    def test_get_facts_empty(self):
        self.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("EmptyUser",))
        empty_user_id = self.cur.fetchone()[0]
        self.conn.commit()
        facts = get_facts(empty_user_id)
        self.assertEqual(facts, [])
        self.cur.execute("DELETE FROM users WHERE id = %s;", (empty_user_id,))
        self.conn.commit()

    def test_invalid_user_id(self):
        with self.assertRaises(Exception):
            store_fact("", "Invalid user test", tags=["error"])

    def test_get_facts_by_tag(self):
        store_fact(self.test_user_id, "Tagged with AI", tags=["ai", "tech"])
        tag_results = get_facts_by_tag(self.test_user_id, "ai")
        self.assertTrue(any("AI" in f[1] for f in tag_results))

    def test_get_top_facts(self):
        store_fact(self.test_user_id, "High relevance", relevance_score=0.95)
        store_fact(self.test_user_id, "Low relevance", relevance_score=0.1)
        top_facts = get_top_facts(self.test_user_id, limit=1)
        self.assertEqual(top_facts[0][1], "High relevance")

    def test_update_fact(self):
        store_fact(self.test_user_id, "Temp fact", tags=["old"], relevance_score=0.5)
        fact_id = get_top_facts(self.test_user_id, limit=1)[0][0]
        update_fact(fact_id, new_text="Updated fact", tags=["new"], relevance_score=0.9)
        updated = get_facts(self.test_user_id)
        self.assertTrue(any("Updated fact" in f[1] for f in updated))

    def test_delete_fact(self):
        store_fact(self.test_user_id, "To be deleted", tags=["temp"], relevance_score=0.5)  # FIXED

        top_facts = get_top_facts(self.test_user_id, limit=1)
        self.assertTrue(len(top_facts) > 0, "No facts returned from get_top_facts")

        fact_id = top_facts[0][0]
        delete_fact(fact_id)

        remaining = get_facts(self.test_user_id)
        self.assertFalse(any("To be deleted" in f[1] for f in remaining))

    # --- M2 additions: tier, entity_type, dedup ---

    def test_store_fact_with_tier_and_entity_type(self):
        store_fact(
            self.test_user_id, "User's wife is named Maria",
            tier="identity", entity_type="person",
            source_type="conversation", source_ref="123"
        )
        results = get_facts(self.test_user_id)
        maria = [f for f in results if "Maria" in f[1]]
        self.assertTrue(len(maria) > 0)
        # Tuple: (id, text, tags, relevance_score, tier, entity_type)
        self.assertEqual(maria[0][4], "identity")
        self.assertEqual(maria[0][5], "person")

    def test_get_facts_by_tier_single(self):
        store_fact(self.test_user_id, "Identity tier fact", tier="identity")
        store_fact(self.test_user_id, "Emotional tier fact", tier="emotional")
        identity_facts = get_facts_by_tier(self.test_user_id, ["identity"])
        self.assertTrue(all(f[4] == "identity" for f in identity_facts))

    def test_get_facts_by_tier_multiple(self):
        store_fact(self.test_user_id, "Multi-tier identity", tier="identity")
        store_fact(self.test_user_id, "Multi-tier emotional", tier="emotional")
        results = get_facts_by_tier(self.test_user_id, ["identity", "emotional"])
        tiers = {f[4] for f in results}
        self.assertTrue(tiers.issubset({"identity", "emotional"}))

    def test_get_facts_by_entity_type(self):
        store_fact(self.test_user_id, "User has pet named Rex",
                   tier="identity", entity_type="pet")
        results = get_facts_by_entity_type(self.test_user_id, "pet")
        self.assertTrue(any("Rex" in f[1] for f in results))

    def test_dedup_prevents_exact_duplicate(self):
        unique_text = "Unique dedup test fact XYZ123"
        # Clean up first
        for f in get_facts(self.test_user_id):
            if unique_text.lower() in f[1].lower():
                delete_fact(f[0])
        store_fact(self.test_user_id, unique_text)
        store_fact(self.test_user_id, unique_text)  # duplicate
        results = [f for f in get_facts(self.test_user_id) if f[1] == unique_text]
        self.assertEqual(len(results), 1, "Dedup should prevent exact duplicate")

    def test_dedup_case_insensitive(self):
        unique_text_lower = "case insensitive dedup test abc789"
        unique_text_upper = "Case Insensitive Dedup Test ABC789"
        for f in get_facts(self.test_user_id):
            if unique_text_lower in f[1].lower():
                delete_fact(f[0])
        store_fact(self.test_user_id, unique_text_lower)
        store_fact(self.test_user_id, unique_text_upper)  # should be deduped
        results = [f for f in get_facts(self.test_user_id)
                   if f[1].lower() == unique_text_lower]
        self.assertEqual(len(results), 1, "Dedup should be case-insensitive")

    def test_get_facts_with_tier_filter(self):
        store_fact(self.test_user_id, "Tier filter test", tier="emotional")
        results = get_facts(self.test_user_id, tier="emotional")
        self.assertTrue(all(f[4] == "emotional" for f in results))

    # --- Additional M2 coverage ---

    def test_store_fact_dedup_returns_none(self):
        """Storing a duplicate fact should return None."""
        unique = "Dedup return None test QQQ"
        for f in get_facts(self.test_user_id):
            if f[1] == unique:
                delete_fact(f[0])
        first_id = store_fact(self.test_user_id, unique)
        self.assertIsNotNone(first_id)
        second_id = store_fact(self.test_user_id, unique)
        self.assertIsNone(second_id)

    def test_update_fact_tier_and_entity_type(self):
        """update_fact should update tier and entity_type fields."""
        fact_id = store_fact(self.test_user_id, "Update tier test", tier="emotional")
        self.assertIsNotNone(fact_id)
        update_fact(fact_id, tier="identity", entity_type="person")
        results = get_facts(self.test_user_id)
        updated = next((f for f in results if f[0] == fact_id), None)
        self.assertIsNotNone(updated)
        self.assertEqual(updated[4], "identity")
        self.assertEqual(updated[5], "person")

    def test_update_fact_no_changes_noop(self):
        """update_fact with no arguments should not raise."""
        fact_id = store_fact(self.test_user_id, "Noop update test")
        self.assertIsNotNone(fact_id)
        update_fact(fact_id)  # Should not raise

    def test_get_top_facts_sort_by_id(self):
        """get_top_facts with sort_by='id' should return most recent first."""
        store_fact(self.test_user_id, "Sort by id old", relevance_score=0.99)
        store_fact(self.test_user_id, "Sort by id new", relevance_score=0.01)
        results = get_top_facts(self.test_user_id, limit=2, sort_by="id")
        self.assertEqual(results[0][1], "Sort by id new")

    def test_get_top_facts_invalid_sort_falls_back(self):
        """Invalid sort_by should fallback to relevance_score."""
        store_fact(self.test_user_id, "Invalid sort test", relevance_score=0.5)
        results = get_top_facts(self.test_user_id, limit=1, sort_by="invalid")
        self.assertIsInstance(results, list)

    def test_store_fact_all_source_fields(self):
        """Verify source_type and source_ref are accepted without error."""
        fact_id = store_fact(
            self.test_user_id, "Full source fields test",
            source_type="image_analysis", source_ref="img_001"
        )
        self.assertIsNotNone(fact_id)

    def test_get_facts_by_tier_empty_list(self):
        """get_facts_by_tier with empty tier list should return empty."""
        results = get_facts_by_tier(self.test_user_id, [])
        self.assertEqual(results, [])

    # --- Batch 2: store_fact_blobs and make_fact_blob ---

    def test_make_fact_blob(self):
        """make_fact_blob should create a dict with all standard keys."""
        blob = make_fact_blob("Test fact", tier="identity", entity_type="person",
                              confidence=0.9, tags=["test"], source_type="document",
                              source_ref="doc_001")
        self.assertEqual(blob["text"], "Test fact")
        self.assertEqual(blob["tier"], "identity")
        self.assertEqual(blob["entity_type"], "person")
        self.assertEqual(blob["confidence"], 0.9)
        self.assertEqual(blob["source_type"], "document")

    def test_make_fact_blob_defaults(self):
        """make_fact_blob defaults should be sensible."""
        blob = make_fact_blob("Minimal")
        self.assertEqual(blob["tier"], "identity")
        self.assertIsNone(blob["entity_type"])
        self.assertEqual(blob["tags"], [])
        self.assertEqual(blob["source_type"], "conversation")
        self.assertIsNone(blob["valence"])

    def test_store_fact_blobs_basic(self):
        """store_fact_blobs should insert multiple facts in one call."""
        blobs = [
            make_fact_blob("Bulk test fact alpha 001"),
            make_fact_blob("Bulk test fact beta 002"),
        ]
        ids = store_fact_blobs(self.test_user_id, blobs)
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_store_fact_blobs_dedup_within_batch(self):
        """Duplicate facts within same batch should be deduped."""
        blobs = [
            make_fact_blob("Batch dedup test gamma 003"),
            make_fact_blob("Batch dedup test gamma 003"),
        ]
        ids = store_fact_blobs(self.test_user_id, blobs)
        self.assertIsNotNone(ids[0])
        self.assertIsNone(ids[1])

    def test_store_fact_blobs_dedup_against_db(self):
        """Facts already in DB should be deduped."""
        text = "DB dedup test delta 004"
        store_fact(self.test_user_id, text)
        ids = store_fact_blobs(self.test_user_id, [make_fact_blob(text)])
        self.assertIsNone(ids[0])

    def test_store_fact_blobs_empty_list(self):
        """Empty blob list should return empty list."""
        ids = store_fact_blobs(self.test_user_id, [])
        self.assertEqual(ids, [])

    def test_store_fact_blobs_source_override(self):
        """Source type/ref override should apply to blobs without their own."""
        blobs = [make_fact_blob("Source override test epsilon 005")]
        ids = store_fact_blobs(self.test_user_id, blobs,
                               source_type="image_analysis", source_ref="img_99")
        self.assertIsNotNone(ids[0])

    def test_store_fact_blobs_skips_empty_text(self):
        """Blobs with empty text should be skipped."""
        blobs = [{"text": "", "tier": "identity"}, {"text": "   ", "tier": "identity"}]
        ids = store_fact_blobs(self.test_user_id, blobs)
        self.assertEqual(ids, [None, None])

    # --- Valence support ---

    def test_store_fact_with_valence(self):
        """store_fact should accept and store valence."""
        fact_id = store_fact(
            self.test_user_id, "User's friend Lars is great",
            tier="identity", entity_type="person", valence="positive"
        )
        self.assertIsNotNone(fact_id)

    def test_make_fact_blob_with_valence(self):
        """make_fact_blob should include valence."""
        blob = make_fact_blob("User dislikes Tom", tier="emotional",
                              entity_type="person", valence="negative")
        self.assertEqual(blob["valence"], "negative")
        self.assertEqual(blob["tier"], "emotional")

    def test_store_fact_blobs_with_valence(self):
        """store_fact_blobs should handle blobs with valence."""
        blobs = [
            make_fact_blob("Valence test positive 001", tier="identity",
                           entity_type="person", valence="positive"),
            make_fact_blob("Valence test negative 002", tier="emotional",
                           entity_type="person", valence="negative"),
        ]
        ids = store_fact_blobs(self.test_user_id, blobs)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_get_facts_by_tier_emotional(self):
        """get_facts_by_tier should work with the new 'emotional' tier."""
        store_fact(self.test_user_id, "Emotional tier test fact", tier="emotional")
        results = get_facts_by_tier(self.test_user_id, ["emotional"])
        self.assertTrue(any(f[4] == "emotional" for f in results))

    def test_get_facts_by_tier_identity_and_emotional(self):
        """get_facts_by_tier with both tiers should return both."""
        store_fact(self.test_user_id, "Two-tier identity test", tier="identity")
        store_fact(self.test_user_id, "Two-tier emotional test", tier="emotional")
        results = get_facts_by_tier(self.test_user_id, ["identity", "emotional"])
        tiers = {f[4] for f in results}
        self.assertTrue(tiers.issubset({"identity", "emotional"}))


if __name__ == "__main__":
    unittest.main()

