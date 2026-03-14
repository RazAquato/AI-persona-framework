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
from unittest.mock import patch, MagicMock
from tools.mealie_sync import MealieClient, sync_mealie


FAKE_MEALPLAN = {
    "page": 1, "per_page": 50, "total": 5, "total_pages": 1,
    "items": [
        {"date": "2026-03-01", "entryType": "dinner",
         "recipe": {"id": "r1", "name": "Taco", "slug": "taco"}},
        {"date": "2026-03-02", "entryType": "dinner",
         "recipe": {"id": "r1", "name": "Taco", "slug": "taco"}},
        {"date": "2026-03-03", "entryType": "dinner",
         "recipe": {"id": "r1", "name": "Taco", "slug": "taco"}},
        {"date": "2026-03-04", "entryType": "dinner",
         "recipe": {"id": "r2", "name": "Pizza", "slug": "pizza"}},
        {"date": "2026-03-05", "entryType": "dinner",
         "recipe": {"id": "r3", "name": "Fish", "slug": "fish"}},
    ],
}

FAKE_RECIPE_TACO = {
    "name": "Taco", "slug": "taco", "id": "r1",
    "recipeCategory": [{"name": "Mexican"}],
    "tags": [{"name": "Taco"}],
    "recipeIngredient": [
        {"food": {"name": "kjøttdeig"}, "quantity": 400, "unit": {"name": "gram"}},
        {"food": {"name": "ost"}, "quantity": 200, "unit": {"name": "gram"}},
        {"food": {"name": "salsa"}, "quantity": 1, "unit": {"name": "glass"}},
    ],
}

FAKE_RECIPE_PIZZA = {
    "name": "Pizza", "slug": "pizza", "id": "r2",
    "recipeCategory": [{"name": "Italian"}],
    "tags": [],
    "recipeIngredient": [
        {"food": {"name": "ost"}, "quantity": 300, "unit": {"name": "gram"}},
        {"food": {"name": "tomatsaus"}, "quantity": 1, "unit": {"name": "boks"}},
    ],
}

FAKE_RECIPE_FISH = {
    "name": "Fish", "slug": "fish", "id": "r3",
    "recipeCategory": [],
    "tags": [],
    "recipeIngredient": [
        {"food": {"name": "torsk"}, "quantity": 500, "unit": {"name": "gram"}},
    ],
}

RECIPE_MAP = {"taco": FAKE_RECIPE_TACO, "pizza": FAKE_RECIPE_PIZZA, "fish": FAKE_RECIPE_FISH}


def _mock_get(path, params=None):
    if "/mealplans" in path:
        return FAKE_MEALPLAN
    for slug, recipe in RECIPE_MAP.items():
        if path.endswith(f"/{slug}"):
            return recipe
    return {"items": [], "total_pages": 1, "page": 1}


class TestMealieClient(unittest.TestCase):

    @patch.object(MealieClient, '_get', side_effect=_mock_get)
    def test_get_all_mealplans(self, mock_get):
        client = MealieClient("http://fake", "token")
        entries = client.get_all_mealplans()
        self.assertEqual(len(entries), 5)

    @patch.object(MealieClient, '_get', side_effect=_mock_get)
    def test_get_recipe_detail(self, mock_get):
        client = MealieClient("http://fake", "token")
        detail = client.get_recipe_detail("taco")
        self.assertEqual(detail["name"], "Taco")
        self.assertEqual(len(detail["recipeIngredient"]), 3)


class TestSyncMealie(unittest.TestCase):

    @patch.dict("os.environ", {"MEALIE_URL": "", "MEALIE_TOKEN": ""})
    def test_missing_credentials(self):
        result = sync_mealie(user_id=9999, base_url=None, token=None)
        self.assertIn("error", result)

    @patch("tools.mealie_sync.delete_facts_by_source", return_value=0)
    @patch("tools.mealie_sync.store_fact_blobs")
    @patch.object(MealieClient, '_get', side_effect=_mock_get)
    def test_sync_produces_recipe_facts(self, mock_get, mock_store, mock_delete):
        result = sync_mealie(user_id=9999, base_url="http://fake", token="token")
        self.assertEqual(result["mealplan_entries"], 5)
        self.assertEqual(result["recipes_synced"], 3)

        fact_texts = [f["text"] for f in result["facts"]]
        # Taco cooked 3 times should say "regularly"
        taco_fact = next(f for f in fact_texts if "Taco" in f)
        self.assertIn("3 times", taco_fact)
        self.assertIn("regularly", taco_fact)

        # Fish cooked once should say "has made"
        fish_fact = next(f for f in fact_texts if "Fish" in f)
        self.assertIn("has made", fish_fact)

    @patch("tools.mealie_sync.delete_facts_by_source", return_value=0)
    @patch("tools.mealie_sync.store_fact_blobs")
    @patch.object(MealieClient, '_get', side_effect=_mock_get)
    def test_sync_extracts_categories(self, mock_get, mock_store, mock_delete):
        result = sync_mealie(user_id=9999, base_url="http://fake", token="token")
        cat_facts = [f["text"] for f in result["facts"] if "food_category" in f["tags"]]
        self.assertTrue(any("Mexican" in f for f in cat_facts))
        self.assertTrue(any("Italian" in f for f in cat_facts))

    @patch("tools.mealie_sync.delete_facts_by_source", return_value=0)
    @patch("tools.mealie_sync.store_fact_blobs")
    @patch.object(MealieClient, '_get', side_effect=_mock_get)
    def test_sync_extracts_frequent_ingredients(self, mock_get, mock_store, mock_delete):
        result = sync_mealie(user_id=9999, base_url="http://fake", token="token")
        ingredient_facts = [f["text"] for f in result["facts"] if "ingredient" in f["tags"]]
        # "ost" appears in both taco and pizza
        self.assertTrue(any("ost" in f for f in ingredient_facts))
        # "torsk" only in fish (1 recipe) — should NOT appear
        self.assertFalse(any("torsk" in f for f in ingredient_facts))

    @patch("tools.mealie_sync.delete_facts_by_source", return_value=0)
    @patch("tools.mealie_sync.store_fact_blobs")
    @patch.object(MealieClient, '_get', side_effect=_mock_get)
    def test_confidence_scales_with_frequency(self, mock_get, mock_store, mock_delete):
        result = sync_mealie(user_id=9999, base_url="http://fake", token="token")
        taco = next(f for f in result["facts"] if "Taco" in f["text"])
        fish = next(f for f in result["facts"] if "Fish" in f["text"])
        self.assertGreater(taco["confidence"], fish["confidence"])

    @patch("tools.mealie_sync.delete_facts_by_source", return_value=0)
    @patch("tools.mealie_sync.store_fact_blobs")
    @patch.object(MealieClient, '_get', side_effect=_mock_get)
    def test_all_facts_are_identity_tier(self, mock_get, mock_store, mock_delete):
        result = sync_mealie(user_id=9999, base_url="http://fake", token="token")
        for fact in result["facts"]:
            self.assertEqual(fact["tier"], "identity")

    @patch("tools.mealie_sync.delete_facts_by_source", return_value=0)
    @patch("tools.mealie_sync.store_fact_blobs")
    @patch.object(MealieClient, '_get', side_effect=_mock_get)
    def test_all_facts_have_hobbies_domain(self, mock_get, mock_store, mock_delete):
        result = sync_mealie(user_id=9999, base_url="http://fake", token="token")
        for fact in result["facts"]:
            self.assertEqual(fact["domain"], "hobbies")

    @patch("tools.mealie_sync.delete_facts_by_source", return_value=0)
    @patch("tools.mealie_sync.store_fact_blobs")
    @patch.object(MealieClient, '_get', side_effect=_mock_get)
    def test_all_facts_have_mealie_source_type(self, mock_get, mock_store, mock_delete):
        result = sync_mealie(user_id=9999, base_url="http://fake", token="token")
        for fact in result["facts"]:
            self.assertEqual(fact["source_type"], "mealie")

    @patch("tools.mealie_sync.delete_facts_by_source", return_value=3)
    @patch("tools.mealie_sync.store_fact_blobs")
    @patch.object(MealieClient, '_get', side_effect=_mock_get)
    def test_sync_deletes_old_facts_before_insert(self, mock_get, mock_store, mock_delete):
        result = sync_mealie(user_id=9999, base_url="http://fake", token="token")
        mock_delete.assert_called_once_with(9999, "mealie")
        mock_store.assert_called_once()
        self.assertEqual(result["deleted_old"], 3)

    @patch.object(MealieClient, '_get', side_effect=_mock_get)
    def test_dry_run_does_not_store(self, mock_get):
        with patch("tools.mealie_sync.store_fact_blobs") as mock_store, \
             patch("tools.mealie_sync.delete_facts_by_source") as mock_delete:
            result = sync_mealie(user_id=9999, base_url="http://fake",
                                 token="token", dry_run=True)
            mock_store.assert_not_called()
            mock_delete.assert_not_called()
            self.assertGreater(result["facts_generated"], 0)

    @patch.object(MealieClient, '_get', return_value={
        "page": 1, "per_page": 50, "total": 0, "total_pages": 1, "items": []
    })
    def test_empty_mealplan_returns_zero(self, mock_get):
        result = sync_mealie(user_id=9999, base_url="http://fake", token="token",
                             dry_run=True)
        self.assertEqual(result["recipes_synced"], 0)
        self.assertEqual(result["facts_generated"], 0)


if __name__ == "__main__":
    unittest.main()
