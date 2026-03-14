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

"""
Mealie Sync Adapter
-------------------
Fetches meal plan history from a Mealie instance and converts it into
identity-tier facts for the user's knowledge profile.

Frequency-based approach: each meal plan entry is one actual meal cooked.
Counting how often each recipe appears reveals real preferences —
"Taco (cooked 5 times)" beats "Fish (cooked once)".

Each sync is a full snapshot: old mealie facts are deleted and replaced
with fresh ones, so counts stay accurate as the meal plan grows.

Facts produced:
  - "User makes {recipe} (cooked {N} times)" — per recipe, with frequency
  - "User cooks {category} dishes" — per unique category
  - "User frequently uses {ingredient}" — ingredients appearing in 2+ recipes

All facts use source_type="mealie" for clean snapshot replacement.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, ".."))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "memory-server"))
for p in [SHARED_PATH, MEMORY_PATH]:
    if p not in sys.path:
        sys.path.append(p)

from memory.fact_store import store_fact_blobs, make_fact_blob, delete_facts_by_source


class MealieClient:
    """Minimal HTTP client for the Mealie REST API."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{qs}"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def get_all_mealplans(self) -> list:
        """Fetch all meal plan entries (paginated)."""
        entries = []
        page = 1
        while True:
            data = self._get("/api/households/mealplans", {"page": page, "perPage": 50})
            entries.extend(data.get("items", []))
            if page >= data.get("total_pages", 1):
                break
            page += 1
        return entries

    def get_all_recipes(self) -> list:
        """Fetch all recipes (paginated)."""
        recipes = []
        page = 1
        while True:
            data = self._get("/api/recipes", {"page": page, "perPage": 50})
            recipes.extend(data.get("items", []))
            if page >= data.get("total_pages", 1):
                break
            page += 1
        return recipes

    def get_recipe_detail(self, slug: str) -> dict:
        """Fetch full recipe detail including ingredients."""
        return self._get(f"/api/recipes/{slug}")


def sync_mealie(user_id: int, base_url: str = None, token: str = None,
                dry_run: bool = False) -> dict:
    """
    Sync meal data from Mealie into the fact store.

    Uses meal plan history for frequency counting — each entry is one
    actual meal cooked. Old mealie facts are replaced with a fresh snapshot.

    Args:
        user_id: the user who owns these facts
        base_url: Mealie server URL (falls back to MEALIE_URL env var)
        token: Mealie API token (falls back to MEALIE_TOKEN env var)
        dry_run: if True, return facts without storing them

    Returns:
        dict with keys: mealplan_entries, recipes_synced, facts_generated,
                        facts, deleted_old
    """
    base_url = base_url or os.getenv("MEALIE_URL")
    token = token or os.getenv("MEALIE_TOKEN")

    if not base_url or not token:
        return {"error": "MEALIE_URL and MEALIE_TOKEN must be set",
                "mealplan_entries": 0, "recipes_synced": 0,
                "facts_generated": 0, "facts": [], "deleted_old": 0}

    client = MealieClient(base_url, token)

    # 1. Fetch meal plan history — this is the frequency source
    mealplan_entries = client.get_all_mealplans()

    # Count how often each recipe was cooked
    recipe_cook_count = Counter()
    recipe_names = {}
    recipe_slugs = {}

    for entry in mealplan_entries:
        recipe = entry.get("recipe")
        if not recipe:
            continue
        recipe_id = recipe.get("id", "")
        recipe_name = recipe.get("name", "")
        recipe_slug = recipe.get("slug", "")
        if recipe_id and recipe_name:
            recipe_cook_count[recipe_id] += 1
            recipe_names[recipe_id] = recipe_name
            recipe_slugs[recipe_id] = recipe_slug

    if not recipe_cook_count:
        return {"mealplan_entries": len(mealplan_entries), "recipes_synced": 0,
                "facts_generated": 0, "facts": [], "deleted_old": 0}

    # 2. Fetch full recipe details for categories and ingredients
    all_categories = set()
    ingredient_counter = Counter()

    for recipe_id, slug in recipe_slugs.items():
        try:
            detail = client.get_recipe_detail(slug)

            for cat in detail.get("recipeCategory", []):
                cat_name = cat.get("name", "") if isinstance(cat, dict) else str(cat)
                if cat_name:
                    all_categories.add(cat_name)

            for ing in detail.get("recipeIngredient", []):
                food = ing.get("food")
                if isinstance(food, dict):
                    food_name = food.get("name", "")
                elif food:
                    food_name = str(food)
                else:
                    continue
                if food_name:
                    ingredient_counter[food_name.lower()] += 1
        except (urllib.error.URLError, Exception):
            pass

    # 3. Build facts
    facts = []

    # Recipe facts with frequency
    for recipe_id, count in recipe_cook_count.most_common():
        name = recipe_names[recipe_id]
        if count == 1:
            text = f"User has made {name}"
        else:
            text = f"User makes {name} regularly (cooked {count} times)"

        # Higher confidence for more frequently cooked recipes
        confidence = min(0.95, 0.5 + (count * 0.1))

        facts.append(make_fact_blob(
            text=text,
            tier="identity",
            tags=["mealie", "recipe"],
            source_type="mealie",
            source_ref=recipe_id,
            confidence=confidence,
            domain="hobbies",
        ))

    # Category facts
    for cat_name in sorted(all_categories):
        facts.append(make_fact_blob(
            text=f"User cooks {cat_name} dishes",
            tier="identity",
            tags=["mealie", "food_category"],
            source_type="mealie",
            confidence=0.7,
            domain="hobbies",
        ))

    # Frequent ingredients (used in 2+ recipes)
    for ingredient, count in ingredient_counter.most_common():
        if count >= 2:
            facts.append(make_fact_blob(
                text=f"User frequently uses {ingredient} in cooking",
                tier="identity",
                tags=["mealie", "ingredient"],
                source_type="mealie",
                confidence=0.6,
                domain="hobbies",
            ))

    # 4. Store: delete old snapshot, write new one
    deleted_old = 0
    if not dry_run:
        deleted_old = delete_facts_by_source(user_id, "mealie")
        store_fact_blobs(user_id, facts, source_type="mealie")

    return {
        "mealplan_entries": len(mealplan_entries),
        "recipes_synced": len(recipe_cook_count),
        "facts_generated": len(facts),
        "facts": facts,
        "deleted_old": deleted_old,
    }
