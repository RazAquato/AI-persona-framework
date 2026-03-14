# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Mealie Recipe Sync
------------------
Syncs meal plan history from Mealie into the fact store as identity-tier facts.
Frequency-based: recipes cooked more often get higher confidence scores.

Usage:
    python3 sync_mealie.py --user 52           # sync for user 52
    python3 sync_mealie.py --user 52 --dry-run # show what would be synced
    python3 sync_mealie.py --user 9999         # sync for test user

Designed to run as a monthly cron job.
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, ".."))
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared"))

for p in [MEMORY_PATH, SHARED_PATH]:
    if p not in sys.path:
        sys.path.append(p)

from dotenv import load_dotenv

SHARED_ENV = os.path.abspath(os.path.join(SHARED_PATH, "config", ".env"))
load_dotenv(dotenv_path=SHARED_ENV)

from tools.mealie_sync import sync_mealie


def main():
    parser = argparse.ArgumentParser(description="Sync Mealie recipes into fact store")
    parser.add_argument("--user", type=int, required=True, help="User ID to associate facts with")
    parser.add_argument("--dry-run", action="store_true", help="Show facts without storing them")
    args = parser.parse_args()

    print(f"[mealie-sync] Syncing meal data for user {args.user}...")

    result = sync_mealie(user_id=args.user, dry_run=args.dry_run)

    if result.get("error"):
        print(f"[mealie-sync] ERROR: {result['error']}")
        return

    print(f"[mealie-sync] Meal plan entries: {result['mealplan_entries']}")
    print(f"[mealie-sync] Unique recipes cooked: {result['recipes_synced']}")
    print(f"[mealie-sync] Facts generated: {result['facts_generated']}")
    if not args.dry_run:
        print(f"[mealie-sync] Old facts replaced: {result['deleted_old']}")

    if result["facts"]:
        print("\nFacts:")
        for fact in result["facts"]:
            print(f"  [{fact['tier']}] {fact['text']}")

    if args.dry_run:
        print("\n[mealie-sync] Dry run — no facts stored.")
    else:
        print(f"\n[mealie-sync] Done.")


if __name__ == "__main__":
    main()
