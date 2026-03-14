# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Immich Photo Archive Sync
-------------------------
Syncs photo metadata from Immich into the fact store as identity-tier facts.
Extracts people (faces), locations (EXIF GPS), and devices (camera models).

Usage:
    python3 sync_immich.py --user 52           # sync for user 52
    python3 sync_immich.py --user 52 --dry-run # show what would be synced
    python3 sync_immich.py --user 9999         # sync for test user

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

from tools.immich_sync import sync_immich


def main():
    parser = argparse.ArgumentParser(description="Sync Immich photo metadata into fact store")
    parser.add_argument("--user", type=int, required=True, help="User ID to associate facts with")
    parser.add_argument("--dry-run", action="store_true", help="Show facts without storing them")
    args = parser.parse_args()

    print(f"[immich-sync] Syncing photo data for user {args.user}...")

    result = sync_immich(user_id=args.user, dry_run=args.dry_run)

    if result.get("error"):
        print(f"[immich-sync] ERROR: {result['error']}")
        return

    print(f"[immich-sync] People found: {result['people_found']}")
    print(f"[immich-sync] Locations found: {result['locations_found']}")
    print(f"[immich-sync] Devices found: {result['devices_found']}")
    print(f"[immich-sync] Facts generated: {result['facts_generated']}")
    if not args.dry_run:
        print(f"[immich-sync] Old facts replaced: {result['deleted_old']}")

    if result["facts"]:
        print("\nFacts:")
        for fact in result["facts"]:
            tags = ", ".join(fact.get("tags", []))
            print(f"  [{fact['tier']}] {fact['text']}  ({tags})")

    if args.dry_run:
        print("\n[immich-sync] Dry run — no facts stored.")
    else:
        print(f"\n[immich-sync] Done.")


if __name__ == "__main__":
    main()
