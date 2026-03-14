#!/bin/bash
# AI-persona-framework — Monthly sync cron script
# Copyright (C) 2025 Kenneth Haider — GPLv3
#
# Syncs external data sources (Mealie recipes, Immich photos) into the
# fact store for user 52 (Kenneth). Each adapter does a full snapshot
# replacement: old facts are deleted, fresh ones inserted.
#
# Install with:
#   crontab -e
#   # Run at 03:00 on the 1st of each month
#   0 3 1 * * /home/kenneth/AI-persona-framework/archive/cron_sync_adapters.sh >> /home/kenneth/AI-persona-framework/archive/sync.log 2>&1
#

set -euo pipefail

VENV="/home/kenneth/venvs/AI-persona-framework-venv/bin/activate"
PROJECT="/home/kenneth/AI-persona-framework"
USER_ID=52
LOG_PREFIX="[cron-sync $(date '+%Y-%m-%d %H:%M')]"

source "$VENV"
cd "$PROJECT"

echo "$LOG_PREFIX Starting monthly sync for user $USER_ID"

echo "$LOG_PREFIX Running Mealie sync..."
python3 memory-server/scripts/sync_mealie.py --user "$USER_ID" || echo "$LOG_PREFIX WARNING: Mealie sync failed"

echo "$LOG_PREFIX Running Immich sync..."
python3 memory-server/scripts/sync_immich.py --user "$USER_ID" || echo "$LOG_PREFIX WARNING: Immich sync failed"

echo "$LOG_PREFIX Sync complete."
