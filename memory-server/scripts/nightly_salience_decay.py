#!/usr/bin/env python3
# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

"""
Nightly Salience Decay Script

Applies time-based decay to all topic salience records.
Run as a cron job (e.g., daily at 3am).

Usage:
    python3 memory-server/scripts/nightly_salience_decay.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, ".."))
if MEMORY_PATH not in sys.path:
    sys.path.insert(0, MEMORY_PATH)

from memory.topic_store import decay_all_salience

if __name__ == "__main__":
    updated = decay_all_salience()
    print(f"[nightly_decay] Decayed {updated} salience records.")
