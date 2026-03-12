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

import os
import json
from dotenv import load_dotenv

# Base path of current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load shared config first
SHARED_ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared", "config", ".env"))
load_dotenv(dotenv_path=SHARED_ENV_PATH)

# Then load local config (overrides shared if keys overlap)
LOCAL_ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "config", ".env"))
load_dotenv(dotenv_path=LOCAL_ENV_PATH, override=True)

#print("Exists:", os.path.exists(LOCAL_ENV_PATH))
#print("Exists:", os.path.exists(SHARED_ENV_PATH))
_config_raw = os.getenv("AGENT_CONFIG_PATH", "./config/personality_config.json")
# Resolve relative paths from the LLM-client root (BASE_DIR/../) so this works
# regardless of the process working directory
CONFIG_PATH = _config_raw if os.path.isabs(_config_raw) else os.path.abspath(
    os.path.join(BASE_DIR, "..", _config_raw)
)

#print("Shared ENV loaded from:", SHARED_ENV_PATH)
#print("Local ENV loaded from :", LOCAL_ENV_PATH)


DEFAULT_MEMORY_SCOPE = {
    "tier1": True,
    "tier2": "all",
    "tier3": "private",
}


def load_persona_config(personality_id: str = "default") -> dict:
    """
    Load configuration for a given personality.
    Returns a dictionary with fields like:
    {
        "name": "Eva",
        "description": "A poetic assistant",
        "system_prompt": "You are Eva, a poetic and thoughtful assistant...",
        "memory_scope": {"tier1": true, "tier2": "all", "tier3": "private"},
        ...
    }

    If memory_scope is missing from the config, a default scope is applied
    (tier1 always on, tier2 all, tier3 private).
    """
    try:
        with open(CONFIG_PATH, "r") as f:
            all_configs = json.load(f)

        if personality_id in all_configs:
            config = all_configs[personality_id]
        else:
            print(f"[loader] Personality '{personality_id}' not found. Falling back to 'default'.")
            config = all_configs.get("default", {
                "name": "Default",
                "system_prompt": "You are a helpful assistant."
            })

        return _ensure_memory_scope(config)

    except Exception as e:
        print(f"[loader] Error loading personality config: {e}")
        return _ensure_memory_scope({
            "name": "Fallback",
            "system_prompt": "You are a helpful assistant."
        })


def _ensure_memory_scope(config: dict) -> dict:
    """Ensure memory_scope exists and has valid structure."""
    if "memory_scope" not in config:
        config["memory_scope"] = dict(DEFAULT_MEMORY_SCOPE)
    else:
        scope = config["memory_scope"]
        if "tier1" not in scope:
            scope["tier1"] = True
        if "tier2" not in scope:
            scope["tier2"] = "all"
        if "tier3" not in scope:
            scope["tier3"] = "private"
    return config

