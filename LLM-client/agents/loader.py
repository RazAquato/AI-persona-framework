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
CONFIG_PATH = os.getenv("AGENT_CONFIG_PATH")

#print("Shared ENV loaded from:", SHARED_ENV_PATH)
#print("Local ENV loaded from :", LOCAL_ENV_PATH)


def load_persona_config(personality_id: str = "default") -> dict:
    """
    Load configuration for a given personality.
    Returns a dictionary with fields like:
    {
        "name": "Eva",
        "description": "A poetic assistant",
        "system_prompt": "You are Eva, a poetic and thoughtful assistant...",
        ...
    }
    """
    try:
        with open(CONFIG_PATH, "r") as f:
            all_configs = json.load(f)

        if personality_id in all_configs:
            return all_configs[personality_id]
        else:
            print(f"[loader] Personality '{personality_id}' not found. Falling back to 'default'.")
            return all_configs.get("default", {
                "name": "Default",
                "system_prompt": "You are a helpful assistant."
            })

    except Exception as e:
        print(f"[loader] Error loading personality config: {e}")
        return {
            "name": "Fallback",
            "system_prompt": "You are a helpful assistant."
        }

