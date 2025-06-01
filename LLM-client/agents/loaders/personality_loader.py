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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import json
import yaml
from .agent_loader import load_user_agents
from dotenv import load_dotenv

# Base path of current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load shared config first
SHARED_ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "shared", "config", ".env"))
load_dotenv(dotenv_path=SHARED_ENV_PATH)

# Then load local config (overrides shared if keys overlap)
EMOTION_CONFIG_PATH=os.path.join(os.path.dirname(SHARED_ENV_PATH), "emotions.yaml")
LOCAL_ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "config", ".env"))

def load_emotion_defaults():
    with open(EMOTION_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return {k: 0.0 for k in config.get("emotions", {})}

def apply_emotion_defaults(personality_config):
    defaults = load_emotion_defaults()
    if "default_emotions" not in personality_config:
        personality_config["default_emotions"] = defaults
    else:
        # Fill in any missing emotions with default value
        for key, val in defaults.items():
            personality_config["default_emotions"].setdefault(key, val)
    return personality_config

def load_active_agent(user_id, agent_name):
    """
    Load a specific agent by name for a given user, with emotion defaults injected.
    """
    agents = load_user_agents(user_id)
    for agent in agents:
        if agent["name"].lower() == agent_name.lower():
            return apply_emotion_defaults(agent["config"])
    raise ValueError(f"Agent '{agent_name}' not found for user {user_id}")

if __name__ == "__main__":
    import pprint
    user_id = 52
    agent_name = "HelperBot"
    pprint.pprint(load_active_agent(user_id, agent_name))
