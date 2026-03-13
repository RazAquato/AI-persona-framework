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
Tool Registry
-------------
Central registry of tools available to both the router (user /commands)
and the LLM (autonomous tool calls via <tool_call> blocks).

Each tool has:
- A callable function
- A JSON schema definition (for injection into the LLM system prompt)
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from shared.tools import image_gen

# ──────────────────────────────────────────────────────────────
# Tool function registry: command name → callable
# ──────────────────────────────────────────────────────────────
TOOLS = {
    "generate_image": image_gen.generate,
}

# ──────────────────────────────────────────────────────────────
# Tool definitions for LLM (Qwen/OpenAI function-calling format)
# The LLM sees these in the system prompt to know what it can call.
# user_id and user_permission are injected by the engine at execution
# time — the LLM never needs to specify them.
# ──────────────────────────────────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate an image from a text description using AI image generation. "
                "Use this to create visual gifts, illustrate scenes, or share images "
                "in conversation. The image will be generated and shown to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Detailed description of the image to generate. "
                            "Be specific about subject, style, mood, and composition."
                        ),
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "What to avoid in the image (optional).",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


def get_tool(tool_name: str):
    """Get the callable function for a tool by name. Returns None if not found."""
    return TOOLS.get(tool_name)


def list_tools():
    """Returns a list of all registered tool names."""
    return list(TOOLS.keys())


def get_tool_definitions() -> list:
    """
    Returns the tool definitions list for injection into the LLM prompt.
    Format matches Qwen/OpenAI function-calling schema.
    """
    return TOOL_DEFINITIONS


def describe_tools() -> dict:
    """
    Returns a dict of tool names → human-readable descriptions.
    """
    return {
        defn["function"]["name"]: defn["function"]["description"]
        for defn in TOOL_DEFINITIONS
    }
