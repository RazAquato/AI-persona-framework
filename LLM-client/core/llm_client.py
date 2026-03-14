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
LLM Client
----------
Calls llama.cpp's OpenAI-compatible /v1/chat/completions endpoint.
Supports native tool calling (the server handles chat template formatting).
"""

import os
import json
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load shared config first
SHARED_ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared", "config", ".env"))
load_dotenv(dotenv_path=SHARED_ENV_PATH)

# Then load local config (overrides shared if keys overlap)
LOCAL_ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "config", ".env"))
load_dotenv(dotenv_path=LOCAL_ENV_PATH, override=True)

LLM_SERVER = os.getenv("LLM_SERVER")
CHAT_URL = LLM_SERVER.rstrip("/") + "/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}


def call_llm(messages, temperature=0.7, max_tokens=1024, tools=None,
             response_format=None):
    """
    Call llama.cpp's OpenAI-compatible chat completions endpoint.

    Args:
        messages: List of {"role": str, "content": str} dicts.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        tools: Optional list of tool definitions (OpenAI function-calling format).
        response_format: Optional response format spec (e.g. JSON schema for
                         grammar-constrained output). Passed directly to the API.

    Returns:
        {
            "content": str,           # The assistant's text reply
            "tool_calls": list,       # Any tool calls the model wants to make
            "reasoning": str,         # The model's thinking (if available)
            "raw": dict,              # Full API response
        }
    """
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if response_format:
        payload["response_format"] = response_format

    try:
        resp = requests.post(CHAT_URL, json=payload, headers=HEADERS, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls", [])
        reasoning = message.get("reasoning_content", "")
        finish_reason = choice.get("finish_reason", "")

        return {
            "content": content.strip(),
            "tool_calls": tool_calls,
            "reasoning": reasoning,
            "finish_reason": finish_reason,
            "raw": data,
        }
    except requests.RequestException as e:
        print("[LLM ERROR] {}".format(e))
        return {
            "content": "Sorry, I'm unavailable.",
            "tool_calls": [],
            "reasoning": "",
            "finish_reason": "error",
            "raw": {},
        }
