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
import requests
from dotenv import load_dotenv

import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load shared config first
SHARED_ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared", "config", ".env"))
load_dotenv(dotenv_path=SHARED_ENV_PATH)

# Then load local config (overrides shared if keys overlap)
LOCAL_ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "config", ".env"))
load_dotenv(dotenv_path=LOCAL_ENV_PATH, override=True)

LLM_SERVER = os.getenv("LLM_SERVER")
COMPLETION_URL = LLM_SERVER.rstrip("/") + "/completion"
HEADERS = {"Content-Type": "application/json"}


def format_prompt(messages, tools=None):
    """
    Format prompt using llama.cpp's <|im_start|> chat template.
    Includes <tools> section if provided.
    """
    # System/message header
    if tools:
        # System block + tools doc
        sys_msg = next((m for m in messages if m["role"] == "system"), None)
        content = sys_msg["content"] if sys_msg else "You are a helpful assistant."
        prompt = (
            f"<|im_start|>system\n"
            f"{content}\n\n"
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
        )
        for t in tools:
            prompt += json.dumps(t) + "\n"
        prompt += (
            "</tools>\n\n"
            "For each function call, return a json object with function name "
            "and arguments within <tool_call></tool_call> tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\n"
            "<|im_end|>\n"
        )
    else:
        # Only system block
        sys_msg = next((m for m in messages if m["role"] == "system"), None)
        content = sys_msg["content"] if sys_msg else "You are a helpful assistant."
        prompt = f"<|im_start|>system\n{content}<|im_end|>\n"

    # Other messages
    for m in messages:
        if m["role"] == "system":
            continue
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"

    # Ready for assistant generation
    prompt += "<|im_start|>assistant\n"
    return prompt


def call_llm(messages, temperature=0.7, max_tokens=1024, tools=None):
    """
    Calls llama.cpp /completion with a manually formatted prompt.
    """
    prompt = format_prompt(messages, tools)
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "n_predict": max_tokens
    }
    print(f"payload={payload}")

    try:
        resp = requests.post(COMPLETION_URL, json=payload, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        return {"content": data.get("content", "").strip(), "raw": data}
    except requests.RequestException as e:
        print(f"[LLM ERROR] {e}")
        return {"error": str(e), "content": "Sorry, I'm unavailable.", "tool_calls": []}

