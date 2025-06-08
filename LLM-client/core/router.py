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

# LLM-client/core/router.py
import os, re, sys

#from shared.tools import tool_registry
#from core import engine

# Add memory-server to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add LLM-client to path so we can import `core.engine`
LLM_CLIENT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(LLM_CLIENT_ROOT)

# Add shared tools to path
TOOL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared", "tools"))
sys.path.append(TOOL_PATH)

from core import engine
import tool_registry


# Config toggle: if True, tool failure messages go into the LLM
TOOL_FAILURES_GO_TO_LLM = False

def is_tool_command(user_input: str):
    return user_input.strip().startswith("/")

def parse_tool_command(user_input: str):
    """
    NOTE: we need better parsing, this is just a skeleton-code!
    Parses inputs like: /generate_image "cat with hat"
    Returns: ("generate_image", "cat with hat")
    """
    match = re.match(r'^/(\w+)\s*["“]?(.*?)["”]?$', user_input.strip())
    if match:
        return match.group(1), match.group(2)
    return None, None

def handle_user_input(user_input: str, session_id: str = None):
    """
    Router entry point. Routes to a tool or the core engine.
    Returns: assistant_output (str)
    """
    if is_tool_command(user_input):
        command, prompt = parse_tool_command(user_input)
        tool_func = tool_registry.get_tool(command)

        if tool_func:
            try:
                result = tool_func(prompt)
                if isinstance(result, str):
                    return result
                elif isinstance(result, dict) and "output" in result:
                    return result["output"]
                else:
                    return "✅ Tool executed, but no readable output was returned."
            except Exception as e:
                error_msg = f"⚠️ Tool '{command}' failed to execute."
                print(f"[router] Tool error: {e}")
                if TOOL_FAILURES_GO_TO_LLM:
                    user_input = f"{user_input}\n[NOTE: Tool '{command}' failed to run.]"
                else:
                    return error_msg
        else:
            error_msg = f"❌ Tool '{command}' not found."
            if TOOL_FAILURES_GO_TO_LLM:
                user_input = f"{user_input}\n[NOTE: Tool '{command}' is not available.]"
            else:
                return error_msg

    # Fallback: call LLM engine
    response = engine.process_input(user_input, session_id=session_id)
    return response

