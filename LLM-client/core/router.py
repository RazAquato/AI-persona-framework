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

# User-friendly command aliases -> registry tool names
COMMAND_ALIASES = {
    "image": "generate_image",
    "img": "generate_image",
}


def is_tool_command(user_input: str):
    return user_input.strip().startswith("/")


def parse_tool_command(user_input: str):
    """
    Parses inputs like: /generate_image "cat with hat" or /image a sunset
    Returns: ("generate_image", "cat with hat")
    """
    match = re.match(r'^/(\w+)\s*["\u201c]?(.*?)["\u201d]?$', user_input.strip())
    if match:
        command = match.group(1)
        # Resolve aliases
        command = COMMAND_ALIASES.get(command, command)
        return command, match.group(2)
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
                elif isinstance(result, dict):
                    if "output" in result:
                        return result["output"]
                    elif "images" in result:
                        if result.get("success"):
                            imgs = ", ".join(result["images"]) if result["images"] else "no output"
                            return "Image generated: " + imgs
                        else:
                            return "Image generation failed: " + result.get("error", "unknown error")
                    else:
                        return "Tool executed: " + str(result)
                else:
                    return "Tool executed, but no readable output was returned."
            except Exception as e:
                error_msg = "Tool '{}' failed to execute.".format(command)
                print("[router] Tool error: {}".format(e))
                if TOOL_FAILURES_GO_TO_LLM:
                    user_input = "{}\n[NOTE: Tool '{}' failed to run.]".format(user_input, command)
                else:
                    return error_msg
        else:
            error_msg = "Tool '{}' not found.".format(command)
            if TOOL_FAILURES_GO_TO_LLM:
                user_input = "{}\n[NOTE: Tool '{}' is not available.]".format(user_input, command)
            else:
                return error_msg

    # Fallback: call LLM engine
    response = engine.process_input(user_input, session_id=session_id)
    return response
