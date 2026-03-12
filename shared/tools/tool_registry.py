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

# shared/tools/tool_registry.py
import os, sys
# Add shared to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add LLM-client to path so we can import `core.engine`
ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.append(ROOT)

# Import all tool modules
from shared.tools import image_gen #, web_research, sandbox_env

# Define a registry of available tools
# Map: command name (str) → callable function
TOOLS = {
    "generate_image": image_gen.generate,
    #"web_search": web_research.search_web,
    #"run_code": sandbox_env.run_code,
    #"generate_chart": plot_chart.generate
}

def get_tool(tool_name: str):
    """
    Get the callable function for a tool by name.
    Returns None if tool is not registered.
    """
    return TOOLS.get(tool_name)

def list_tools():
    """
    Returns a list of all registered tool names.
    """
    return list(TOOLS.keys())

def describe_tools():
    """
    Returns a dict of tool names and descriptions.
    Used by LLM to understand tool capabilities.
    """
    return {
        "generate_image": "Creates an image from a text prompt via ComfyUI. "
                          "Accepts: prompt (str), user_id (int), user_permission (str), "
                          "width (int), height (int), seed (int), negative_prompt (str), "
                          "workflow (str). Returns dict with success, images, error.",
    }

