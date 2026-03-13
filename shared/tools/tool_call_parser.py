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
Tool Call Parser
----------------
Parses <tool_call> blocks from LLM output, executes them through the
tool registry, and produces a clean response with results inlined.

The LLM emits tool calls in this format (Qwen native):
    <tool_call>
    {"name": "generate_image", "arguments": {"prompt": "a birthday cake"}}
    </tool_call>

The parser:
1. Extracts all <tool_call> blocks from the response
2. Validates tool names against the registry
3. Injects execution context (user_id, user_permission) into arguments
4. Executes each tool
5. Replaces the <tool_call> blocks with human-readable result summaries
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

# Pattern to match <tool_call>...</tool_call> blocks (possibly with whitespace)
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


@dataclass
class ToolCallResult:
    """Result of executing a single tool call."""
    tool_name: str
    arguments: dict
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None
    display_text: str = ""


def parse_tool_calls(response_text: str) -> list:
    """
    Extract tool call requests from LLM response text.

    Returns list of dicts: [{"name": str, "arguments": dict, "raw": str}, ...]
    """
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(response_text):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            calls.append({"name": name, "arguments": arguments, "raw": match.group(0)})
        except (json.JSONDecodeError, AttributeError):
            calls.append({"name": "", "arguments": {}, "raw": match.group(0), "parse_error": True})
    return calls


def has_tool_calls(response_text: str) -> bool:
    """Check if the response contains any tool call blocks."""
    return bool(TOOL_CALL_PATTERN.search(response_text))


def execute_tool_calls(
    response_text: str,
    tool_getter,
    user_id: int,
    user_permission: str = "adult",
) -> tuple:
    """
    Parse, execute, and inline tool call results into the response.

    Args:
        response_text: Raw LLM response that may contain <tool_call> blocks.
        tool_getter: Callable that takes a tool name and returns a callable or None.
                     Typically tool_registry.get_tool.
        user_id: Current user's ID (injected into tool args).
        user_permission: User's permission level (injected into tool args).

    Returns:
        (clean_response: str, results: list[ToolCallResult])
        clean_response has <tool_call> blocks replaced with result text.
    """
    calls = parse_tool_calls(response_text)
    if not calls:
        return response_text, []

    results = []
    clean = response_text

    for call in calls:
        if call.get("parse_error"):
            tcr = ToolCallResult(
                tool_name="",
                arguments={},
                success=False,
                error="Failed to parse tool call JSON",
                display_text="",
            )
            results.append(tcr)
            # Remove the malformed block silently
            clean = clean.replace(call["raw"], "")
            continue

        tool_name = call["name"]
        arguments = call["arguments"]

        # Look up the tool
        tool_func = tool_getter(tool_name)
        if tool_func is None:
            tcr = ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error=f"Unknown tool: {tool_name}",
                display_text="",
            )
            results.append(tcr)
            clean = clean.replace(call["raw"], "")
            continue

        # Inject execution context into arguments
        arguments["user_id"] = user_id
        arguments["user_permission"] = user_permission

        # Execute the tool
        try:
            result = tool_func(**arguments)
            display = _format_tool_result(tool_name, result)
            tcr = ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                success=True,
                result=result,
                display_text=display,
            )
        except Exception as e:
            tcr = ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                error=str(e),
                display_text="",
            )

        results.append(tcr)
        clean = clean.replace(call["raw"], tcr.display_text)

    # Clean up any extra whitespace from removed blocks
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()

    return clean, results


def _format_tool_result(tool_name: str, result) -> str:
    """
    Format a tool result into human-readable inline text.
    """
    if tool_name == "generate_image":
        if isinstance(result, dict):
            if result.get("success") and result.get("images"):
                paths = ", ".join(result["images"])
                return f"\n[Image: {paths}]\n"
            elif not result.get("success"):
                return ""  # silently drop failed generations
        return ""

    # Generic fallback for future tools
    if isinstance(result, dict):
        if "output" in result:
            return f"\n[{tool_name}: {result['output']}]\n"
        return f"\n[{tool_name}: done]\n"
    if isinstance(result, str):
        return f"\n[{tool_name}: {result}]\n"
    return ""
