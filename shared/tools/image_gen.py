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
Image Generation Tool
---------------------
Top-level tool function registered in the tool registry.
Delegates to ImageOrchestrator for safety, workflow selection, and ComfyUI API.
"""

from tools.image_orchestrator import ImageOrchestrator

_orchestrator = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ImageOrchestrator()
    return _orchestrator


def generate(prompt: str, user_id: int = 9999, user_permission: str = "adult",
             persona_id: int = None, width: int = 512, height: int = 512,
             seed: int = None, negative_prompt: str = None,
             workflow: str = None) -> dict:
    """
    Generate an image from a text prompt.

    Args:
        prompt: Text description of the desired image
        user_id: Requesting user's ID
        user_permission: "adult", "teen", or "child"
        persona_id: Persona ID (for output folder structure)
        width: Image width
        height: Image height
        seed: Random seed (None = random)
        negative_prompt: What to avoid
        workflow: Override workflow name

    Returns:
        dict with success, images, error, etc.
    """
    orch = _get_orchestrator()
    return orch.generate(
        prompt=prompt,
        user_id=user_id,
        user_permission=user_permission,
        persona_id=persona_id,
        workflow_name=workflow,
        width=width,
        height=height,
        seed=seed,
        negative_prompt=negative_prompt,
    )
