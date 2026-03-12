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
Image Generation Orchestrator
------------------------------
The bridge between the chatbot and ComfyUI.

Responsibilities:
1. Safety Layer 1 — Pre-generation prompt filtering based on user permissions
2. Workflow selection — Choose the right ComfyUI workflow template
3. Submit to ComfyUI API and return results
4. (Future) Safety Layer 2 — Post-generation output classification

Architecture (from design doc):
    Chatbot engine
      → ImageOrchestrator.generate(prompt, user_id, ...)
        → check_user_permissions(user_id)
        → select_workflow(user_permissions, prompt)
        → apply_prompt_filter(prompt, user_permissions)
        → ComfyUIClient.generate(workflow)
        → (future) classify_output(image_path)
      → Return image path(s) to chatbot

User permission levels:
    "adult"    — full access, all workflows including NSFW
    "teen"     — SFW workflows only, prompt sanitization
    "child"    — SFW workflows only, strict prompt sanitization
"""

import os
import re
import sys
from typing import Optional

# Path setup for cross-package imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if SHARED_ROOT not in sys.path:
    sys.path.append(SHARED_ROOT)

from tools.comfyui_client import ComfyUIClient
from tools.comfyui_workflows import get_workflow, list_workflows, sd15_basic

# Default ComfyUI server URL (same machine)
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

# Words that trigger NSFW filtering for non-adult users
NSFW_KEYWORDS = {
    "nude", "naked", "nsfw", "erotic", "sexual", "porn", "xxx",
    "topless", "lingerie", "hentai", "lewd", "explicit",
    "undressed", "bondage", "fetish",
}


class ImageOrchestrator:
    """
    Orchestrates image generation with safety filtering and workflow selection.
    """

    def __init__(self, comfyui_url: str = None):
        self.client = ComfyUIClient(base_url=comfyui_url or COMFYUI_URL)

    def generate(
        self,
        prompt: str,
        user_id: int,
        user_permission: str = "adult",
        workflow_name: str = None,
        width: int = 512,
        height: int = 512,
        seed: int = None,
        negative_prompt: str = None,
    ) -> dict:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image
            user_id: The requesting user's ID (for logging and permissions)
            user_permission: "adult", "teen", or "child"
            workflow_name: Override workflow selection (default: auto-select)
            width: Image width in pixels
            height: Image height in pixels
            seed: Random seed (None = random)
            negative_prompt: What to avoid in the image

        Returns:
            dict with keys: success, images, prompt_id, error, filtered_prompt
        """
        # 1. Check ComfyUI availability
        if not self.client.is_alive():
            return self._error("ComfyUI server is not running")

        # 2. Safety Layer 1: prompt filtering
        filter_result = self.filter_prompt(prompt, user_permission)
        if filter_result["blocked"]:
            return self._error(f"Prompt blocked: {filter_result['reason']}")
        filtered_prompt = filter_result["filtered_prompt"]

        # 3. Select workflow
        wf_name = workflow_name or self._select_workflow(user_permission)
        wf_entry = get_workflow(wf_name)
        if wf_entry is None:
            return self._error(f"Unknown workflow: {wf_name}")

        wf_func, wf_desc, _ = wf_entry

        # 4. Build workflow with parameters
        neg = negative_prompt or "ugly, blurry, low quality, deformed"
        if user_permission != "adult":
            neg = f"{neg}, nsfw, nude, explicit"

        workflow = wf_func(
            prompt=filtered_prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            seed=seed,
        )

        # 5. Submit to ComfyUI
        result = self.client.generate(workflow)
        result["filtered_prompt"] = filtered_prompt
        result["workflow"] = wf_name
        result["user_id"] = user_id
        return result

    def filter_prompt(self, prompt: str, user_permission: str) -> dict:
        """
        Safety Layer 1: Pre-generation prompt filtering.

        For non-adult users, blocks prompts containing NSFW keywords.
        For all users, strips potentially dangerous injection patterns.

        Returns:
            {"blocked": bool, "reason": str|None, "filtered_prompt": str}
        """
        cleaned = prompt.strip()

        # Block empty prompts
        if not cleaned:
            return {"blocked": True, "reason": "Empty prompt", "filtered_prompt": ""}

        # For non-adult users, check NSFW keywords
        if user_permission != "adult":
            prompt_lower = cleaned.lower()
            found = [kw for kw in NSFW_KEYWORDS if kw in prompt_lower]
            if found:
                return {
                    "blocked": True,
                    "reason": f"Content not appropriate for {user_permission} users",
                    "filtered_prompt": cleaned,
                }

        # Strip potential ComfyUI workflow injection (e.g., JSON fragments)
        cleaned = re.sub(r'\{[^}]*"class_type"[^}]*\}', '', cleaned)

        return {"blocked": False, "reason": None, "filtered_prompt": cleaned}

    def _select_workflow(self, user_permission: str) -> str:
        """
        Auto-select the best workflow based on user permissions.
        For now, defaults to sd15_basic (lowest VRAM, most compatible).
        """
        # Future: check available VRAM, user preferences, etc.
        return "sd15_basic"

    def _error(self, message: str) -> dict:
        return {
            "success": False,
            "images": [],
            "prompt_id": None,
            "error": message,
            "filtered_prompt": None,
            "workflow": None,
            "user_id": None,
        }

    def is_available(self) -> bool:
        """Check if the image generation system is available."""
        return self.client.is_alive()

    def list_available_workflows(self) -> list:
        """List all workflow templates."""
        return list_workflows()
