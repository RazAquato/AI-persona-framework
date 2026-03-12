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
ComfyUI API Client
------------------
Low-level client for ComfyUI's HTTP + WebSocket API.

Usage:
    client = ComfyUIClient("http://10.0.20.200:8188")
    result = client.generate(workflow_json)
    # result = {"images": ["/path/to/output.png"], "prompt_id": "..."}

The client:
1. Submits a workflow via POST /api/prompt
2. Listens on WebSocket for execution progress
3. Retrieves output images when done
"""

import json
import uuid
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


COMFYUI_OUTPUT_DIR = Path("/home/kenneth/AI-persona-framework/comfyui/output")


class ComfyUIClient:
    """Client for ComfyUI's REST + WebSocket API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    def is_alive(self) -> bool:
        """Check if ComfyUI server is running."""
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, ConnectionError, OSError):
            return False

    def get_checkpoints(self) -> list:
        """List available checkpoint models."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/object_info/CheckpointLoaderSimple")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError):
            return []

    def queue_prompt(self, workflow: dict) -> str:
        """
        Submit a workflow to ComfyUI for execution.
        Returns the prompt_id for tracking.
        """
        payload = json.dumps({
            "prompt": workflow,
            "client_id": self.client_id,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
            return result["prompt_id"]

    def poll_until_done(self, prompt_id: str, poll_interval: float = 1.0) -> bool:
        """
        Poll the queue until the given prompt_id has finished executing.
        Returns True if completed, False if timed out.
        """
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                req = urllib.request.Request(f"{self.base_url}/api/history/{prompt_id}")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    history = json.loads(resp.read())
                    if prompt_id in history:
                        return True
            except (urllib.error.URLError, json.JSONDecodeError):
                pass
            time.sleep(poll_interval)
        return False

    def get_output_images(self, prompt_id: str) -> list:
        """
        Retrieve output image paths for a completed prompt.
        Returns list of absolute file paths.
        """
        try:
            req = urllib.request.Request(f"{self.base_url}/api/history/{prompt_id}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                history = json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError):
            return []

        if prompt_id not in history:
            return []

        images = []
        outputs = history[prompt_id].get("outputs", {})
        for node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                filename = img.get("filename", "")
                subfolder = img.get("subfolder", "")
                if filename:
                    path = COMFYUI_OUTPUT_DIR / subfolder / filename
                    images.append(str(path))
        return images

    def generate(self, workflow: dict) -> dict:
        """
        Submit workflow, wait for completion, return output image paths.

        Returns:
            {"prompt_id": str, "images": [str], "success": bool, "error": str|None}
        """
        try:
            prompt_id = self.queue_prompt(workflow)
        except Exception as e:
            return {"prompt_id": None, "images": [], "success": False, "error": f"Queue failed: {e}"}

        completed = self.poll_until_done(prompt_id)
        if not completed:
            return {"prompt_id": prompt_id, "images": [], "success": False, "error": "Timed out"}

        images = self.get_output_images(prompt_id)
        return {"prompt_id": prompt_id, "images": images, "success": True, "error": None}
