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

import unittest
from unittest.mock import patch, MagicMock
from tools.image_orchestrator import ImageOrchestrator, NSFW_KEYWORDS
from tools.comfyui_workflows import sd15_basic, sdxl_turbo, list_workflows, get_workflow


class TestPromptFilter(unittest.TestCase):
    """Test the safety prompt filtering layer."""

    def setUp(self):
        self.orch = ImageOrchestrator()

    def test_adult_allows_nsfw(self):
        result = self.orch.filter_prompt("a nude painting", "adult")
        self.assertFalse(result["blocked"])

    def test_teen_blocks_nsfw(self):
        result = self.orch.filter_prompt("a nude painting", "teen")
        self.assertTrue(result["blocked"])
        self.assertIn("not appropriate", result["reason"])

    def test_child_blocks_nsfw(self):
        result = self.orch.filter_prompt("erotic scene", "child")
        self.assertTrue(result["blocked"])

    def test_child_allows_safe_prompt(self):
        result = self.orch.filter_prompt("a cute dog in a park", "child")
        self.assertFalse(result["blocked"])

    def test_empty_prompt_blocked(self):
        result = self.orch.filter_prompt("", "adult")
        self.assertTrue(result["blocked"])
        self.assertIn("Empty", result["reason"])

    def test_whitespace_only_blocked(self):
        result = self.orch.filter_prompt("   ", "adult")
        self.assertTrue(result["blocked"])

    def test_injection_stripped(self):
        malicious = 'a cat {"class_type": "evil"} in a garden'
        result = self.orch.filter_prompt(malicious, "adult")
        self.assertFalse(result["blocked"])
        self.assertNotIn("class_type", result["filtered_prompt"])

    def test_all_nsfw_keywords_blocked_for_teen(self):
        for kw in NSFW_KEYWORDS:
            result = self.orch.filter_prompt(f"a {kw} image", "teen")
            self.assertTrue(result["blocked"], f"Keyword '{kw}' was not blocked for teen")


class TestWorkflowSelection(unittest.TestCase):
    """Test workflow auto-selection and registry."""

    def setUp(self):
        self.orch = ImageOrchestrator()

    def test_default_workflow_selection(self):
        wf = self.orch._select_workflow("adult")
        self.assertEqual(wf, "sd15_basic")

    def test_list_workflows_not_empty(self):
        wfs = list_workflows()
        self.assertGreater(len(wfs), 0)
        for wf in wfs:
            self.assertIn("name", wf)
            self.assertIn("description", wf)
            self.assertIn("min_vram_gb", wf)

    def test_get_known_workflow(self):
        entry = get_workflow("sd15_basic")
        self.assertIsNotNone(entry)
        func, desc, vram = entry
        self.assertTrue(callable(func))

    def test_get_unknown_workflow(self):
        self.assertIsNone(get_workflow("nonexistent"))


class TestWorkflowTemplates(unittest.TestCase):
    """Test that workflow templates produce valid ComfyUI JSON."""

    def test_sd15_basic_structure(self):
        wf = sd15_basic("a cat")
        self.assertIn("3", wf)  # KSampler
        self.assertIn("4", wf)  # CheckpointLoader
        self.assertIn("9", wf)  # SaveImage
        self.assertEqual(wf["4"]["class_type"], "CheckpointLoaderSimple")
        self.assertEqual(wf["6"]["inputs"]["text"], "a cat")

    def test_sdxl_turbo_structure(self):
        wf = sdxl_turbo("a dog")
        self.assertEqual(wf["3"]["inputs"]["steps"], 4)
        self.assertEqual(wf["3"]["inputs"]["cfg"], 1.0)

    def test_sd15_custom_params(self):
        wf = sd15_basic("test", width=768, height=768, steps=50, cfg=10.0, seed=42)
        self.assertEqual(wf["5"]["inputs"]["width"], 768)
        self.assertEqual(wf["5"]["inputs"]["height"], 768)
        self.assertEqual(wf["3"]["inputs"]["steps"], 50)
        self.assertEqual(wf["3"]["inputs"]["cfg"], 10.0)
        self.assertEqual(wf["3"]["inputs"]["seed"], 42)

    def test_negative_prompt_set(self):
        wf = sd15_basic("cat", negative_prompt="dog, ugly")
        self.assertEqual(wf["7"]["inputs"]["text"], "dog, ugly")


class TestOrchestratorGenerate(unittest.TestCase):
    """Test the full generate flow with mocked ComfyUI."""

    def setUp(self):
        self.orch = ImageOrchestrator()

    @patch.object(ImageOrchestrator, 'is_available', return_value=False)
    def test_generate_fails_when_comfyui_down(self, _):
        # Patch client.is_alive directly
        self.orch.client.is_alive = MagicMock(return_value=False)
        result = self.orch.generate("a cat", user_id=9999)
        self.assertFalse(result["success"])
        self.assertIn("not running", result["error"])

    def test_generate_blocks_nsfw_for_child(self):
        self.orch.client.is_alive = MagicMock(return_value=True)
        result = self.orch.generate("a nude image", user_id=1, user_permission="child")
        self.assertFalse(result["success"])
        self.assertIn("blocked", result["error"].lower())

    @patch("tools.image_orchestrator.ComfyUIClient")
    def test_generate_success_flow(self, MockClient):
        mock_client = MagicMock()
        mock_client.is_alive.return_value = True
        mock_client.generate.return_value = {
            "prompt_id": "abc123",
            "images": ["/path/to/image.png"],
            "success": True,
            "error": None,
        }
        orch = ImageOrchestrator()
        orch.client = mock_client

        result = orch.generate("a cute cat", user_id=9999)
        self.assertTrue(result["success"])
        self.assertEqual(result["images"], ["/path/to/image.png"])
        self.assertEqual(result["filtered_prompt"], "a cute cat")
        self.assertEqual(result["workflow"], "sd15_basic")

    def test_generate_unknown_workflow_fails(self):
        self.orch.client.is_alive = MagicMock(return_value=True)
        result = self.orch.generate("a cat", user_id=9999, workflow_name="fake_workflow")
        self.assertFalse(result["success"])
        self.assertIn("Unknown workflow", result["error"])

    def test_generate_adds_nsfw_negative_for_teen(self):
        """Teen users should get nsfw in negative prompt automatically."""
        mock_client = MagicMock()
        mock_client.is_alive.return_value = True
        mock_client.generate.return_value = {
            "prompt_id": "x", "images": [], "success": True, "error": None,
        }
        self.orch.client = mock_client
        self.orch.generate("a flower", user_id=1, user_permission="teen")

        # Inspect the workflow passed to client.generate
        call_args = mock_client.generate.call_args[0][0]
        neg_text = call_args["7"]["inputs"]["text"]
        self.assertIn("nsfw", neg_text)
        self.assertIn("nude", neg_text)


class TestComfyUIClient(unittest.TestCase):
    """Test ComfyUI client (mocked — no real server needed)."""

    def test_is_alive_when_down(self):
        from tools.comfyui_client import ComfyUIClient
        client = ComfyUIClient("http://127.0.0.1:99999")
        self.assertFalse(client.is_alive())

    def test_generate_returns_error_on_failure(self):
        from tools.comfyui_client import ComfyUIClient
        client = ComfyUIClient("http://127.0.0.1:99999")
        result = client.generate({"fake": "workflow"})
        self.assertFalse(result["success"])
        self.assertIn("Queue failed", result["error"])


if __name__ == "__main__":
    unittest.main()
