# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
# GPLv3 License - See <https://www.gnu.org/licenses/>

import unittest
from unittest.mock import patch, MagicMock
import os

FAKE_CONFIGS = {
    "qwen9b": {
        "name": "Qwen 3.5 9B (Uncensored Q8)",
        "path": "/modeller/fake/qwen.gguf",
        "port": 8080, "ctx_size": 8192,
        "n_gpu_layers": 999, "main_gpu": 0, "vram_gb": 9.5,
    },
    "tinyllama": {
        "name": "TinyLlama 1.1B (Q4_K_M)",
        "path": "/modeller/fake/tiny.gguf",
        "port": 8080, "ctx_size": 2048,
        "n_gpu_layers": 999, "main_gpu": 0, "vram_gb": 1,
    },
}


class TestListModels(unittest.TestCase):

    @patch("core.model_manager.load_model_configs", return_value=FAKE_CONFIGS)
    def test_list_available_models(self, mock_configs):
        from core.model_manager import list_available_models
        models = list_available_models()
        self.assertEqual(len(models), 2)
        keys = [m["key"] for m in models]
        self.assertIn("qwen9b", keys)
        self.assertIn("tinyllama", keys)

    @patch("core.model_manager.load_model_configs", return_value=FAKE_CONFIGS)
    def test_models_have_required_fields(self, mock_configs):
        from core.model_manager import list_available_models
        for m in list_available_models():
            self.assertIn("key", m)
            self.assertIn("name", m)
            self.assertIn("vram_gb", m)
            self.assertIn("ctx_size", m)

    @patch("core.model_manager.load_model_configs", return_value=FAKE_CONFIGS)
    def test_vram_values(self, mock_configs):
        from core.model_manager import list_available_models
        models = {m["key"]: m for m in list_available_models()}
        self.assertEqual(models["qwen9b"]["vram_gb"], 9.5)
        self.assertEqual(models["tinyllama"]["vram_gb"], 1)


class TestKillServer(unittest.TestCase):

    @patch("core.model_manager._find_llama_server_pids", return_value=[])
    def test_kill_no_process(self, mock_pids):
        from core.model_manager import kill_llama_server
        result = kill_llama_server(timeout=1)
        self.assertFalse(result)

    @patch("core.model_manager.time.sleep")
    @patch("core.model_manager.os.kill")
    @patch("core.model_manager._find_llama_server_pids")
    def test_kill_sends_sigterm(self, mock_pids, mock_kill, mock_sleep):
        import signal
        mock_pids.side_effect = [[1234], []]  # first call finds pid, second finds none
        from core.model_manager import kill_llama_server
        result = kill_llama_server(timeout=2)
        self.assertTrue(result)
        mock_kill.assert_called_with(1234, signal.SIGTERM)


class TestStartServer(unittest.TestCase):

    @patch("core.model_manager.load_model_configs", return_value=FAKE_CONFIGS)
    def test_unknown_model_returns_error(self, mock_configs):
        from core.model_manager import start_llama_server
        result = start_llama_server("nonexistent")
        self.assertFalse(result["success"])
        self.assertIn("Unknown model", result["error"])

    @patch("core.model_manager.load_model_configs", return_value=FAKE_CONFIGS)
    @patch("os.path.isfile", return_value=False)
    def test_missing_model_file_returns_error(self, mock_isfile, mock_configs):
        from core.model_manager import start_llama_server
        result = start_llama_server("qwen9b")
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])


class TestSwitchModel(unittest.TestCase):

    @patch("core.model_manager.start_llama_server")
    @patch("core.model_manager.kill_llama_server", return_value=True)
    def test_switch_kills_then_starts(self, mock_kill, mock_start):
        mock_start.return_value = {"success": True, "model_key": "tinyllama", "pid": 5678}
        from core.model_manager import switch_model
        result = switch_model("tinyllama")
        self.assertTrue(result["success"])
        self.assertTrue(result["killed_previous"])
        mock_kill.assert_called_once()
        mock_start.assert_called_once_with("tinyllama")

    @patch("core.model_manager.start_llama_server")
    @patch("core.model_manager.kill_llama_server", return_value=False)
    def test_switch_no_previous_server(self, mock_kill, mock_start):
        mock_start.return_value = {"success": True, "model_key": "qwen9b", "pid": 9999}
        from core.model_manager import switch_model
        result = switch_model("qwen9b")
        self.assertTrue(result["success"])
        self.assertFalse(result["killed_previous"])


if __name__ == "__main__":
    unittest.main()
