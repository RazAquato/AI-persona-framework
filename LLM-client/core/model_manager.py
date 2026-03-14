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
Model Manager
-------------
Handles loading, unloading, and switching LLM models by managing the
llama-server process. Reads model definitions from model_configs.yaml.

The llama-server runs as a subprocess. Switching models means:
  1. Kill the current llama-server process (SIGTERM, then SIGKILL)
  2. Wait for the port to free up
  3. Launch a new llama-server with the requested model
  4. Wait for the /health endpoint to respond
"""

import glob
import os
import signal
import socket
import subprocess
import time
import yaml


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_CLIENT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
LLAMA_PATH = os.path.abspath(os.path.join(LLM_CLIENT_ROOT, "..", "llama.cpp"))
CONFIG_PATH = os.path.join(LLM_CLIENT_ROOT, "config", "model_configs.yaml")

# Inject CUDA libs from venv (same logic as load_LLM.py)
_venv_nvidia = os.path.expanduser(
    "~/venvs/AI-persona-framework-venv/lib/python3.12/site-packages/nvidia"
)
if os.path.isdir(_venv_nvidia):
    _cuda_dirs = set(
        os.path.dirname(p)
        for p in glob.glob(f"{_venv_nvidia}/**/*.so*", recursive=True)
    )
    _existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join(_cuda_dirs) + (":" + _existing if _existing else "")
    )


def load_model_configs() -> dict:
    """Load model configurations from YAML."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def list_available_models() -> list:
    """Return list of model dicts with key, name, vram_gb, ctx_size."""
    configs = load_model_configs()
    models = []
    for key, conf in configs.items():
        models.append({
            "key": key,
            "name": conf.get("name", key),
            "vram_gb": conf.get("vram_gb", 0),
            "ctx_size": conf.get("ctx_size", 0),
        })
    return models


def _find_llama_server_pids() -> list:
    """Find PIDs of running llama-server processes."""
    pids = []
    try:
        result = subprocess.run(
            ["pgrep", "-f", "llama-server"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return pids


def _port_in_use(port: int) -> bool:
    """Check if a port is currently bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_llama_server(timeout: float = 10.0) -> bool:
    """
    Kill all running llama-server processes.
    Sends SIGTERM first, then SIGKILL after timeout.
    Returns True if processes were found and killed.
    """
    pids = _find_llama_server_pids()
    if not pids:
        return False

    # SIGTERM first
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    # Wait for graceful shutdown
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = _find_llama_server_pids()
        if not remaining:
            return True
        time.sleep(0.5)

    # SIGKILL stragglers
    for pid in _find_llama_server_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    time.sleep(1)
    return True


def start_llama_server(model_key: str) -> dict:
    """
    Start llama-server with the specified model.

    Returns dict with:
        success: bool
        model_key: str
        error: str (if failed)
        pid: int (if successful)
    """
    configs = load_model_configs()
    conf = configs.get(model_key)
    if not conf:
        return {"success": False, "model_key": model_key,
                "error": f"Unknown model: {model_key}"}

    model_path = conf["path"]
    if not os.path.isfile(model_path):
        return {"success": False, "model_key": model_key,
                "error": f"Model file not found: {model_path}"}

    port = conf.get("port", 8080)

    # Wait for port to free up
    for _ in range(20):
        if not _port_in_use(port):
            break
        time.sleep(0.5)
    else:
        return {"success": False, "model_key": model_key,
                "error": f"Port {port} still in use after waiting"}

    llama_binary = os.path.join(LLAMA_PATH, "build", "bin", "llama-server")
    if not os.path.isfile(llama_binary):
        return {"success": False, "model_key": model_key,
                "error": f"llama-server binary not found: {llama_binary}"}

    cmd = [
        llama_binary,
        "-m", model_path,
        "--port", str(port),
        "--ctx-size", str(conf.get("ctx_size", 4096)),
        "--n-gpu-layers", str(conf.get("n_gpu_layers", 999)),
        "--main-gpu", str(conf.get("main_gpu", 0)),
        "--host", "0.0.0.0",
        "--no-warmup",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        return {"success": False, "model_key": model_key,
                "error": f"Failed to start process: {e}"}

    # Wait for server to become healthy
    import urllib.request
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + 60  # 60s timeout for model loading
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    return {"success": True, "model_key": model_key,
                            "pid": proc.pid}
        except Exception:
            pass

        # Check if process died
        if proc.poll() is not None:
            return {"success": False, "model_key": model_key,
                    "error": "llama-server exited during startup"}
        time.sleep(1)

    # Timeout — kill the failed process
    proc.kill()
    return {"success": False, "model_key": model_key,
            "error": "llama-server did not become healthy within 60 seconds"}


def switch_model(model_key: str) -> dict:
    """
    Switch to a different model. Kills current server, starts new one.

    Returns dict with:
        success: bool
        model_key: str
        error: str (if failed)
        killed_previous: bool
    """
    killed = kill_llama_server()

    result = start_llama_server(model_key)
    result["killed_previous"] = killed
    return result
