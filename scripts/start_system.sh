#!/usr/bin/env bash
# AI Persona Framework — System Startup Script
# Starts: venv, LLM server (qwen9b), ComfyUI, FastAPI web UI
#
# Usage:
#   ./scripts/start_system.sh              # start everything
#   ./scripts/start_system.sh --no-comfyui # skip ComfyUI
#   ./scripts/start_system.sh --no-llm     # skip LLM server
#   ./scripts/start_system.sh --no-api     # skip FastAPI
#
# Logs go to /tmp/ai-persona-*.log

set -euo pipefail

PROJECT_DIR="/home/kenneth/AI-persona-framework"
VENV="/home/kenneth/venvs/AI-persona-framework-venv"
LLM_MODEL="${LLM_MODEL:-qwen9b}"

# Parse flags
START_LLM=true
START_COMFYUI=true
START_API=true
for arg in "$@"; do
    case "$arg" in
        --no-llm)     START_LLM=false ;;
        --no-comfyui) START_COMFYUI=false ;;
        --no-api)     START_API=false ;;
    esac
done

# Activate venv
source "$VENV/bin/activate"
echo "[startup] Venv activated: $VENV"

# --- LLM Server (llama-server via load_LLM.py) ---
if $START_LLM; then
    if curl -sf http://10.0.20.200:8080/health > /dev/null 2>&1; then
        echo "[startup] LLM server already running on :8080"
    else
        echo "[startup] Starting LLM server (model: $LLM_MODEL)..."
        nohup python3 "$PROJECT_DIR/LLM-client/load_LLM.py" --model "$LLM_MODEL" \
            > /tmp/ai-persona-llm.log 2>&1 &
        LLM_PID=$!
        echo "[startup] LLM server PID: $LLM_PID (log: /tmp/ai-persona-llm.log)"
        # Wait for health
        for i in $(seq 1 60); do
            if curl -sf http://10.0.20.200:8080/health > /dev/null 2>&1; then
                echo "[startup] LLM server healthy after ${i}s"
                break
            fi
            sleep 1
        done
        if ! curl -sf http://10.0.20.200:8080/health > /dev/null 2>&1; then
            echo "[startup] WARNING: LLM server not healthy after 60s — check /tmp/ai-persona-llm.log"
        fi
    fi
fi

# --- ComfyUI (image generation server) ---
if $START_COMFYUI; then
    if curl -sf http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "[startup] ComfyUI already running on :8188"
    else
        echo "[startup] Starting ComfyUI..."
        cd "$PROJECT_DIR/comfyui"
        nohup python3 main.py --listen 0.0.0.0 --port 8188 \
            > /tmp/ai-persona-comfyui.log 2>&1 &
        COMFY_PID=$!
        echo "[startup] ComfyUI PID: $COMFY_PID (log: /tmp/ai-persona-comfyui.log)"
        cd "$PROJECT_DIR"
        # Wait for health
        for i in $(seq 1 30); do
            if curl -sf http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
                echo "[startup] ComfyUI healthy after ${i}s"
                break
            fi
            sleep 1
        done
        if ! curl -sf http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
            echo "[startup] WARNING: ComfyUI not healthy after 30s — check /tmp/ai-persona-comfyui.log"
        fi
    fi
fi

# --- FastAPI Web UI ---
if $START_API; then
    # Check if already running
    if curl -sf http://10.0.20.200:8000/docs > /dev/null 2>&1; then
        echo "[startup] FastAPI already running on :8000"
    else
        echo "[startup] Starting FastAPI web UI..."
        cd "$PROJECT_DIR/LLM-client/interface/api"
        nohup uvicorn app:app --host 0.0.0.0 --port 8000 \
            > /tmp/ai-persona-api.log 2>&1 &
        API_PID=$!
        echo "[startup] FastAPI PID: $API_PID (log: /tmp/ai-persona-api.log)"
        cd "$PROJECT_DIR"
        sleep 2
        if curl -sf http://10.0.20.200:8000/docs > /dev/null 2>&1; then
            echo "[startup] FastAPI healthy"
        else
            echo "[startup] WARNING: FastAPI not responding — check /tmp/ai-persona-api.log"
        fi
    fi
fi

echo ""
echo "[startup] System ready:"
$START_LLM     && echo "  LLM:     http://10.0.20.200:8080  (model: $LLM_MODEL)"
$START_COMFYUI && echo "  ComfyUI: http://10.0.20.200:8188"
$START_API     && echo "  Web UI:  http://10.0.20.200:8000"
echo "  Logs:    /tmp/ai-persona-*.log"
