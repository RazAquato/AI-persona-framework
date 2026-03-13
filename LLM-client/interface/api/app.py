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

"""
Web API for AI Persona Chat
----------------------------
FastAPI app that routes user messages through the engine.
Handles /commands (like /image) directly via the tool registry,
and regular messages through the full conversation pipeline.

Usage:
    cd LLM-client/interface/api && uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import re

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_CLIENT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "memory-server"))
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "shared"))
TOOLS_PATH = os.path.abspath(os.path.join(SHARED_PATH, "tools"))

for p in [LLM_CLIENT_ROOT, MEMORY_PATH, SHARED_PATH, TOOLS_PATH]:
    if p not in sys.path:
        sys.path.append(p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional

from core.engine import run_conversation_turn
from core.router import is_tool_command, parse_tool_command
import tool_registry

app = FastAPI(title="AI Persona Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    user_id: int = 9999
    persona: str = "girlfriend"
    session_id: Optional[int] = None
    user_permission: str = "adult"


class ChatResponse(BaseModel):
    reply: str
    images: list = []
    session_id: Optional[int] = None
    persona_emotions: Optional[dict] = None
    emotion_description: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Handles both /commands and regular messages.
    """
    user_input = req.message.strip()

    # Direct tool commands bypass the LLM entirely
    if is_tool_command(user_input):
        command, prompt = parse_tool_command(user_input)
        tool_func = tool_registry.get_tool(command)

        if tool_func:
            try:
                result = tool_func(
                    prompt,
                    user_id=req.user_id,
                    user_permission=req.user_permission,
                )
                if isinstance(result, dict):
                    images = result.get("images", [])
                    if result.get("success") and images:
                        return ChatResponse(
                            reply="Image generated!",
                            images=images,
                        )
                    elif not result.get("success"):
                        return ChatResponse(
                            reply="Failed: {}".format(result.get("error", "unknown")),
                        )
                return ChatResponse(reply="Tool executed.")
            except Exception as e:
                return ChatResponse(reply="Tool error: {}".format(e))
        else:
            return ChatResponse(reply="Unknown command: {}".format(command))

    # Regular message: full conversation pipeline
    result = run_conversation_turn(
        user_id=req.user_id,
        user_input=user_input,
        personality_id=req.persona,
        session_id=req.session_id,
    )

    reply = result.get("assistant_reply", "")
    # Strip <think> blocks
    reply = re.sub(r'<think>.*?</think>\s*', '', reply, flags=re.DOTALL).strip()

    # Extract image paths from tool results
    images = []
    for tr in result.get("tool_results", []):
        if tr.success and tr.result and isinstance(tr.result, dict):
            images.extend(tr.result.get("images", []))

    return ChatResponse(
        reply=reply,
        images=images,
        session_id=result.get("session_id"),
        persona_emotions=result.get("persona_emotions"),
        emotion_description=result.get("emotion_description"),
    )


@app.get("/image/{filename}")
async def serve_image(filename: str):
    """Serve generated images from the ComfyUI output directory."""
    output_dir = os.path.abspath(
        os.path.join(BASE_DIR, "..", "..", "..", "comfyui", "output")
    )
    filepath = os.path.join(output_dir, filename)
    if os.path.isfile(filepath) and filepath.startswith(output_dir):
        return FileResponse(filepath, media_type="image/png")
    return {"error": "not found"}


@app.get("/tools")
async def list_tools():
    """List available tools."""
    return {"tools": tool_registry.describe_tools()}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    """Minimal chat UI."""
    return HTMLResponse(CHAT_HTML)


CHAT_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AI Persona Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; flex-direction: column; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 70%; padding: 12px 16px; border-radius: 16px; line-height: 1.5; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: #4a4a8a; }
  .msg.assistant { align-self: flex-start; background: #2a2a4a; }
  .msg img { max-width: 100%; border-radius: 8px; margin-top: 8px; display: block; }
  #input-area { display: flex; padding: 16px; gap: 8px; background: #16213e; }
  #input { flex: 1; padding: 12px; border: 1px solid #444; border-radius: 8px; background: #1a1a2e; color: #eee; font-size: 16px; }
  #input:focus { outline: none; border-color: #6a6aaa; }
  #send { padding: 12px 24px; background: #4a4a8a; border: none; border-radius: 8px; color: #eee; font-size: 16px; cursor: pointer; }
  #send:hover { background: #5a5a9a; }
  .typing { color: #888; font-style: italic; }
</style>
</head>
<body>
<div id="chat"></div>
<div id="input-area">
  <input id="input" placeholder="Type a message or /image prompt..." autofocus>
  <button id="send" onclick="sendMsg()">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
let sessionId = null;

input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); } });

async function sendMsg() {
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  addMsg('user', msg);
  const typing = addMsg('assistant', 'Thinking...', 'typing');
  try {
    const body = { message: msg, user_id: 9999, persona: 'girlfriend' };
    if (sessionId) body.session_id = sessionId;
    const resp = await fetch('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
    const data = await resp.json();
    sessionId = data.session_id || sessionId;
    typing.remove();
    let html = escHtml(data.reply);
    if (data.images && data.images.length > 0) {
      for (const img of data.images) {
        const fname = img.split('/').pop();
        html += '<img src="/image/' + fname + '" alt="generated image">';
      }
    }
    addMsgHtml('assistant', html);
  } catch(e) {
    typing.remove();
    addMsg('assistant', 'Error: ' + e.message);
  }
}

function addMsg(role, text, cls) {
  const div = document.createElement('div');
  div.className = 'msg ' + role + (cls ? ' ' + cls : '');
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addMsgHtml(role, html) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function escHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
</script>
</body>
</html>"""
