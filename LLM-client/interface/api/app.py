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
FastAPI app with cookie-based authentication.
Routes user messages through the engine pipeline.

Usage:
    cd LLM-client/interface/api && uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import re
import json
import requests as http_requests

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_CLIENT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "memory-server"))
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "shared"))
TOOLS_PATH = os.path.abspath(os.path.join(SHARED_PATH, "tools"))

for p in [LLM_CLIENT_ROOT, MEMORY_PATH, SHARED_PATH, TOOLS_PATH]:
    if p not in sys.path:
        sys.path.append(p)

from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional

from core.engine import run_conversation_turn
from core.router import is_tool_command, parse_tool_command
from core.auth import hash_password, verify_password, create_auth_cookie, verify_auth_cookie
from memory.chat_store import list_sessions, start_chat_session, get_chat_messages
from memory.user_store import (
    create_user, get_user_by_name, get_user_by_id, get_session_owner,
)
from memory.persona_store import (
    list_personas, get_persona, get_persona_by_slug,
    create_persona as db_create_persona, update_persona as db_update_persona,
    delete_persona as db_delete_persona, seed_default_personas,
)
import tool_registry
from core.model_manager import list_available_models, switch_model

# LLM server URL (for model info queries)
from dotenv import load_dotenv
_llm_env = os.path.join(LLM_CLIENT_ROOT, "config", ".env")
load_dotenv(dotenv_path=_llm_env)
LLM_SERVER = os.getenv("LLM_SERVER", "http://10.0.20.200:8080")

COOKIE_NAME = "session"
COOKIE_MAX_AGE = 30 * 24 * 3600  # 30 days

app = FastAPI(title="AI Persona Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth dependency ---

def get_current_user(request: Request) -> int:
    """Extract authenticated user_id from cookie. Raises 401 if invalid."""
    cookie = request.cookies.get(COOKIE_NAME)
    user_id = verify_auth_cookie(cookie)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_id


# --- Auth endpoints ---

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
async def api_login(req: LoginRequest, response: Response):
    """Verify credentials and set session cookie."""
    row = get_user_by_name(req.username)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    user_id, name, pw_hash = row
    if not pw_hash or not verify_password(req.password, pw_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    cookie = create_auth_cookie(user_id)
    response.set_cookie(
        COOKIE_NAME, cookie, max_age=COOKIE_MAX_AGE,
        httponly=True, samesite="lax",
    )
    return {"user_id": user_id, "username": name}


@app.post("/api/register")
async def api_register(req: RegisterRequest, response: Response):
    """Create a new user account and set session cookie."""
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    if len(req.username) < 2:
        raise HTTPException(status_code=400, detail="Username must be at least 2 characters")
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")
    existing = get_user_by_name(req.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")
    pw_hash = hash_password(req.password)
    user_id = create_user(req.username, pw_hash)
    seed_default_personas(user_id)
    cookie = create_auth_cookie(user_id)
    response.set_cookie(
        COOKIE_NAME, cookie, max_age=COOKIE_MAX_AGE,
        httponly=True, samesite="lax",
    )
    return {"user_id": user_id, "username": req.username}


@app.post("/api/logout")
async def api_logout(response: Response):
    """Clear session cookie."""
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@app.get("/api/me")
async def api_me(user_id: int = Depends(get_current_user)):
    """Return the currently authenticated user."""
    row = get_user_by_id(user_id)
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    return {"user_id": row[0], "username": row[1]}


# --- Chat endpoints (all require auth) ---

class ChatRequest(BaseModel):
    message: str
    persona_id: int
    session_id: Optional[int] = None
    nsfw_mode: bool = False
    incognito: bool = False


class ChatResponse(BaseModel):
    reply: str
    images: list = []
    session_id: Optional[int] = None
    persona_emotions: Optional[dict] = None
    emotion_description: Optional[str] = None
    incognito: bool = False
    nsfw_mode: bool = False


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user_id: int = Depends(get_current_user)):
    """Main chat endpoint. Handles both /commands and regular messages."""
    user_input = req.message.strip()

    # Verify persona belongs to user
    persona = get_persona(req.persona_id)
    if not persona or persona["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Persona does not belong to you")

    # Verify session ownership if resuming
    if req.session_id:
        owner = get_session_owner(req.session_id)
        if owner != user_id:
            raise HTTPException(status_code=403, detail="Session does not belong to you")

    # Direct tool commands bypass the LLM entirely
    if is_tool_command(user_input):
        command, prompt = parse_tool_command(user_input)
        tool_func = tool_registry.get_tool(command)

        if tool_func:
            try:
                user_perm = "adult" if req.nsfw_mode else "adult"
                result = tool_func(
                    prompt,
                    user_id=user_id,
                    user_permission=user_perm,
                    persona_id=req.persona_id,
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
        user_id=user_id,
        user_input=user_input,
        persona_id=req.persona_id,
        session_id=req.session_id,
        nsfw_mode=req.nsfw_mode,
        incognito=req.incognito,
    )

    reply = result.get("assistant_reply", "")
    reply = re.sub(r'<think>.*?</think>\s*', '', reply, flags=re.DOTALL).strip()

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
        incognito=result.get("incognito", False),
        nsfw_mode=result.get("nsfw_mode", False),
    )


@app.get("/image/{filepath:path}")
async def serve_image(filepath: str):
    """Serve generated images from the ComfyUI output directory."""
    output_dir = os.path.abspath(
        os.path.join(BASE_DIR, "..", "..", "..", "comfyui", "output")
    )
    full_path = os.path.abspath(os.path.join(output_dir, filepath))
    if os.path.isfile(full_path) and full_path.startswith(output_dir):
        return FileResponse(full_path, media_type="image/png")
    return {"error": "not found"}


@app.get("/tools")
async def list_tools(user_id: int = Depends(get_current_user)):
    """List available tools."""
    return {"tools": tool_registry.describe_tools()}


@app.get("/model")
async def model_info():
    """Get the currently loaded LLM model name."""
    try:
        resp = http_requests.get(LLM_SERVER.rstrip("/") + "/v1/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if models:
            model_id = models[0].get("id", "unknown")
            name = model_id.replace(".gguf", "")
            params = models[0].get("meta", {}).get("n_params", 0)
            return {"model": name, "params": params}
        return {"model": "unknown", "params": 0}
    except Exception:
        return {"model": "offline", "params": 0}


@app.get("/models")
async def models_list(user_id: int = Depends(get_current_user)):
    """List all available models with their VRAM requirements."""
    return {"models": list_available_models()}


class SwitchModelRequest(BaseModel):
    model_key: str


@app.post("/model/switch")
async def model_switch(req: SwitchModelRequest, user_id: int = Depends(get_current_user)):
    """Kill the current llama-server and start a new one with the requested model."""
    result = switch_model(req.model_key)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Switch failed"))
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/personas")
async def personas_list(user_id: int = Depends(get_current_user)):
    """List personas for the authenticated user."""
    rows = list_personas(user_id)
    return {"personas": rows}


class CreatePersonaRequest(BaseModel):
    slug: str
    name: str
    description: str = ""
    system_prompt: str = ""
    nsfw_capable: bool = False
    nsfw_prompt_addon: str = None
    memory_scope: Optional[dict] = None


@app.post("/personas")
async def personas_create(req: CreatePersonaRequest, user_id: int = Depends(get_current_user)):
    """Create a new persona for the authenticated user."""
    if not req.slug or not req.name:
        raise HTTPException(status_code=400, detail="slug and name are required")
    existing = get_persona_by_slug(user_id, req.slug)
    if existing:
        raise HTTPException(status_code=409, detail="Slug already in use")
    pid = db_create_persona(
        user_id=user_id, slug=req.slug, name=req.name,
        description=req.description, system_prompt=req.system_prompt,
        nsfw_capable=req.nsfw_capable, nsfw_prompt_addon=req.nsfw_prompt_addon,
        memory_scope=req.memory_scope,
    )
    return {"persona_id": pid, "slug": req.slug, "name": req.name}


@app.put("/personas/{persona_id}")
async def personas_update(persona_id: int, req: CreatePersonaRequest,
                          user_id: int = Depends(get_current_user)):
    """Update a persona (must belong to the authenticated user)."""
    persona = get_persona(persona_id)
    if not persona or persona["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Persona does not belong to you")
    db_update_persona(
        persona_id, slug=req.slug, name=req.name,
        description=req.description, system_prompt=req.system_prompt,
        nsfw_capable=req.nsfw_capable, nsfw_prompt_addon=req.nsfw_prompt_addon,
        memory_scope=req.memory_scope,
    )
    return {"ok": True}


@app.delete("/personas/{persona_id}")
async def personas_delete(persona_id: int, user_id: int = Depends(get_current_user)):
    """Delete a persona (must belong to the authenticated user)."""
    persona = get_persona(persona_id)
    if not persona or persona["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Persona does not belong to you")
    db_delete_persona(persona_id)
    return {"ok": True}


VALID_DOMAINS = {"family", "physical", "hobbies", "work", "emotional", "memories", "other"}


class DomainAccessRequest(BaseModel):
    domain_access: list


@app.get("/personas/{persona_id}/domains")
async def personas_get_domains(persona_id: int, user_id: int = Depends(get_current_user)):
    """Get a persona's domain access list."""
    persona = get_persona(persona_id)
    if not persona or persona["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Persona does not belong to you")
    return {"persona_id": persona_id, "domain_access": persona.get("domain_access", [])}


@app.put("/personas/{persona_id}/domains")
async def personas_update_domains(persona_id: int, req: DomainAccessRequest,
                                  user_id: int = Depends(get_current_user)):
    """Update a persona's domain access list."""
    persona = get_persona(persona_id)
    if not persona or persona["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Persona does not belong to you")
    # Validate domains
    invalid = set(req.domain_access) - VALID_DOMAINS
    if invalid:
        raise HTTPException(status_code=400,
                            detail=f"Invalid domains: {', '.join(sorted(invalid))}")
    db_update_persona(persona_id, domain_access=req.domain_access)
    return {"ok": True, "domain_access": req.domain_access}


@app.get("/sessions")
async def sessions(user_id: int = Depends(get_current_user)):
    """List sessions for the authenticated user, grouped by persona."""
    rows = list_sessions(user_id, limit=100)
    grouped = {}
    for s in rows:
        key = s.get("persona_slug") or str(s.get("persona_id") or "unknown")
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(s)
    return {"sessions": grouped}


class NewSessionRequest(BaseModel):
    persona_id: int
    nsfw_mode: bool = False
    incognito: bool = False


@app.post("/sessions/new")
async def new_session(req: NewSessionRequest, user_id: int = Depends(get_current_user)):
    """Create a new chat session for the authenticated user."""
    persona = get_persona(req.persona_id)
    if not persona or persona["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Persona does not belong to you")
    sid = start_chat_session(user_id, req.persona_id,
                             incognito=req.incognito, nsfw_mode=req.nsfw_mode)
    return {"session_id": sid, "persona_id": req.persona_id,
            "persona_name": persona["name"],
            "incognito": req.incognito, "nsfw_mode": req.nsfw_mode}


@app.get("/sessions/{session_id}/messages")
async def session_messages(session_id: int, user_id: int = Depends(get_current_user)):
    """Get all messages for a session (with ownership check)."""
    owner = get_session_owner(session_id)
    if owner != user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to you")
    rows = get_chat_messages(session_id)
    messages = []
    for r in rows:
        messages.append({
            "id": r[0],
            "role": r[1],
            "content": r[2],
        })
    return {"messages": messages}


# --- Pages ---

@app.get("/login")
async def login_page(request: Request):
    """Login/register page. Redirect to / if already authenticated."""
    cookie = request.cookies.get(COOKIE_NAME)
    if verify_auth_cookie(cookie):
        return RedirectResponse("/", status_code=302)
    return HTMLResponse(LOGIN_HTML)


@app.get("/")
async def index(request: Request):
    """Chat UI — redirect to /login if not authenticated."""
    cookie = request.cookies.get(COOKIE_NAME)
    if not verify_auth_cookie(cookie):
        return RedirectResponse("/login", status_code=302)
    return HTMLResponse(CHAT_HTML)


# ----- Login / Register HTML -----

LOGIN_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Login - AI Persona Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; align-items: center; justify-content: center; }
  .card { background: #16213e; padding: 40px; border-radius: 12px; width: 360px; border: 1px solid #2a2a4a; }
  .card h1 { font-size: 22px; margin-bottom: 8px; color: #ccd; }
  .card p.sub { font-size: 13px; color: #667; margin-bottom: 24px; }
  .tabs { display: flex; gap: 0; margin-bottom: 24px; }
  .tab { flex: 1; padding: 10px; text-align: center; cursor: pointer; border-bottom: 2px solid transparent; color: #667; font-size: 14px; transition: all 0.2s; }
  .tab.active { color: #ccd; border-bottom-color: #6a6aaa; }
  .tab:hover { color: #aab; }
  .field { margin-bottom: 16px; }
  .field label { display: block; font-size: 13px; color: #aab; margin-bottom: 6px; }
  .field input { width: 100%; padding: 10px 12px; background: #1a1a2e; color: #eee; border: 1px solid #444; border-radius: 6px; font-size: 14px; }
  .field input:focus { outline: none; border-color: #6a6aaa; }
  .btn { width: 100%; padding: 12px; background: #4a4a8a; border: none; border-radius: 6px; color: #eee; font-size: 15px; cursor: pointer; margin-top: 8px; }
  .btn:hover { background: #5a5a9a; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .error { color: #e77; font-size: 13px; margin-top: 12px; min-height: 20px; }
</style>
</head>
<body>
<div class="card">
  <h1>AI Persona Chat</h1>
  <p class="sub">Sign in or create an account</p>
  <div class="tabs">
    <div class="tab active" id="tab-login" onclick="switchTab('login')">Login</div>
    <div class="tab" id="tab-register" onclick="switchTab('register')">Register</div>
  </div>
  <form id="auth-form" onsubmit="submitForm(event)">
    <div class="field">
      <label for="username">Username</label>
      <input id="username" name="username" type="text" autocomplete="username" required autofocus>
    </div>
    <div class="field">
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
    </div>
    <button class="btn" type="submit" id="submit-btn">Login</button>
    <div class="error" id="error"></div>
  </form>
</div>
<script>
let mode = 'login';
function switchTab(m) {
  mode = m;
  document.getElementById('tab-login').className = 'tab' + (m === 'login' ? ' active' : '');
  document.getElementById('tab-register').className = 'tab' + (m === 'register' ? ' active' : '');
  document.getElementById('submit-btn').textContent = m === 'login' ? 'Login' : 'Create Account';
  document.getElementById('password').autocomplete = m === 'login' ? 'current-password' : 'new-password';
  document.getElementById('error').textContent = '';
}
async function submitForm(e) {
  e.preventDefault();
  const btn = document.getElementById('submit-btn');
  const err = document.getElementById('error');
  err.textContent = '';
  btn.disabled = true;
  const username = document.getElementById('username').value.trim();
  const password = document.getElementById('password').value;
  try {
    const url = mode === 'login' ? '/api/login' : '/api/register';
    const resp = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password})
    });
    if (resp.ok) {
      window.location.href = '/';
    } else {
      const data = await resp.json();
      err.textContent = data.detail || 'Something went wrong';
    }
  } catch(e) {
    err.textContent = 'Connection error';
  }
  btn.disabled = false;
}
</script>
</body>
</html>"""


# ----- Chat HTML -----

CHAT_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AI Persona Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; }

  /* Sidebar */
  #sidebar { width: 280px; background: #16213e; display: flex; flex-direction: column; border-right: 1px solid #2a2a4a; flex-shrink: 0; }
  #sidebar-header { padding: 16px; border-bottom: 1px solid #2a2a4a; }
  #sidebar-header h2 { font-size: 16px; margin-bottom: 12px; color: #aab; }
  #user-bar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; padding: 8px 0; border-bottom: 1px solid #2a2a4a; }
  #user-bar .username { font-size: 13px; color: #aab; }
  #user-bar .logout-btn { font-size: 12px; color: #667; cursor: pointer; background: none; border: 1px solid #444; border-radius: 4px; padding: 4px 8px; }
  #user-bar .logout-btn:hover { color: #aab; border-color: #666; }
  #persona-select { width: 100%; padding: 8px 12px; background: #1a1a2e; color: #eee; border: 1px solid #444; border-radius: 6px; font-size: 14px; cursor: pointer; }
  #persona-select:focus { outline: none; border-color: #6a6aaa; }
  #new-chat-btn { width: 100%; margin-top: 10px; padding: 10px; background: #4a4a8a; border: none; border-radius: 6px; color: #eee; font-size: 14px; cursor: pointer; }
  #new-chat-btn:hover { background: #5a5a9a; }
  .toggle-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 13px; color: #aab; }
  .toggle-row label { cursor: pointer; user-select: none; }
  .toggle-row input[type="checkbox"] { accent-color: #6a6aaa; cursor: pointer; }
  .toggle-row.hidden { display: none; }
  #domain-config { margin-top: 10px; font-size: 13px; color: #aab; }
  #domain-config summary { cursor: pointer; user-select: none; padding: 4px 0; }
  #domain-checkboxes { padding: 6px 0 4px 4px; }
  #domain-checkboxes label { display: block; padding: 2px 0; cursor: pointer; user-select: none; }
  #domain-checkboxes input { accent-color: #6a6aaa; cursor: pointer; margin-right: 6px; }
  #save-domains-btn { width: 100%; margin-top: 4px; padding: 6px; background: #3a3a6a; border: none; border-radius: 4px; color: #ccc; font-size: 12px; cursor: pointer; }
  #save-domains-btn:hover { background: #4a4a8a; }

  #session-list { flex: 1; overflow-y: auto; padding: 8px; }
  .persona-group { margin-bottom: 12px; }
  .persona-group-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #667; padding: 8px 8px 4px; }
  .session-item { padding: 10px 12px; border-radius: 6px; cursor: pointer; margin-bottom: 2px; font-size: 13px; line-height: 1.4; }
  .session-item:hover { background: #1a1a2e; }
  .session-item.active { background: #2a2a5a; }
  .session-item .session-preview { color: #999; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 2px; }
  .session-item .session-meta { color: #556; font-size: 11px; margin-top: 2px; }
  .session-item.incognito { opacity: 0.6; border-left: 2px solid #665; }
  .badge { font-size: 10px; padding: 1px 5px; border-radius: 3px; margin-left: 4px; }
  .badge.incognito { background: #443; color: #aa9; }
  .badge.nsfw { background: #533; color: #c99; }

  /* Main chat area */
  #main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  #chat-header { padding: 12px 20px; background: #16213e; border-bottom: 1px solid #2a2a4a; font-size: 14px; color: #aab; display: flex; align-items: center; gap: 12px; }
  #chat-header .persona-name { font-weight: 600; color: #ccd; }
  #chat-header .emotion-badge { font-size: 12px; color: #8a8aaa; }
  #chat-header .model-badge { font-size: 11px; color: #556; margin-left: auto; }

  /* Model switcher */
  #model-switcher { display: flex; align-items: center; gap: 8px; margin-left: 12px; }
  #model-select { padding: 4px 8px; background: #1a1a2e; color: #eee; border: 1px solid #444; border-radius: 4px; font-size: 12px; cursor: pointer; max-width: 200px; }
  #model-select:focus { outline: none; border-color: #6a6aaa; }
  #model-switch-btn { padding: 4px 12px; background: #4a4a8a; border: none; border-radius: 4px; color: #eee; font-size: 12px; cursor: pointer; display: none; }
  #model-switch-btn:hover { background: #5a5a9a; }
  #model-switch-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  #model-status { font-size: 11px; color: #8a8aaa; }

  /* Modal overlay */
  .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.6); z-index: 100; align-items: center; justify-content: center; }
  .modal-overlay.visible { display: flex; }
  .modal-box { background: #16213e; border: 1px solid #2a2a4a; border-radius: 12px; padding: 28px; max-width: 420px; width: 90%; }
  .modal-box h3 { font-size: 16px; margin-bottom: 12px; color: #ccd; }
  .modal-box p { font-size: 14px; color: #aab; margin-bottom: 16px; line-height: 1.5; }
  .modal-box .modal-info { font-size: 12px; color: #667; margin-bottom: 16px; padding: 8px; background: #1a1a2e; border-radius: 6px; }
  .modal-actions { display: flex; gap: 10px; justify-content: flex-end; }
  .modal-actions button { padding: 8px 20px; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; }
  .modal-actions .cancel { background: #2a2a4a; color: #aab; }
  .modal-actions .cancel:hover { background: #3a3a5a; }
  .modal-actions .confirm { background: #4a4a8a; color: #eee; }
  .modal-actions .confirm:hover { background: #5a5a9a; }
  .modal-actions .confirm:disabled { opacity: 0.5; cursor: not-allowed; }

  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 70%; padding: 12px 16px; border-radius: 16px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: #4a4a8a; }
  .msg.assistant { align-self: flex-start; background: #2a2a4a; }
  .msg img { max-width: 100%; border-radius: 8px; margin-top: 8px; display: block; }
  .typing { color: #888; font-style: italic; }

  #input-area { display: flex; padding: 16px; gap: 8px; background: #16213e; }
  #input { flex: 1; padding: 12px; border: 1px solid #444; border-radius: 8px; background: #1a1a2e; color: #eee; font-size: 15px; }
  #input:focus { outline: none; border-color: #6a6aaa; }
  #send { padding: 12px 24px; background: #4a4a8a; border: none; border-radius: 8px; color: #eee; font-size: 15px; cursor: pointer; }
  #send:hover { background: #5a5a9a; }

  /* Empty state */
  #empty-state { flex: 1; display: flex; align-items: center; justify-content: center; color: #556; font-size: 16px; }
</style>
</head>
<body>

<!-- Sidebar -->
<div id="sidebar">
  <div id="sidebar-header">
    <div id="user-bar">
      <span class="username" id="username-display">...</span>
      <button class="logout-btn" onclick="logout()">Logout</button>
    </div>
    <h2>Personas</h2>
    <select id="persona-select" onchange="onPersonaChange()"></select>
    <div class="toggle-row" id="nsfw-toggle-row">
      <input type="checkbox" id="nsfw-toggle">
      <label for="nsfw-toggle">NSFW mode</label>
    </div>
    <div class="toggle-row">
      <input type="checkbox" id="incognito-toggle">
      <label for="incognito-toggle">Incognito</label>
    </div>
    <details id="domain-config">
      <summary>Knowledge Domains</summary>
      <div id="domain-checkboxes"></div>
      <button id="save-domains-btn" onclick="saveDomains()">Save</button>
    </details>
    <button id="new-chat-btn" onclick="newChat()">+ New Chat</button>
  </div>
  <div id="session-list"></div>
</div>

<!-- Main -->
<div id="main">
  <div id="chat-header">
    <span class="persona-name" id="header-persona">Select a persona</span>
    <span class="emotion-badge" id="header-emotion"></span>
    <span class="model-badge" id="header-model"></span>
    <div id="model-switcher">
      <select id="model-select" onchange="onModelSelectChange()"></select>
      <button id="model-switch-btn" onclick="showSwitchModal()">Switch</button>
      <span id="model-status"></span>
    </div>
  </div>

  <!-- Model switch confirmation modal -->
  <div class="modal-overlay" id="switch-modal">
    <div class="modal-box">
      <h3>Switch LLM Model</h3>
      <p>This will stop the current model and load the new one. The service will be unavailable for up to a minute during the switch.</p>
      <div class="modal-info" id="switch-modal-info"></div>
      <div class="modal-actions">
        <button class="cancel" onclick="hideSwitchModal()">Cancel</button>
        <button class="confirm" id="switch-confirm-btn" onclick="confirmSwitch()">OK, Switch Model</button>
      </div>
    </div>
  </div>
  <div id="chat">
    <div id="empty-state">Start a new chat or select an existing one</div>
  </div>
  <div id="input-area">
    <input id="input" placeholder="Type a message or /image prompt..." autofocus>
    <button id="send" onclick="sendMsg()">Send</button>
  </div>
</div>

<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const personaSelect = document.getElementById('persona-select');
const sessionListEl = document.getElementById('session-list');
const headerPersona = document.getElementById('header-persona');
const headerEmotion = document.getElementById('header-emotion');
const headerModel = document.getElementById('header-model');
const nsfwToggle = document.getElementById('nsfw-toggle');
const nsfwToggleRow = document.getElementById('nsfw-toggle-row');
const incognitoToggle = document.getElementById('incognito-toggle');
const usernameDisplay = document.getElementById('username-display');
const modelSelect = document.getElementById('model-select');
const modelSwitchBtn = document.getElementById('model-switch-btn');
const modelStatus = document.getElementById('model-status');
const switchModal = document.getElementById('switch-modal');
const switchModalInfo = document.getElementById('switch-modal-info');
const switchConfirmBtn = document.getElementById('switch-confirm-btn');

let sessionId = null;
let currentPersonaId = null;
let sessionNsfw = false;
let sessionIncognito = false;
let personas = [];       // array of {id, slug, name, nsfw_capable, ...}
let allSessions = {};    // grouped by persona_slug
let availableModels = [];  // [{key, name, vram_gb, ctx_size}]
let currentModelKey = null; // key of currently loaded model (matched from /model)

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
});

// --- Helpers ---
function getPersonaById(id) {
  return personas.find(p => p.id === id) || null;
}
function getPersonaBySlug(slug) {
  return personas.find(p => p.slug === slug) || null;
}
function currentPersona() {
  return getPersonaById(currentPersonaId);
}

// --- Auth helpers ---
async function authFetch(url, opts) {
  const resp = await fetch(url, opts);
  if (resp.status === 401) {
    window.location.href = '/login';
    throw new Error('Not authenticated');
  }
  return resp;
}

async function logout() {
  await fetch('/api/logout', {method: 'POST'});
  window.location.href = '/login';
}

// --- Init ---
async function init() {
  await loadUser();
  await loadPersonas();
  await loadSessions();
  await loadModelInfo();
}

async function loadUser() {
  try {
    const resp = await authFetch('/api/me');
    const data = await resp.json();
    usernameDisplay.textContent = data.username;
  } catch(e) { /* redirect handled by authFetch */ }
}

async function loadModelInfo() {
  // Load available models list
  try {
    const resp = await authFetch('/models');
    const data = await resp.json();
    availableModels = data.models || [];
  } catch(e) {
    availableModels = [];
  }

  // Load currently running model
  let currentModelName = 'offline';
  try {
    const resp = await fetch('/model');
    const data = await resp.json();
    if (data.model && data.model !== 'offline') {
      currentModelName = data.model;
    }
  } catch(e) {}

  // Match current model to config key by checking if filename is in the model path
  currentModelKey = null;
  for (const m of availableModels) {
    // The /model endpoint returns the gguf filename without extension
    // Match by checking if the model name contains the key or vice versa
    if (currentModelName.toLowerCase().includes(m.key.toLowerCase()) ||
        m.name.toLowerCase().includes(currentModelName.toLowerCase())) {
      currentModelKey = m.key;
      break;
    }
  }

  headerModel.textContent = currentModelKey
    ? availableModels.find(m => m.key === currentModelKey)?.name || currentModelName
    : (currentModelName === 'offline' ? 'LLM offline' : currentModelName);

  // Populate model dropdown
  modelSelect.innerHTML = '';
  for (const m of availableModels) {
    const opt = document.createElement('option');
    opt.value = m.key;
    opt.textContent = m.name + ' (' + m.vram_gb + ' GB)';
    if (m.key === currentModelKey) opt.selected = true;
    modelSelect.appendChild(opt);
  }
  onModelSelectChange();
}

function onModelSelectChange() {
  const selected = modelSelect.value;
  if (selected && selected !== currentModelKey) {
    modelSwitchBtn.style.display = 'inline-block';
  } else {
    modelSwitchBtn.style.display = 'none';
  }
}

function showSwitchModal() {
  const selected = modelSelect.value;
  const model = availableModels.find(m => m.key === selected);
  if (!model) return;

  const current = availableModels.find(m => m.key === currentModelKey);
  let info = 'Loading: ' + model.name + ' (' + model.vram_gb + ' GB VRAM, ctx ' + model.ctx_size + ')';
  if (current) {
    info = 'Current: ' + current.name + ' (' + current.vram_gb + ' GB)\\n' + info;
  }
  switchModalInfo.textContent = info;
  switchConfirmBtn.disabled = false;
  switchConfirmBtn.textContent = 'OK, Switch Model';
  switchModal.classList.add('visible');
}

function hideSwitchModal() {
  switchModal.classList.remove('visible');
}

async function confirmSwitch() {
  const selected = modelSelect.value;
  switchConfirmBtn.disabled = true;
  switchConfirmBtn.textContent = 'Switching...';
  modelStatus.textContent = 'Stopping current model...';

  try {
    const resp = await authFetch('/model/switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model_key: selected})
    });

    if (resp.ok) {
      const data = await resp.json();
      modelStatus.textContent = '';
      hideSwitchModal();
      await loadModelInfo();
    } else {
      const data = await resp.json();
      modelStatus.textContent = 'Error: ' + (data.detail || 'Switch failed');
      switchConfirmBtn.disabled = false;
      switchConfirmBtn.textContent = 'OK, Switch Model';
    }
  } catch(e) {
    modelStatus.textContent = 'Error: ' + e.message;
    switchConfirmBtn.disabled = false;
    switchConfirmBtn.textContent = 'OK, Switch Model';
  }
}

async function loadPersonas() {
  const resp = await authFetch('/personas');
  const data = await resp.json();
  personas = data.personas;  // array of persona objects
  personaSelect.innerHTML = '';
  for (const p of personas) {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = p.name + ' (' + p.slug + ')';
    if (p.id === currentPersonaId) opt.selected = true;
    personaSelect.appendChild(opt);
  }
  // Select first persona if none selected
  if (!currentPersonaId && personas.length > 0) {
    currentPersonaId = personas[0].id;
    personaSelect.value = currentPersonaId;
  }
  onPersonaChange();
}

async function loadSessions() {
  const resp = await authFetch('/sessions');
  const data = await resp.json();
  allSessions = data.sessions;  // grouped by persona_slug
  renderSessionList();
}

function renderSessionList() {
  sessionListEl.innerHTML = '';
  const cp = currentPersona();
  const currentSlug = cp ? cp.slug : null;

  // Collect all group keys, current persona's slug first
  const allKeys = Object.keys(allSessions);
  const ordered = [];
  if (currentSlug && allSessions[currentSlug]) ordered.push(currentSlug);
  for (const k of allKeys) {
    if (k !== currentSlug) ordered.push(k);
  }

  for (const slug of ordered) {
    const sessions = allSessions[slug];
    if (!sessions || sessions.length === 0) continue;

    const label = document.createElement('div');
    label.className = 'persona-group-label';
    const pInfo = getPersonaBySlug(slug);
    label.textContent = pInfo ? pInfo.name : slug;
    sessionListEl.appendChild(label);

    for (const s of sessions) {
      const item = document.createElement('div');
      item.className = 'session-item' + (s.id === sessionId ? ' active' : '') + (s.incognito ? ' incognito' : '');
      item.onclick = () => resumeSession(s.id, s.persona_id, s.incognito, s.nsfw_mode);

      const preview = s.last_user_msg || 'Empty session';
      const time = s.last_time ? new Date(s.last_time).toLocaleString() : '';
      const count = s.message_count || 0;
      let badges = '';
      if (s.incognito) badges += '<span class="badge incognito">incognito</span>';
      if (s.nsfw_mode) badges += '<span class="badge nsfw">nsfw</span>';

      item.innerHTML = '<div>' + escHtml(preview) + badges + '</div>'
        + '<div class="session-meta">' + count + ' msgs' + (time ? ' &middot; ' + time : '') + '</div>';
      sessionListEl.appendChild(item);
    }
  }
}

const ALL_DOMAINS = ['family', 'physical', 'hobbies', 'work', 'emotional', 'memories', 'other'];
const DOMAIN_LABELS = {family:'Family', physical:'Physical', hobbies:'Hobbies', work:'Work', emotional:'Emotional', memories:'Memories', other:'Other'};

function onPersonaChange() {
  currentPersonaId = parseInt(personaSelect.value) || null;
  const p = currentPersona();
  if (p && p.nsfw_capable) {
    nsfwToggleRow.classList.remove('hidden');
  } else {
    nsfwToggleRow.classList.add('hidden');
    nsfwToggle.checked = false;
  }
  renderDomainCheckboxes();
  renderSessionList();
}

function renderDomainCheckboxes() {
  const container = document.getElementById('domain-checkboxes');
  container.innerHTML = '';
  const p = currentPersona();
  const access = (p && p.domain_access) ? p.domain_access : [];
  for (const d of ALL_DOMAINS) {
    const label = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = d;
    cb.checked = access.includes(d);
    label.appendChild(cb);
    label.appendChild(document.createTextNode(' ' + (DOMAIN_LABELS[d] || d)));
    container.appendChild(label);
  }
}

async function saveDomains() {
  if (!currentPersonaId) return;
  const checks = document.querySelectorAll('#domain-checkboxes input[type=checkbox]');
  const selected = [];
  checks.forEach(cb => { if (cb.checked) selected.push(cb.value); });
  const resp = await authFetch('/personas/' + currentPersonaId + '/domains', {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({domain_access: selected}),
  });
  if (resp.ok) {
    const p = currentPersona();
    if (p) p.domain_access = selected;
    document.getElementById('save-domains-btn').textContent = 'Saved!';
    setTimeout(() => { document.getElementById('save-domains-btn').textContent = 'Save'; }, 1500);
  }
}

function buildHeader(persona) {
  let header = persona ? persona.name : 'Unknown';
  if (sessionIncognito) header += ' [incognito]';
  if (sessionNsfw) header += ' [nsfw]';
  return header;
}

async function newChat() {
  if (!currentPersonaId) return;
  sessionNsfw = nsfwToggle.checked;
  sessionIncognito = incognitoToggle.checked;
  const resp = await authFetch('/sessions/new', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      persona_id: currentPersonaId,
      nsfw_mode: sessionNsfw, incognito: sessionIncognito
    })
  });
  const data = await resp.json();
  sessionId = data.session_id;
  chatEl.innerHTML = '';
  headerPersona.textContent = buildHeader(currentPersona());
  headerEmotion.textContent = '';
  await loadSessions();
  inputEl.focus();
}

async function resumeSession(sid, personaId, isIncognito, isNsfw) {
  sessionId = sid;
  currentPersonaId = personaId;
  sessionIncognito = isIncognito || false;
  sessionNsfw = isNsfw || false;
  personaSelect.value = personaId;

  chatEl.innerHTML = '';
  headerPersona.textContent = buildHeader(getPersonaById(personaId));
  headerEmotion.textContent = 'Loading...';

  try {
    const resp = await authFetch('/sessions/' + sid + '/messages');
    const data = await resp.json();
    for (const m of data.messages) {
      addMsg(m.role, m.content);
    }
    headerEmotion.textContent = '';
  } catch(e) {
    headerEmotion.textContent = 'Failed to load messages';
  }

  renderSessionList();
  inputEl.focus();
}

async function sendMsg() {
  const msg = inputEl.value.trim();
  if (!msg) return;
  if (!currentPersonaId) return;
  inputEl.value = '';

  if (!sessionId) {
    sessionNsfw = nsfwToggle.checked;
    sessionIncognito = incognitoToggle.checked;
    const resp = await authFetch('/sessions/new', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        persona_id: currentPersonaId,
        nsfw_mode: sessionNsfw, incognito: sessionIncognito
      })
    });
    const data = await resp.json();
    sessionId = data.session_id;
    chatEl.innerHTML = '';
    headerPersona.textContent = buildHeader(currentPersona());
  }

  addMsg('user', msg);
  const typing = addMsg('assistant', 'Thinking...', 'typing');

  try {
    const body = {
      message: msg,
      persona_id: currentPersonaId,
      session_id: sessionId,
      nsfw_mode: sessionNsfw,
      incognito: sessionIncognito
    };
    const resp = await authFetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await resp.json();
    sessionId = data.session_id || sessionId;
    typing.remove();

    let html = escHtml(data.reply);
    if (data.images && data.images.length > 0) {
      for (const img of data.images) {
        // Extract relative path from comfyui/output/ onwards (e.g. "52/3/persona_gen_0001.png")
        const outputIdx = img.indexOf('comfyui/output/');
        const relPath = outputIdx >= 0 ? img.substring(outputIdx + 'comfyui/output/'.length) : img.split('/').pop();
        html += '<img src="/image/' + relPath + '" alt="generated image">';
      }
    }
    addMsgHtml('assistant', html);

    if (data.emotion_description) {
      headerEmotion.textContent = data.emotion_description;
    }

    loadSessions();
  } catch(e) {
    typing.remove();
    addMsg('assistant', 'Error: ' + e.message);
  }
}

function addMsg(role, text, cls) {
  const empty = document.getElementById('empty-state');
  if (empty) empty.remove();

  const div = document.createElement('div');
  div.className = 'msg ' + role + (cls ? ' ' + cls : '');
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function addMsgHtml(role, html) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = html;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

init();
</script>
</body>
</html>"""
