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

from fastapi import FastAPI, Request
from pydantic import BaseModel
import core.prompt_builder as pb
import core.llm_client as llm

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str
    persona: str = "maya"

@app.post("/chat")
async def chat(req: ChatRequest):
    persona_cfg = pb.load_persona(req.persona)
    history = pb.load_recent_history(req.session_id)
    emotion_state = pb.get_emotion_state(req.session_id, req.persona)

    prompt = pb.build_prompt(persona_cfg, history, emotion_state, req.message)
    response = llm.query(prompt)

    pb.update_emotions(req.session_id, response, emotion_state)
    pb.save_message(req.session_id, req.message, response)

    return {"response": response}

