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

import requests

persona = "maya"
session_id = "test-user-1"

while True:
    msg = input("You: ")
    if msg.lower() in ("exit", "quit"):
        break

    res = requests.post("http://localhost:8000/chat", json={
        "persona": persona,
        "session_id": session_id,
        "message": msg
    })
    print(f"{persona.capitalize()}: {res.json()['response']}")

