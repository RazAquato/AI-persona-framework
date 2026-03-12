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
CLI Chat Interface
------------------
Direct conversation through the engine — no API dependency.
Supports persona selection, optional emotion display, and /commands.

Usage:
    python3 cli_chat.py                          # default persona, user 9999
    python3 cli_chat.py --persona eva            # specific persona
    python3 cli_chat.py --user 52 --persona eva  # specific user
    python3 cli_chat.py --show-emotions          # display emotion state each turn
"""

import sys
import os
import argparse

# Path setup — make sure we can import from all packages
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_CLIENT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MEMORY_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "memory-server"))
SHARED_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "shared"))

for p in [LLM_CLIENT_ROOT, MEMORY_PATH, SHARED_PATH]:
    if p not in sys.path:
        sys.path.append(p)

from core.engine import run_conversation_turn


def format_emotions(emotions: dict, top_n: int = 5) -> str:
    """Format the top N emotions as a compact status line."""
    active = sorted(
        [(k, v) for k, v in emotions.items() if v > 0.05],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]
    if not active:
        return "  [neutral]"
    parts = [f"{k}={v:.2f}" for k, v in active]
    return "  " + " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="AI Persona CLI Chat")
    parser.add_argument("--persona", default="default", help="Personality ID (default, eva, eva-nsfw, debug)")
    parser.add_argument("--user", type=int, default=9999, help="User ID (default: 9999 test user)")
    parser.add_argument("--show-emotions", action="store_true", help="Show persona emotion state each turn")
    parser.add_argument("--show-user-emotions", action="store_true", help="Show detected user emotions each turn")
    args = parser.parse_args()

    persona_name = args.persona
    user_id = args.user
    session_id = None  # engine will auto-create/resume

    print(f"--- AI Persona Chat ---")
    print(f"Persona: {persona_name} | User ID: {user_id}")
    print(f"Type 'quit' or 'exit' to end. Type '/emotions' to toggle emotion display.\n")

    show_emotions = args.show_emotions
    show_user_emotions = args.show_user_emotions

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/emotions":
            show_emotions = not show_emotions
            print(f"  [Emotion display: {'ON' if show_emotions else 'OFF'}]")
            continue

        if user_input.lower() == "/user-emotions":
            show_user_emotions = not show_user_emotions
            print(f"  [User emotion display: {'ON' if show_user_emotions else 'OFF'}]")
            continue

        try:
            result = run_conversation_turn(
                user_id=user_id,
                user_input=user_input,
                personality_id=persona_name,
                session_id=session_id,
            )

            # Capture session_id for subsequent turns
            session_id = result["session_id"]

            # Print reply
            print(f"\n{persona_name.capitalize()}: {result['assistant_reply']}\n")

            # Show emotions if enabled
            if show_emotions and result.get("persona_emotions"):
                print(f"  [Persona]{format_emotions(result['persona_emotions'])}")
                if result.get("emotion_description"):
                    print(f"  [{result['emotion_description']}]")

            if show_user_emotions and result.get("user_emotions"):
                print(f"  [User]{format_emotions(result['user_emotions'])}")

            if show_emotions or show_user_emotions:
                print()

        except Exception as e:
            print(f"\n  [ERROR] {e}\n")


if __name__ == "__main__":
    main()
