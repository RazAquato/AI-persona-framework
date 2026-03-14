# AI Persona Framework

Local-first LLM platform with persistent memory, dynamic emotions, and evolving AI personas.

**Author:** Kenneth Haider
**License:** [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html)

---

## Vision

Your AI companion remembers your dog's name from 2 years ago, your favorite sports team, and the day you felt heartbroken. Each persona develops its own personality and emotional relationship with you over time — not by mirroring you, but by being its own character.

As AI enters everyday life, the most intimate data you produce — your emotions, struggles, relationships, and inner world — should stay on your own hardware. This framework exists so that depth and privacy aren't mutually exclusive.

### Echo: Digital Legacy

When you interact with AI personas over months and years, the system builds a rich profile of who you are: your personality, values, humor, knowledge, and feelings about the people in your life. Echo distills this into a persona that your children could talk to someday — carrying forward the positive parts of who you were, filtered for privacy and safety.

### Brain 2.0: External Memory

For those who hyperfocus on topics that go dormant, the system acts as a searchable knowledge graph with a conversational interface. Topics, facts, and connections accumulate across sessions. The persona layer makes daily engagement sustainable — you're not writing a diary, you're having a conversation.

---

## What It Does Today

- **4 default personas** with distinct personalities (Maya/girlfriend, Coach/trainer, Dr. Lumen/psychiatrist, DebugBot/debug)
- **Per-user persona ownership** — create, edit, delete your own personas via API/UI
- **Persistent memory** across sessions: chat history, semantic search, structured facts, topic graph
- **Dynamic persona emotions** — each persona maintains emotional state toward you (joy, trust, sadness, etc.) with time-based decay and absence drift
- **User emotion detection** — 18-dimensional emotion vector per message
- **LLM-based knowledge extraction** — extracts facts, entities, and topics from conversations with two-tier classification (identity vs emotional)
- **External data sync** — Mealie (recipes/cooking habits) and Immich (photo archive: people, locations, devices) as identity-tier facts
- **NSFW mode** — per-session toggle on capable personas, with safety-filtered extraction
- **Incognito mode** — chat without any DB persistence
- **Image generation** — ComfyUI integration with per-user/per-persona output folders
- **Model hot-swap** — switch LLM models from the web UI (kills/restarts llama-server)
- **Cookie-based auth** — PBKDF2 password hashing, HMAC-signed session cookies, full user isolation
- **Web UI** — dark-themed chat interface with sidebar, session history, persona selector, model switcher
- **CLI chat** — terminal interface with `--persona`, `--nsfw`, `--incognito`, `--show-emotions` flags
- **Echo MVP** — prompt-based personality simulation from conversation history corpus

---

## Architecture

Three packages, no shared install — cross-imports via `sys.path`.

```
User input
  → LLM-client/core/engine.py              # orchestrates one conversation turn
      → memory-server/memory/context_builder.py   # aggregates facts, vectors, topics
      → shared/analysis/emotion_handler.py         # user + persona emotion processing
      → LLM-client/core/prompt_builder.py          # assembles system prompt
      → LLM-client/core/llm_client.py              # HTTP to llama-server
      → memory-server/memory/*_store.py            # persist everything
  → Response with emotions, extracted knowledge, tool results
```

### Memory Layers

| Layer | Backend | Purpose |
|---|---|---|
| Short-term buffer | In-memory | Current session context |
| Chat history | PostgreSQL | Full conversation log |
| Semantic search | Qdrant (384-dim) | Similar past messages via embeddings |
| Structured facts | PostgreSQL | Identity + emotional tier facts with tags, valence, confidence |
| Topic graph | Neo4j | Topic relationships, cross-conversation links |
| User emotions | PostgreSQL (JSONB) | Per-message 18-dim emotion vectors |
| Persona emotions | PostgreSQL (JSONB) | Per-user-per-persona emotional state + history |

### Two-Tier Memory Model

- **Identity tier** — who the user IS: stable facts, preferences, personality, life events, positive sentiments. Visible to all personas and Echo.
- **Emotional tier** — how the user FEELS now: moods, struggles, relationship tensions. Visible to personas only, never Echo. Decays and archives over time.

### External Data Adapters

| Adapter | Source | Facts Produced |
|---|---|---|
| Mealie | Recipe/meal plan API | Cooking frequency, categories, ingredients |
| Immich | Photo archive API | People (face frequency), locations (EXIF GPS), devices |

Adapters use snapshot sync: delete old facts by source_type, insert fresh set. Designed for monthly cron runs.

---

## Running

### Prerequisites

- Python 3.10+ with venv
- PostgreSQL 15+ (via PgBouncer)
- Qdrant 1.17+
- Neo4j 2026.02+
- NVIDIA GPU with CUDA (for local LLM inference)
- llama.cpp (compiled with CUDA support)

### Quick Start

```bash
# Activate venv
source ~/venvs/AI-persona-framework-venv/bin/activate

# Start the LLM server
python3 LLM-client/load_LLM.py --model qwen9b

# Start the web UI (in another terminal)
cd LLM-client/interface/api && uvicorn app:app --host 0.0.0.0 --port 8000

# Or use the CLI
python3 LLM-client/interface/cli_chat.py --persona girlfriend --show-emotions
```

### Available Models

| Key | Model | VRAM | Notes |
|---|---|---|---|
| `qwen9b` | Qwen 3.5 9B (Uncensored Q8) | ~9.5 GB | Primary chat model |
| `evathene` | Evathene v1.3 (Q4_K_M) | ~8 GB | Legacy persona model |
| `deepseek` | DeepSeek v2 (Q4_K_M) | ~8 GB | Coding/debug |
| `cpp_tutor` | CppTutor (Q4_K) | ~4 GB | C++ tutor |
| `tinyllama` | TinyLlama 1.1B (Q4_K_M) | ~1 GB | Testing only |

### Tests

```bash
python3 test_all_python_code.py    # 433 tests across all modules
```

---

## Folder Structure

```
AI-persona-framework/
├── LLM-client/                    # Persona engine and orchestration
│   ├── core/
│   │   ├── engine.py              # Main conversation pipeline (17 steps)
│   │   ├── llm_client.py          # HTTP client for llama-server
│   │   ├── prompt_builder.py      # System prompt assembly
│   │   ├── router.py              # Tool command routing (/image, etc.)
│   │   ├── auth.py                # PBKDF2 passwords + HMAC cookies
│   │   └── model_manager.py       # Kill/restart llama-server for model switching
│   ├── interface/
│   │   ├── api/app.py             # FastAPI web UI + REST API
│   │   └── cli_chat.py            # Terminal chat interface
│   ├── config/
│   │   ├── model_configs.yaml     # LLM model definitions (paths, VRAM, ctx)
│   │   └── personality_config.json # Default persona definitions
│   ├── load_LLM.py                # Bootstraps llama-server with CUDA
│   └── tests/                     # 87 tests
│
├── memory-server/                 # Memory backends and persistence
│   ├── memory/
│   │   ├── chat_store.py          # Session + message CRUD
│   │   ├── vector_store.py        # Qdrant semantic search
│   │   ├── fact_store.py          # Structured facts with tiers, tags, valence
│   │   ├── topic_graph.py         # Neo4j topic relationships
│   │   ├── emotion_store.py       # User emotion vectors per message
│   │   ├── persona_emotion_store.py # Persona emotional state per user
│   │   ├── persona_store.py       # Persona CRUD (DB-backed)
│   │   ├── context_builder.py     # Aggregates all memory layers for prompt
│   │   ├── buffer.py              # In-memory conversation buffer
│   │   └── user_store.py          # User accounts
│   ├── echo/
│   │   ├── corpus_builder.py      # Builds Echo training corpus (identity-tier only)
│   │   ├── echo_prompt.py         # Echo system prompt constructor
│   │   └── traits_extractor.py    # Stub — personality trait extraction
│   ├── scripts/
│   │   ├── init_postgres.py       # DB schema setup
│   │   ├── sync_mealie.py         # CLI: monthly Mealie recipe sync
│   │   └── sync_immich.py         # CLI: monthly Immich photo sync
│   └── tests/                     # 150 tests
│
├── shared/                        # Cross-package utilities
│   ├── analysis/
│   │   ├── emotion_handler.py     # User emotion detection + persona emotion engine
│   │   ├── knowledge_extractor.py # Regex-based fact extraction (fallback)
│   │   └── llm_knowledge_extractor.py # LLM-based extraction with tier classification
│   ├── tools/
│   │   ├── tool_registry.py       # Tool dispatch registry
│   │   ├── image_gen.py           # Image generation entry point
│   │   ├── image_orchestrator.py  # ComfyUI bridge
│   │   ├── mealie_sync.py         # Mealie recipe/meal plan adapter
│   │   └── immich_sync.py         # Immich photo archive adapter
│   └── tests/                     # 196 tests
│
├── scripts/
│   └── cron_sync_adapters.sh      # Monthly cron: Mealie + Immich sync
├── doc/                           # Design documents
├── test_all_python_code.py        # Runs all tests across all modules
└── LICENSE
```

---

## Status

**Beta** — core conversation pipeline, memory system, web UI, auth, emotions, knowledge extraction, external adapters, and model switching are all functional with 433 passing tests.

**In design:** Topic-emotion architecture with per-persona salience tracking, temporal pattern recognition, reflection agent, and persona autonomy (personas develop their own opinions rather than mirroring the user).

**Roadmap:**
- Per-persona fact scoping (stop oversharing between personas)
- Topic salience with decay (surface relevant facts, not everything)
- Nightly reflection agent (pattern detection in temporal data)
- Persona topic emotions (autonomous, non-compliant personality development)
- Additional data adapters (Fitbit, ChatGPT history import, document ingestion)
- Home automation integration (AI-driven Home Assistant via Neo4j house graph)

---

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).

You are free to **use, modify, and distribute** this software — under the condition that **any derivative works must also be licensed under GPLv3**.

Privacy is non-negotiable. The data this system produces — emotional, psychiatric, intimate — is the most personal dataset a human can generate. It stays on your hardware.

Pull requests, ideas, and contributors are welcome. Local-first AI should be **ours**, not theirs.
