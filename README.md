# UNDER CONSTRUCTION!

# AI Persona Framework
Modular LLM assistant with memory, tools and persona simulation

**Author:** Kenneth  
**License:** MIT

---

## Vision

> Imagine a world where your AI companion remembers your dog’s name from 2 years ago, your favorite sports team, and the day you felt heartbroken. This framework aims to simulate real-life emotional continuity through layered memory and dynamic agent design.  
> As AI ramps up in the world, what you give to companies that offer "AI girlfriends" is not just 'a fee for their services' — you give them trust, emotions, and most importantly: a detailed view of yourself.  
> **This data should be in your hands.**  
> The vision of this framework is to create a digital assistant that is as capable as you want — but where your data resides locally, on your own server.

## Future vision: AI ecco
> Imagine having an idea, pitching that idea to "an ecco of yourself", and get feedback based on historic knowledge/past conversations you have had?
> This idea stems from the "Second brain with zettelkasten" just broader. instad of having hyperlinked notes you can review; imagine having hyperlinked metadata that your AI assistant can distill from, fine-tune, see connections and present to you?
> Imagine when you one day disappear, that your children could have an ecco to seek guidance from?
> To support this vision, we need to harvest tons of metadata from each conversation and make that metadata accessible for the AI to train on or search upon

---

## Description

The aim is to create an **open-source cognitive architecture** for building persistent, evolving AI agents that learn from long-term interactions.

A secondary goal is to create **"digital echos" of users** it interacts with — so if a user disappears, the AI framework can simulate their personality and style based on previous conversations.

> "When I die, my echo will keep resonating for my children." — unknown

---

## Core Functionality

- Use **any LLM backend** (local or remote), modular and swappable
- Dynamic **AI agent personifications** with memory, personality, emotion
- **Long-term memory** across sessions (vector, structured, and graph-based)
- **Context-aware prompts**, topic detection, and memory recall
- **Fact retention and journaling** for persistent knowledge
- Optional **image generation tools** (e.g., Stable Diffusion)
- Future support for **sandboxed tools** (web research, files, automation)

---

## Memory Layers

| Layer               | Backend                | Purpose                                |
|---------------------|------------------------|----------------------------------------|
| **Short-term**      | In-memory              | Current session context buffer         |
| **Semantic recall** | Qdrant                 | Similar past messages via embeddings   |
| **Structured facts**| PostgreSQL + pgvector  | Core identity, journal logs            |
| **Graph memory**    | Neo4j                  | Topic trees, emotional scoring, links  |

You can read more under the doc/ folder on how this is intented to work.
---

## Requirements

### System Requirements

- Python 3.10+
- PostgreSQL 15+ with `pgvector` extension
- Neo4j (community or enterprise)
- Qdrant (Docker or binary)
- NVIDIA GPU with CUDA (for LLM and image generation)
- Optional: llama.cpp backend running on port `8080`

### Python Libraries

Install with:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

- `sentence-transformers`
- `qdrant-client`
- `psycopg2-binary`
- `sqlalchemy`
- `neo4j`
- `python-dotenv`
- `textblob`
- `transformers`
- `keybert`
- `scikit-learn`

---

## Folder Structure

```
ai-assistant/
├── core/                          # Central processing and orchestration
│   ├── engine.py                  # Main controller (input → memory → output)
│   ├── context_builder.py         # Builds final prompt from memory layers
│   ├── router.py                  # Routes tools, memory, agents dynamically
│   └── echo_controller.py         # (Future) Handles echo persona invocation
│
├── memory/                        # Memory access and logic
│   ├── buffer.py                  # In-memory chat buffer (short-term memory)
│   ├── vector_store.py            # Qdrant-based vector memory
│   ├── fact_store.py              # PostgreSQL: facts, structured memory
│   ├── topic_graph.py             # Neo4j: topic and relation memory
│   ├── metadata.py                # Active metadata analysis per message
│   └── summarizer.py              # (Future) Session summarization and memory decay
│
├── metadata/                      # (New) Raw and processed metadata logs
│   ├── message_metadata.jsonl     # Exported metadata (topic, sentiment, etc.)
│   ├── tool_usage.jsonl           # Tool calls per message
│   └── topic_stats.json           # Aggregated user/topic interaction stats
│
├── agents/                        # Persona configuration and logic
│   ├── personality_config.json    # Persona definitions (tone, thresholds, etc.)
│   ├── profiles/                  # Per-agent behavioral rules / extensions
│   │   └── maya.json              # Example override config
│   └── loader.py                  # Loads agents and temperature logic
│
├── echo/                          # Echo generation system
│   ├── builder.py                 # Extracts echo corpus from chat logs
│   ├── traits_extractor.py        # Analyzes tone, behavior, values
│   ├── echo_prompt.py             # Creates echo-mode prompts for LLM
│   └── data/                      # Structured training data
│       └── user_123/              # Logs and extracted traits
│           ├── logs.jsonl
│           ├── facts.json
│           ├── traits.json
│           └── sessions/
│               └── session_2024_01.json
│
├── tools/                         # Agent-accessible external tools
│   ├── image_gen.py               # Stable Diffusion image interface
│   ├── sandbox_env.py             # Sandboxed chroot environment (file/web)
│   ├── web_research.py            # Web search & document summarizer
│   └── tool_registry.py           # Tool definitions and input/output schemas
│
├── interface/                     # User interfaces
│   ├── cli_chat.py                # Terminal interface
│   ├── api/                       # REST or WebSocket server
│   │   └── routes.py
│   ├── web/                       # Optional web-based UI
│   └── discord/                   # Optional Discord chatbot interface
│
├── config/                        # Environment and runtime config
│   ├── .env                       # Environment variables
│   └── settings.yaml              # (Optional) Central override config
│
├── analytics/                     # (Optional) Memory + metadata visualizations
│   ├── graph_dashboard.ipynb      # View Neo4j topics and entity graphs
│   ├── metadata_timeline.ipynb    # Plot emotional arc over time
│   └── memory_report.py           # Exports memory snapshots per user
│
├── scripts/                       # Init and admin tools
│   ├── init_postgres.py
│   ├── init_qdrant.py
│   ├── init_neo4j.py
│   ├── init_structure.sh
│   └── migrate_logs_to_echo.py    # Convert chat logs into echo-ready format
│
├── data/                          # Local data + storage
│   ├── logs/                      # Raw conversation logs
│   ├── backups/                   # Database dumps
│   └── echo_corpus/               # (Deprecated; now in echo/data/)
│
├── tests/                         # Unit and integration tests
│   └── test_memory_flow.py
│
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Status
pre-alpha 0.01
This project is at it's birth and is under **active construction**.  
Initial setup scripts are in place.  
LLM prompting, vector memory, topic tagging, and persona switching are being built.  
Stable Diffusion + sandbox tools planned next.

Pull requests, ideas, and contributors are welcome — local-first AI should be **ours**, not theirs.
