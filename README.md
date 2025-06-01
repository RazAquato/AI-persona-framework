# UNDER CONSTRUCTION!

# AI Persona Framework
Modular LLM assistant with memory, tools and persona simulation

**Author:** Kenneth Haider

**License**: [GNU General Public License v3.0 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html)

---

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).

You are free to **use, modify, and distribute** this software — under the condition that **any derivative works or redistributions must also be licensed under GPLv3**.

This ensures the project remains **free and open-source**, even when modified or extended.

See the [LICENSE](./LICENSE) file for full details.

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

##
this is not done yet! we aim for modular requirements for LLM client and memory-server
Install with:

```bash
pip install -r requirements.txt
```

`requirements.txt` will probably include:

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
AI-persona-framework/
├── LLM-client/                        # Core persona engine and orchestration logic
│   ├── agents/                        # Agent configs and personality logic
│   │   ├── loader.py                  # Main agent loader interface
│   │   ├── personality_config.json    # Global agent defaults
│   │   ├── loaders/                   # Personality + agent sub-loaders
│   │   │   ├── agent_loader.py        # Loads agent metadata
│   │   │   └── personality_loader.py  # Loads JSON personality definitions
│   │   └── system_personas/          # Built-in example personas
│   │       ├── arthur.json
│   │       ├── maya.json
│   │       ├── rufus.json
│   │       └── personality_config.json
│   ├── config/                        # Environment-specific setup
│   │   ├── .env                       # Local environment overrides
│   │   └── env.template               # Sample config template
│   ├── core/                          # Central processing and orchestration
│   │   ├── engine.py                  # Main controller (input → memory → output)
│   │   ├── context_builder.py         # Builds final prompt from memory layers
│   │   ├── router.py                  # Routes tools, memory, agents dynamically
│   │   └── echo_controller.py         # (Future) Handles echo persona invocation
│   ├── interface/                     # User interfaces
│   │   ├── cli_chat.py                # Terminal interface
│   │   ├── api/routes.py              # API routes for external access
│   │   ├── web/                       # Optional: browser front-end
│   │   └── discord/                   # Optional: Discord chatbot interface
│   └── tests/                         # Unit tests for LLM-client
│       ├── test_engine.py
│       ├── test_loader.py
│       ├── test_agent_loader.py
│       ├── test_llm_client.py
│       └── test_personality_loader.py

├── memory-server/                    # Server-side memory logic
│   ├── memory/                        # Memory backend logic
│   │   ├── buffer.py                  # In-memory short-term buffer
│   │   ├── vector_store.py            # Qdrant-based semantic memory
│   │   ├── fact_store.py              # PostgreSQL facts + journal memory
│   │   ├── topic_graph.py             # Neo4j topic/emotion relationship graph
│   │   ├── metadata.py                # Metadata tagging per message
│   │   └── summarizer.py              # (Planned) Session summarization & decay
│   ├── echo/                          # Echo generation (simulated personalities)
│   │   ├── builder.py                 # Extracts echo corpuses from logs
│   │   ├── echo_prompt.py             # Prompt constructor for echo-mode
│   │   ├── traits_extractor.py        # Extracts traits from logs
│   │   └── data/user_123/             # Sample extracted echo data
│   │       ├── facts.json
│   │       ├── logs.json1
│   │       ├── traits.json
│   │       └── sessions/
│   │           └── session_2025_05.json
│   ├── analytics/                     # Visualization & analysis
│   │   ├── graph_dashboard.ipynb      # Explore Neo4j topic graphs
│   │   ├── memory_report.py           # Print/Export user memory
│   │   └── metadata_timeline.ipynb    # Visualize emotion/topics over time
│   ├── scripts/                       # Setup and migration
│   │   ├── init_postgres.py           # Create PostgreSQL schema
│   │   ├── init_qdrant.py             # Create/reset Qdrant collections
│   │   ├── init_neo4j.py              # Set up Neo4j schema
│   │   └── migrate_logs_to_echo.py    # Import legacy logs into echo format
│   ├── config/
│   │   ├── .env
│   │   ├── env.template
│   │   └── settings.yaml
│   └── tests/                         # Memory unit/integration tests
│       ├── test_vector_store.py
│       ├── test_fact_store.py
│       ├── test_topic_graph.py
│       ├── test_chat_store.py
│       ├── test_init_postgres.py
│       ├── test_init_neo4j.py
│       └── test_init_qdrant.py
├── shared/                            # Shared utilities across components
│   ├── config/
│   │   ├── .env
│   │   └── emotions.yaml              # Default emotions schema
│   ├── tools/                         # Tool logic available to all agents
│   │   ├── image_gen.py               # Interface for image generation
│   │   ├── sandbox_env.py             # Sandboxed CLI execution
│   │   ├── web_research.py            # Web search/summarization tool
│   │   └── tool_registry.py           # Registers tool capabilities for agents
│   ├── scripts/
│   │   ├── agent_creator.py           # Script to add/update agents
│   │   └── agent_creator_with_emotions.py # Adds emotional defaults
│   └── tests/
│       ├── test_agent_creator.py
│       └── test_agent_creator_with_emotions.py

├── doc/                               # System design and documentation
│   ├── echo.md
│   ├── memory.md
│   ├── flowchart.txt
│   └── system_architecture.png

├── temp/                              # Temporary or dev scripts
│   ├── del_qdrant.py
│   ├── requirements_bloated.txt       # Full CUDA-enabled dependencies
│   └── requirements2.txt              # Alternative/test config

├── data/                              # (optional) Logs, backups, exports
│   ├── logs/                          # Raw chat logs (not yet in echo/)
│   └── backups/                       # Database dumps and snapshots

├── test_all_python_code.py           # Test runner for all unit tests
├── check_for_unittests.py            # Checks if all modules are covered by tests
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
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

