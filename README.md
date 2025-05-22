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

| Layer               | Backend               | Purpose                                |
|---------------------|------------------------|----------------------------------------|
| **Short-term**      | In-memory              | Current session context buffer         |
| **Semantic recall** | Qdrant                 | Similar past messages via embeddings   |
| **Structured facts**| PostgreSQL + pgvector  | Core identity, journal logs            |
| **Graph memory**    | Neo4j                  | Topic trees, emotional scoring, links  |

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
├── core/                      # Core engine logic
│   ├── engine.py              # Main orchestrator (input → memory → output)
│   ├── context_builder.py     # Assembles LLM prompts from memory layers
│   └── router.py              # (Future) Tool/memory routing logic
│
├── memory/                    # Memory modules
│   ├── buffer.py              # In-session chat buffer (short-term memory)
│   ├── vector_store.py        # Qdrant: semantic message retrieval
│   ├── fact_store.py          # PostgreSQL: structured facts, logs
│   ├── topic_graph.py         # Neo4j: topic trees and relationships
│   └── classifier.py          # NLP topic tagging, sentiment scoring
│
├── agents/                    # Persona configuration
│   ├── personality_config.json # JSON file for defining agent personalities
│   └── loader.py              # Loads config and applies temperature/emotion logic
│
├── tools/                     # Optional agent tools (image, search, etc.)
│   ├── image_gen.py           # Stable Diffusion / image creation interface
│   ├── sandbox_env.py         # (Future) Chrooted tool execution environment
│   └── web_research.py        # (Future) Agent browser/research capabilities
│
├── interface/                 # User interfaces
│   ├── cli_chat.py            # CLI chatbot interface for development/testing
│   ├── web/                   # (Optional) Web UI frontend
│   └── discord/               # (Optional) interface for discord chatbot
│   └── api/                   # (Optional) WebSocket or REST API layer
│
├── config/                    # Configuration
│   └── .env                   # Environment variables for DBs, ports, paths
│
├── scripts/                   # Setup and utility scripts
│   ├── init_postgres.py       # Sets up PostgreSQL tables
│   ├── init_neo4j.py          # Sets up Neo4j schema
│   ├── init_qdrant.py         # Creates Qdrant collection
│   └── init_structure.sh      # Creates this folder structure
│
├── data/                      # Local storage (logs, corpus, memory dumps)
│   └── echo_corpus/           # (Future) User training corpora
```

---

## Status
pre-alpha 0.01
This project is at it's birth and is under **active construction**.  
Initial setup scripts are in place.  
LLM prompting, vector memory, topic tagging, and persona switching are being built.  
Stable Diffusion + sandbox tools planned next.

Pull requests, ideas, and contributors are welcome — local-first AI should be **ours**, not theirs.
