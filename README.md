# UNDER CONSTRUCTION!

# AI persona framework
 Modular LLM assistant with memory, tools and persona simulation
AI Persona Framework: Modular LLM Assistant with Memory, Tools, and Persona Simulation

Author: Kenneth
License: MIT

Description:
A modular, open-source AI assistant system designed for personal knowledge integration, persona simulation, and tool augmentation.
This project aims to combine LangChain, Qdrant, and Chainlit to deliver a fully local, extensible framework.

Planned features:
- Chainlit web UI with persona selector and chat interface
- LangChain-based conversation agent with custom persona profiles
- Qdrant vector store for semantic memory (RAG support)
- Optional Neo4j graph for relationships and knowledge graphs
- Tool system for invoking Stable Diffusion and other services
- Future MCP and multi-user integration support
- Local LLM inference (e.g., LLaMA, Athene) using Transformers or llama.cpp

---

Folder Structure:

ai_persona_framework/
├── README.md
├── requirements.txt
├── .env.example
├── main.py                # Entry point for Chainlit
├── agents/
│   ├── base_agent.py      # Conversation agent logic
│   └── persona_loader.py  # Load persona YAMLs
├── memory/
│   ├── qdrant_client.py   # Interface with Qdrant
│   └── memory_utils.py    # Embedding, retrieval, storage
├── tools/
│   ├── image_tool.py      # Stable Diffusion API caller
│   └── registry.py        # Tool loader and integration
├── models/
│   └── model_runner.py    # Load/serve LLMs (local)
├── personas/
│   ├── emma.yaml
│   ├── alfred.yaml
│   └── yourname.yaml
├── data/
│   └── logs/              # Chat logs and extracted facts
└── utils/
    ├── config.py          # Config loader (dotenv/YAML)
    └── helpers.py         # Misc utilities

---

Getting Started:
1. Clone the repo
2. Set up Python env (3.12+ recommended)
3. Install dependencies from requirements.txt
4. Run Qdrant via Docker: `docker run -p 6333:6333 qdrant/qdrant`
5. Launch with: `chainlit run main.py`

---

License: MIT
'''
