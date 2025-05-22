# Local AI Assistant with Long-Term Memory

**Local AI Assistant** is a modular, self-hosted chatbot that learns a personalized “digital echo” of the user over time. It serves as a conversational assistant and builds a persistent persona model of the user from past conversations. The project is designed for privacy (all data stays on your hardware) and extensibility, integrating long-term memory so the AI can recall context, preferences, and facts about the user in future interactions.

## Features

- **Conversational AI Assistant:** Engage in natural dialogue with an AI powered by a local large language model (LLM). The assistant can answer questions, help with tasks, and have extended conversations – all without cloud services.
- **Long-Term Memory (Persona Storage):** The assistant retains important information from each conversation. It stores semantic memories in a vector database for context recall, and structured knowledge (facts, relationships, preferences) in a graph database:contentReference[oaicite:13]{index=13}. This allows it to remember details about the user (e.g. name, interests, writing style) and maintain continuity over time.
- **Evolving User Persona:** The system analyzes conversations to update the user’s persona profile – for example, learning the user’s favorite music or habitual tone. This “digital echo” is used to personalize responses (and can be exported for other uses, like drafting emails in the user’s style in the future).
- **Modular & Open-Source:** Each component of the system is modular. The model, memory stores, and interface can be swapped or upgraded independently. The entire project is open-source, aiming to foster community contributions.
- **Privacy-Focused:** All data and models run locally. No conversation data is sent to external APIs by default. Users retain ownership of their data (with conversation logs and persona data stored on local disk).
- **Extensible Architecture:** Designed to support plugins and additional tools (e.g., email integration, calendars) via a flexible agent framework. The assistant can be extended to interface with other systems as needed.

## Architecture

The assistant consists of multiple components that work together (see diagram below):

- **Frontend (Chat UI):** A web-based chat interface (built with React/Next.js via the LobeChat framework) that mimics ChatGPT’s UI. This allows users to interact with the assistant in real time. It supports features like multi-turn chats and knowledge base uploads.
- **Backend (Agent Server):** A FastAPI (Python) server implementing the conversational logic. This agent receives user messages from the UI, retrieves relevant memories, formulates the prompt, calls the LLM model, and returns the assistant’s answer. It uses LangChain to manage the chain-of-thought and memory retrieval. 
- **LLM Model Service:** A dedicated service hosting the large language model on a GPU. By default this uses an open-source model (e.g. Llama-2-Chat) running via HuggingFace’s Text Generation Inference server. The agent communicates with the model service over an API (OpenAI-compatible REST interface), which makes it easy to swap models. *Model weights are stored locally* (under `models/`), and you can choose a different model by updating the config.
- **Vector Memory Database:** A semantic memory store (ChromaDB). It saves vector embeddings of user and assistant messages or notes. On each new query, the agent finds similar past conversations or statements by vector similarity, and injects them into the prompt to provide context. This is how the assistant “remembers” things you said before, even with long gaps:contentReference[oaicite:14]{index=14}.
- **Graph Memory Database:** A knowledge graph (Neo4j) that stores structured information about the user’s persona and facts. For example, it might contain nodes like `Person(name=User)` linked to `Interest(name=Classical Music)` via a **LIKES** relationship. The agent can query this graph for exact facts (e.g., *“what is the user’s favorite genre?”*) and also updates it when new facts are learned (using Cypher queries). This provides a robust long-term memory that complements the fuzzy semantic search with exact, up-to-date knowledge:contentReference[oaicite:15]{index=15}.
- **Memory Extraction & Reflection:** After each conversation turn, the agent updates the memory: important new information is embedded and upserted into Chroma, and key facts are extracted (via simple NLP rules or an auxiliary LLM prompt) to update Neo4j. There are also background routines (which can be scheduled or queued) that summarize recent conversations or periodically prune/refine memory (for example, maintaining a concise summary of the user’s preferences).
- **(Optional) Redis Queue:** An optional Redis service may be used for queuing background tasks and caching. For instance, if using Celery to schedule nightly summaries or pre-compute embeddings for large file uploads, Redis acts as the broker. It can also cache recent results to speed up responses. This component is not required for basic operation but is available for scaling and advanced use.

**Diagram:** *The following diagram illustrates the high-level architecture and data flows between components (UI, Agent, Model, Vector DB, Graph DB, etc.)*:

![Architecture Diagram](docs/architecture.png)

*(In the diagram, solid arrows show the primary request/response flow for a chat query, while dashed arrows show memory storage and retrieval operations.)*

## Installation

**Prerequisites:** You should have a machine with a CUDA-capable GPU (if using the default GPU-based model). Ensure you have Docker or LXC set up (this guide uses containerization for each component). For LXC, create containers for each service as described below. Alternatively, you can use Docker Compose with the provided `docker-compose.yml` (coming soon) to run everything.

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/yourname/local-ai-assistant.git
   cd local-ai-assistant
