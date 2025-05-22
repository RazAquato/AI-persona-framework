| Component                     | Tool / Stack                                         | Why Chosen                                                                                       |
| ----------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Frontend UI**               | **Chainlit**                                         | Fast setup, tabs for sessions, tool integration, supports LangChain & MCP                        |
| **Backend / Agent Framework** | **LangChain (Python)**                               | Modular chains, memory, tool wrappers, supports RAG and future MCP                               |
| **Vector Store**              | **Qdrant**                                           | Best open-source vector DB right now: blazing fast, hybrid search, persistent, easy Docker setup |
| **Graph DB (optional layer)** | **Neo4j**                                            | To model linked facts, entities, and idea graphs (Zettelkasten-like)                             |
| **Relational DB**             | **PostgreSQL** (optional, not first priority)        | Use only if needed for structured logs/users/etc. later. Not critical now                        |
| **LLM Hosting**               | **Transformers + `gguf` via llama.cpp or Ollama**    | Local, GPU-accelerated model serving, swap models easily                                         |
| **Stable Diffusion**          | **ComfyUI in separate container**                    | You already like it, can be used via API or script from chatbot                                  |
| **Persona Logic**             | Custom in LangChain                                  | Prompt + memory per persona, extendable without code duplication                                 |
| **Knowledge Storage**         | **Qdrant (Semantic)** + **Neo4j (Relational Graph)** | Embeddings + relationship reasoning + reflection-ready                                           |
| **Notebook for Docs**         | Jupyter LXC later                                    | Not urgent, deferred per your input                                                              |
| **Tool Invocation**           | LangChain `Tool` API + MCP prep (not immediate)      | Simple to build now, ready for MCP or agent upgrade later                                        |
