[CHAINLIT Web UI]
        ↓
[LangChain Backend/Router] ←→ [LLM via Ollama/Transformers]
        ↓                        ↑
[Tool APIs: SD, etc.]           |
        ↓                       |
[Vector Store: Qdrant] ←→ [Embeddings / Memory]
        ↓
[Graph DB: Neo4j] (optional now)
