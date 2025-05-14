[CHAINLIT Web UI]
        ↓
[LangChain Backend/Router] ←→ [LLM via Ollama/Transformers]
        ↓                        ↑
[Tool APIs: SD, etc.]           |
        ↓                       |
[Vector Store: Qdrant] ←→ [Embeddings / Memory]
        ↓
[Graph DB: Neo4j] (optional now)



-------
ALTERNATIVE:
                              +------------------+
                              |   Discord Bot    | ← User interface
                              +--------+---------+
                                       |
                     +----------------v------------------+
                     |         Chatbot Router            | ← Detects user, context, target personality
                     +----------------+------------------+
                                      |
             +------------------------+------------------------+
             |                         |                        |
  +----------v--------+     +----------v----------+   +--------v----------+
  | LLM Personality 1 |     | LLM Personality 2   |   | LLM Personality N |
  |   ("Alfred")      |     |   ("Emma")          |   |   ("Tammy")       |
  +-------------------+     +---------------------+   +-------------------+
             |                        |                         |
             +------------------------+-------------------------+
                                      |
                          +-----------v----------+
                          |   Unified Database   | ← Users, logs, memory, personality config
                          +-----------+----------+
                                      |
                      +---------------+---------------+
                      | External Tools API Layer      | ← SD, math, DB, file tools
                      +-------------------------------+
