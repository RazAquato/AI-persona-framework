# Memory System Overview

This document describes the modular memory system of the AI Assistant Framework. The memory system is split into layers that store and retrieve different types of knowledge, enabling the assistant to simulate persistence, relevance, and emotional continuity over time.

---

## Memory Layers

| Layer              | Module                  | Backend       | Purpose                                                   |
|--------------------|--------------------------|---------------|-----------------------------------------------------------|
| Short-Term Memory  | `memory/buffer.py`       | In-memory     | Holds recent messages from current session only           |
| Vector Memory      | `memory/vector_store.py` | Qdrant        | Stores all past messages as embeddings for similarity search |
| Structured Memory  | `memory/fact_store.py`   | PostgreSQL    | Stores facts, logs, and user info in structured format    |
| Graph Memory       | `memory/topic_graph.py`  | Neo4j         | Stores entities, topics, relationships, and preferences   |

---

## How Memory Works

### 1. User Sends a Message

- Message is captured via interface (`interface/cli_chat.py`, API, etc.)
- Passed to `core/engine.py`

### 2. Memory Retrieval

`engine.py` collects contextual memory:

- Short-term memory buffer (recent turns)
- Similar past messages via Qdrant (vector similarity)
- User facts and preferences from PostgreSQL
- Entity/topic links from Neo4j

These are assembled into a prompt in `context_builder.py`.

### 3. LLM Prompt Construction

- Persona profile loaded from `agents/personality_config.json`
- Prompt includes memory snippets, personality, and current input
- Sent to local or remote LLM backend (e.g., llama.cpp)

### 4. AI Response

- Response is returned to the user through the interface

### 5. Memory Update

After LLM responds:

- Message and reply are embedded and stored in Qdrant
- Raw log and topics are stored in PostgreSQL
- Topic links and user preferences are updated in Neo4j

---

## Example

User says:  
> “I hate Tottenham, but I love watching Manchester United.”

System actions:

- Qdrant embeds and stores the message
- PostgreSQL logs message and tags with:
  - Sentiment: Tottenham -0.8, Manchester United +0.9
  - Topics: `["football", "sports"]`
- Neo4j updates:
  - `(:User)-[:DISLIKES]->(:Topic {name: "Tottenham"})`
  - `(:User)-[:LIKES]->(:Topic {name: "Manchester United"})`
  - `(:Topic)-[:SUBTOPIC_OF]->(:Topic {name: "football"})`

---

## Future Considerations

- Add topic clustering using HDBSCAN on vector space
- Build summarizers to create episodic memory
- Apply decay function to rarely referenced memories
