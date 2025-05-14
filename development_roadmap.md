DEVELOPMENT ROADMAP (PHASED BUILD)
✅ Phase 1: Foundation (Local Dev + POC)
Set up LXC container:
ai-core LXC (Ubuntu), install Python 3.10+, Docker, Git

Install and run:

 Qdrant in Docker (docker run -p 6333:6333 qdrant/qdrant)

 LangChain with Python (venv): pip install langchain qdrant-client openai transformers

 Chainlit: pip install chainlit, test with hello world

 Optional: Try Ollama if you want easy model swapping: ollama run llama3

Model POC:

 Load Athene-V2-Chat or small LLaMA 3.1 8B in Transformers or llama.cpp

 Try a basic script: load model → ask question → get response

✅ Phase 2: Memory and Persona
Vector store setup (Qdrant):

 Create a memory collection (can hold all embedded memory)

 Use LangChain’s QdrantVectorStore to handle:

Storing facts

Retrieving relevant memory on user query

Persona system (LangChain):

 Define YAML/JSON for each persona: name, style, system prompt, etc.

 LangChain ConversationChain per persona

 Store session context in RAM (or optionally in Qdrant)

✅ Phase 3: Chainlit Frontend
Install Chainlit UI:

 Connect LangChain agent to Chainlit (supports tabbed UI, tool buttons, etc.)

 Add persona selector on startup

 Add /generate <prompt> as a chat command

✅ Phase 4: Image Generation
Set up ComfyUI in LXC (e.g., sd-core):

 Launch locally on port (e.g. 8188)

 Add API endpoint to generate from prompt

Add LangChain Tool:

 A simple Tool(name="generate_image", func=generate_image_api_call)

 AI can return /<toolname>: ... or wait for user /generate ...

✅ Phase 5: Long-term Memory + Notes
Enable fact extraction after chats:

 Use langchain.text_splitter + EmbeddingRetriever to parse and embed conversations

 Group topics (e.g., “hardware”, “philosophy”) and chunk embeddings under topic labels

[Optional] Neo4j Integration:

 Build a secondary process to map entities/facts → graph nodes

 Add entity relationships like “John likes topic: AI Ethics”

✅ Phase 6: Future-Proofing (Optional)
LangChain Agents + Tool Wrapping:

 Define Tools in LangChain (e.g., SD, weather, email reply)

 Add MCP prep: track community updates, test when needed

LLM finetuning pipeline (later):

 Collect user-specific data

 Add a LoRA finetune prep step (could be used across models)
