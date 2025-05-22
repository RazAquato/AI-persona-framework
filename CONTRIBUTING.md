# Contributing to AI Assistant Framework

Thank you for your interest in contributing!

This is an open-source framework for building a modular, memory-capable AI assistant that can evolve over time and simulate digital echoes of its users. The goal is to empower individuals to run intelligent AI systems locally, on their own terms.

---

## What You Can Contribute

- Core logic improvements (memory handling, routing, inference)
- Memory modules (e.g., additional vector or graph backends)
- Agent personalities (JSON configurations with emotional profiles)
- Prompt construction improvements
- Topic and sentiment analysis enhancements
- Tools: image generation, sandboxed code, research modules
- Web interfaces (Streamlit, FastAPI, Chainlit, etc.)
- Documentation and examples
- Bug fixes and code cleanup

---

## Getting Started

1. Clone the repository:

```
git clone https://github.com/yourname/ai-assistant.git
cd ai-assistant
```

2. Create a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

3. Install the dependencies:

```
pip install -r requirements.txt
```

4. Create the `.env` file from the template and set up the following:

- PostgreSQL with `pgvector` extension
- Qdrant vector store
- Neo4j graph database
- Local LLM server (optional, e.g., llama.cpp on port 8080)

---

## Contribution Guidelines

- Use clear, descriptive commit messages
- Keep pull requests focused on a single feature or fix
- Follow the existing file and folder structure
- Add comments and docstrings where needed
- Avoid hardcoding config values (use the `.env` file)

---

## Pull Request Checklist

- [ ] You have tested your code
- [ ] Your code does not break existing functionality
- [ ] Any configuration is handled through `.env` or JSON files
- [ ] You described your changes clearly in the PR

---

## Code of Conduct

We aim to foster an open, welcoming environment. Please be respectful and constructive when communicating. Discrimination, harassment, or hostile behavior will not be tolerated.

---

## Questions or Ideas?

Open an issue on GitHub or join the discussions tab to ask questions, propose features, or get help.

Thank you for helping build an ethical, locally-owned AI future.
