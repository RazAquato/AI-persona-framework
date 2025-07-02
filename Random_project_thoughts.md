Alternatively the model could be run in a chrooted enviroment with option to create code and train
For example tools like
/write_file path="..." content="...
/exec_python code="..."
/restart_component name="engine"
/install_package name="pandas"
/schema_change db="pg" sql="ALTER TABLE facts ADD COLUMN confidence FLOAT;"
/shell "shell command"

For the model to be able to "think without input" we could also run a scheduler or background service that lets the LLM
re-analyze logs and metadata
self-reflect or propose schema changes
decide what tools/personas to update
propose patch diffs to its own codebase
code like "self_reflect_and_optimize()"

important: we should log any tool usage and file edits (toolname, input, return) so the agent can self-correct if it creates faulty code
maybe even one LLM could use another LLM to perform coding for it
/initiate LLM="deepseek coder R1", prompt="..."

---
Self-tuning a GGUF
Even though a GGUF model is read-only and cannot fine-tune itself:

It can still call external tools (like file editors, DB schema updaters, even code compilers)
It can reflect on logs, change its prompts, generate new personas, spawn helper agents
It can build complex feedback loops and simulate learning without changing weights
Essentially, it becomes a powerful orchestrator and planner, even without modifying itself internally.

store trained adapter checkpoints in models/adapters

Optional: Hybrid Mode (GGUF + Training Tools)
Something like: 
/train_adapter --base Evathene --dataset /echo_corpus/user_9999.json --output /models/evathene_v2.adapter


---
when memory system is operational, we could expand with voice
F.ex RVC v2, a retrieval-Based Voice Conversion v2 is a system that lets you:
- Clone a voice using ~10 minutes of audio
- Convert any input voice (even TTS or your own) into that voice
- Use it for roleplay AIs, assistants, or emotional avatars

https://github.com/RVC-Project - MIT licence


---

Further: if journalling on an ipad (for example using goodnotes or notability or similar) we could also implement an OCR to convert the notes to textfiles and implement this in LLM traning
personal goal: create one LLM that works as a shrink, one LLM that works as a personal trainer and one for roleplay
question is whether the different personas should be able to access notes across. for example if I miss workouts for a while, the shrink should be able to pick up on that.
current thought: tagging notes so the LLMs can access notes if they are tagged.

example:
"01.01.2025 - Journal entry
Today I skipped jogging because it was raining"  
tags: workout, shrink

"02.01.2025 - Journal entry
I felt lonely today"
tags: shrink, roleplay

etc.


