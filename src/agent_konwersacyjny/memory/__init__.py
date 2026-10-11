"""Long-term memory: semantic facts, episodes, and the namespaces backing procedural memory.

Memory taxonomy used across the project:

* conversation history - LangGraph checkpointer (one thread per session).
* session memory       - thread state + the per-thread `/workspace/` files.
* semantic memory      - facts about the user/domain in the store (`semantic`).
* episodic memory      - summaries of past turns and sub-agent work (`episodic`).
* procedural memory    - `/memories/AGENTS.md` (editable by agents) + `skills/`.
"""
