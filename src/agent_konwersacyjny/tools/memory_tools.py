"""Tools giving agents access to semantic and episodic long-term memory."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool
from langgraph.store.base import BaseStore

from agent_konwersacyjny.memory.namespaces import episodic_ns, semantic_ns, user_id_from_runtime
from agent_konwersacyjny.memory.search import search_memories

FactCategory = Literal["user_profile", "preference", "domain_fact", "other"]


def make_memory_tools(store: BaseStore | None = None) -> list[BaseTool]:
    def _store(runtime: ToolRuntime) -> BaseStore:
        resolved = runtime.store or store
        if resolved is None:
            raise RuntimeError("No long-term memory store configured for this graph.")
        return resolved

    @tool
    async def remember_fact(fact: str, category: FactCategory, runtime: ToolRuntime) -> str:
        """Save a durable fact to long-term semantic memory.

        Use for stable information worth knowing in future sessions: who the user is,
        their preferences, definitions and facts about their business/finances.
        Do NOT store transient chit-chat or anything the user asked you not to keep.
        """
        key = uuid.uuid4().hex[:12]
        await _store(runtime).aput(
            semantic_ns(user_id_from_runtime(runtime)),
            key,
            {"fact": fact, "category": category, "created_at": datetime.now(UTC).isoformat()},
        )
        return f"Saved fact {key}."

    @tool
    async def recall_facts(query: str, runtime: ToolRuntime, limit: int = 5) -> str:
        """Search long-term semantic memory for facts relevant to `query`."""
        items = await search_memories(
            _store(runtime), semantic_ns(user_id_from_runtime(runtime)), query, limit
        )
        if not items:
            return "No matching facts."
        return "\n".join(f"- [{i.key}] ({i.value.get('category')}) {i.value.get('fact')}" for i in items)

    @tool
    async def forget_fact(key: str, runtime: ToolRuntime) -> str:
        """Delete a fact from semantic memory by its key (as shown by `recall_facts`)."""
        await _store(runtime).adelete(semantic_ns(user_id_from_runtime(runtime)), key)
        return f"Deleted fact {key}."

    @tool
    async def search_episodes(query: str, runtime: ToolRuntime, limit: int = 5) -> str:
        """Search episodic memory: summaries of earlier conversations and sub-agent work.

        Use when the user refers to something discussed before ("as last time", "that report").
        """
        items = await search_memories(
            _store(runtime), episodic_ns(user_id_from_runtime(runtime)), query, limit
        )
        if not items:
            return "No matching episodes."
        return "\n".join(
            f"- {i.value.get('created_at', '')[:16]} [{i.value.get('kind')}] "
            f"agents={','.join(i.value.get('agents', []))}: {i.value.get('summary')}"
            for i in items
        )

    return [remember_fact, recall_facts, forget_fact, search_episodes]
