"""Application runtime: storage, MCP tools, agents and the conversation graph wired together."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

from agent_konwersacyjny.agents.factory import ModelFactory, build_team
from agent_konwersacyjny.config import Settings, get_settings
from agent_konwersacyjny.context import AppContext
from agent_konwersacyjny.graph import build_conversation_graph, message_text
from agent_konwersacyjny.handoff import ORCHESTRATOR
from agent_konwersacyjny.mcp import load_mcp_tools

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentReply:
    agent: str
    text: str


@dataclass(frozen=True)
class TurnResult:
    replies: list[AgentReply]
    active_agent: str


class AgentApp:
    def __init__(self, graph: CompiledStateGraph, store: BaseStore, settings: Settings) -> None:
        self.graph = graph
        self.store = store
        self.settings = settings

    @staticmethod
    def new_thread_id() -> str:
        return uuid.uuid4().hex

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}

    async def chat(self, text: str, *, thread_id: str, user_id: str | None = None) -> TurnResult:
        """Send one user message; returns the visible replies produced in this turn."""
        context = AppContext(user_id=user_id or self.settings.default_user_id)
        human = HumanMessage(content=text, id=uuid.uuid4().hex)
        state = await self.graph.ainvoke(
            {"messages": [human]}, config=self._config(thread_id), context=context
        )
        messages = state["messages"]
        start = next(i for i, m in enumerate(messages) if m.id == human.id)
        replies = [
            AgentReply(agent=m.name or ORCHESTRATOR, text=message_text(m))
            for m in messages[start + 1 :]
            if isinstance(m, AIMessage)
        ]
        return TurnResult(replies=replies, active_agent=state.get("active_agent", ORCHESTRATOR))

    async def history(self, thread_id: str) -> list[tuple[str, str]]:
        """Visible transcript of a thread as (speaker, text) pairs."""
        snapshot = await self.graph.aget_state(self._config(thread_id))
        out = []
        for m in snapshot.values.get("messages", []):
            speaker = "user" if isinstance(m, HumanMessage) else (m.name or ORCHESTRATOR)
            out.append((speaker, message_text(m)))
        return out


def build_embeddings(settings: Settings) -> Embeddings | None:
    if not settings.embeddings_model:
        return None
    from langchain.embeddings import init_embeddings

    return init_embeddings(settings.embeddings_model)


@asynccontextmanager
async def create_app(
    settings: Settings | None = None,
    *,
    model_factory: ModelFactory | None = None,
    load_mcp: bool = True,
) -> AsyncIterator[AgentApp]:
    """Open persistent storage and build the agent team. Use as `async with create_app() as app`."""
    settings = settings or get_settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    embeddings = build_embeddings(settings)
    index = None
    if embeddings is not None:
        dims = len(await embeddings.aembed_query("probe"))
        index = {"dims": dims, "embed": embeddings, "fields": ["fact", "summary"]}

    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db))
        )
        store = await stack.enter_async_context(
            AsyncSqliteStore.from_conn_string(str(settings.store_db), index=index)
        )
        await store.setup()
        mcp_tools = await load_mcp_tools(settings.mcp_config) if load_mcp else {}
        team = build_team(
            settings, store, mcp_tools=mcp_tools, model_factory=model_factory, embeddings=embeddings
        )
        graph = build_conversation_graph(
            team,
            max_handoffs_per_turn=settings.max_handoffs_per_turn,
            checkpointer=checkpointer,
            store=store,
        )
        yield AgentApp(graph, store, settings)
