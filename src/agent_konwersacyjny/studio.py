"""Graph factory for `langgraph dev` / LangGraph Studio (see `langgraph.json`).

The LangGraph server provides the checkpointer and store, so none are created here.
"""

from __future__ import annotations

from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph

from agent_konwersacyjny.agents.factory import build_team
from agent_konwersacyjny.app import build_embeddings
from agent_konwersacyjny.config import get_settings
from agent_konwersacyjny.graph import build_conversation_graph
from agent_konwersacyjny.mcp import load_mcp_tools


async def make_graph() -> CompiledStateGraph:
    load_dotenv()
    settings = get_settings()
    team = build_team(
        settings,
        store=None,
        mcp_tools=await load_mcp_tools(settings.mcp_config),
        embeddings=build_embeddings(settings),
    )
    return build_conversation_graph(team, max_handoffs_per_turn=settings.max_handoffs_per_turn)
