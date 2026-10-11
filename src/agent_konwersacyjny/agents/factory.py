"""Builds the Orchestrator and the specialist agents as deepagents graphs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from deepagents import SubAgent, create_deep_agent
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from agent_konwersacyjny.agents.prompts import (
    FINANCE_PROMPT,
    ORCHESTRATOR_PROMPT,
    RAG_PROMPT,
    TASK_MODE_SUFFIX,
)
from agent_konwersacyjny.backends import (
    PROCEDURAL_MEMORY_PATH,
    STATIC_INSTRUCTIONS_PATH,
    build_backend,
    default_permissions,
)
from agent_konwersacyjny.config import Settings
from agent_konwersacyjny.context import AppContext
from agent_konwersacyjny.handoff import (
    ORCHESTRATOR,
    ConversationRoleMiddleware,
    make_handoff_tool,
    make_return_tool,
)
from agent_konwersacyjny.models import build_model
from agent_konwersacyjny.rag.knowledge_base import KnowledgeBase
from agent_konwersacyjny.tools.calculator import calculate
from agent_konwersacyjny.tools.finance_tools import FinanceData, make_finance_tools
from agent_konwersacyjny.tools.memory_tools import make_memory_tools
from agent_konwersacyjny.tools.rag_tools import make_rag_tools

ModelFactory = Callable[[str], BaseChatModel]


@dataclass
class SpecialistSpec:
    name: str
    description: str
    system_prompt: str
    tools: list[BaseTool]


@dataclass
class AgentTeam:
    """Compiled agents keyed by name; `ORCHESTRATOR` is always present."""

    agents: dict[str, CompiledStateGraph]
    specialists: dict[str, SpecialistSpec] = field(default_factory=dict)


def specialist_specs(
    settings: Settings, mcp_tools: dict[str, list[BaseTool]], embeddings: Embeddings | None = None
) -> list[SpecialistSpec]:
    kb = KnowledgeBase(settings.docs_dir, settings.chunk_size, settings.chunk_overlap, embeddings)
    finance = FinanceData(settings.finance_dir)
    return [
        SpecialistSpec(
            name="rag",
            description="Answers questions from the local document knowledge base, with citations.",
            system_prompt=RAG_PROMPT,
            tools=[*make_rag_tools(kb), *mcp_tools.get("rag", [])],
        ),
        SpecialistSpec(
            name="finance",
            description=(
                "Analyses financial data from local Excel/CSV/txt/md files: budgets, costs, revenue, "
                "variances, KPIs; computes exact figures."
            ),
            system_prompt=FINANCE_PROMPT,
            tools=[*make_finance_tools(finance), calculate, *mcp_tools.get("finance", [])],
        ),
    ]


def build_team(
    settings: Settings,
    store: BaseStore | None,
    *,
    mcp_tools: dict[str, list[BaseTool]] | None = None,
    model_factory: ModelFactory | None = None,
    embeddings: Embeddings | None = None,
) -> AgentTeam:
    """Create every agent. `mcp_tools` maps agent name (or `*` for all) to extra tools."""
    mcp_tools = mcp_tools or {}
    shared_mcp = mcp_tools.get("*", [])
    make_model = model_factory or (lambda name: build_model(settings, name))
    backend = build_backend(settings, store)
    memory_tools = make_memory_tools(store)
    common = {
        "backend": backend,
        "store": store,
        "memory": [STATIC_INSTRUCTIONS_PATH, PROCEDURAL_MEMORY_PATH],
        "skills": ["/skills/"],
        "permissions": default_permissions(),
        "context_schema": AppContext,
    }

    specs = specialist_specs(settings, mcp_tools, embeddings)
    agents: dict[str, CompiledStateGraph] = {}
    for spec in specs:
        agents[spec.name] = create_deep_agent(
            model=make_model(spec.name),
            tools=[*spec.tools, *memory_tools, *shared_mcp, make_return_tool()],
            system_prompt=spec.system_prompt,
            middleware=[ConversationRoleMiddleware()],
            name=spec.name,
            **common,
        )

    agents[ORCHESTRATOR] = create_deep_agent(
        model=make_model(ORCHESTRATOR),
        tools=[
            make_handoff_tool({s.name: s.description for s in specs}),
            *memory_tools,
            calculate,
            *shared_mcp,
            *mcp_tools.get(ORCHESTRATOR, []),
        ],
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=_as_task_subagents(specs, make_model),
        middleware=[ConversationRoleMiddleware()],
        name=ORCHESTRATOR,
        **common,
    )
    return AgentTeam(agents=agents, specialists={s.name: s for s in specs})


def _as_task_subagents(specs: Sequence[SpecialistSpec], make_model: ModelFactory) -> list[SubAgent]:
    """The same specialists, exposed to the Orchestrator's `task` tool for one-shot jobs."""
    return [
        SubAgent(
            name=spec.name,
            description=spec.description,
            system_prompt=spec.system_prompt + TASK_MODE_SUFFIX,
            tools=list(spec.tools),
            model=make_model(spec.name),
        )
        for spec in specs
    ]
