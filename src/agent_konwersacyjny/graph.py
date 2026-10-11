"""The conversation graph: routes each user turn to whichever agent holds the conversation.

```
START -> start_turn -> [active agent] --handoff--> [target agent] -> ... -> record_episode -> END
```

The graph state keeps the *public* transcript only (user messages and the agents' visible
replies). Each agent run sees that transcript; its internal tool traffic stays inside the run,
so agents with different tool sets never see each other's tool calls.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Command

from agent_konwersacyjny.agents.factory import AgentTeam
from agent_konwersacyjny.agents.prompts import ORCHESTRATOR_RETURN_NOTE, SPECIALIST_CONVERSATION_NOTE
from agent_konwersacyjny.context import AppContext, current_thread_id
from agent_konwersacyjny.handoff import ORCHESTRATOR, Handoff, find_handoff
from agent_konwersacyjny.memory.episodic import record_episode

logger = logging.getLogger(__name__)

RECORD_EPISODE = "record_episode"


class ConversationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Agent that receives the next user message.
    active_agent: NotRequired[str]
    # The handoff that gave `active_agent` control (see `Handoff.as_dict`).
    handoff: NotRequired[dict[str, Any] | None]
    # Per-turn bookkeeping.
    turn_hops: NotRequired[int]
    turn_agents: NotRequired[list[str]]


def message_text(message: AnyMessage) -> str:
    return message.text.strip() if hasattr(message, "text") else str(message.content).strip()


def conversation_note(agent: str, handoff: dict[str, Any] | None) -> str | None:
    if not handoff or handoff.get("target") != agent:
        return None
    if agent == ORCHESTRATOR:
        if not handoff.get("pending_request"):
            return None
        return ORCHESTRATOR_RETURN_NOTE.format(
            source=handoff["source"], reason=handoff["reason"], pending=handoff["pending_request"]
        )
    return SPECIALIST_CONVERSATION_NOTE.format(source=handoff["source"], reason=handoff["reason"])


def build_conversation_graph(
    team: AgentTeam,
    *,
    max_handoffs_per_turn: int = 3,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    agent_names = list(team.agents)

    def start_turn(state: ConversationState) -> Command:
        active = state.get("active_agent") or ORCHESTRATOR
        if active not in team.agents:
            logger.warning("Unknown active agent %r, falling back to orchestrator", active)
            active = ORCHESTRATOR
        return Command(goto=active, update={"active_agent": active, "turn_hops": 0, "turn_agents": []})

    def make_agent_node(name: str):  # noqa: ANN202
        agent = team.agents[name]

        async def run_agent(
            state: ConversationState, config: RunnableConfig, runtime: Runtime[AppContext]
        ) -> Command:
            transcript = state["messages"]
            inputs: dict[str, Any] = {"messages": transcript}
            if note := conversation_note(name, state.get("handoff")):
                inputs["conversation_note"] = note
            result = await agent.ainvoke(inputs, config=config, context=runtime.context)

            seen = {m.id for m in transcript}
            new_messages = [m for m in result["messages"] if m.id not in seen]
            update: dict[str, Any] = {"turn_agents": [*state.get("turn_agents", []), name]}
            replies = [m for m in new_messages if isinstance(m, AIMessage) and message_text(m)]
            if replies:
                update["messages"] = [AIMessage(content=message_text(replies[-1]), name=name)]
            if name == ORCHESTRATOR:
                update["handoff"] = None  # a return note is consumed by one orchestrator run

            handoff = find_handoff(name, new_messages)
            if handoff is None:
                return Command(goto=RECORD_EPISODE, update=update)
            return await _route_handoff(handoff, state, runtime, update)

        run_agent.__name__ = f"run_{name}"
        return run_agent

    async def _route_handoff(
        handoff: Handoff, state: ConversationState, runtime: Runtime[AppContext], update: dict[str, Any]
    ) -> Command:
        if handoff.target not in team.agents:
            logger.warning("Ignoring handoff to unknown agent %r", handoff.target)
            return Command(goto=RECORD_EPISODE, update=update)
        update |= {"active_agent": handoff.target, "handoff": handoff.as_dict()}
        hops = state.get("turn_hops", 0) + 1
        update["turn_hops"] = hops

        if handoff.target == ORCHESTRATOR:
            if runtime.store is not None:
                await record_episode(
                    runtime.store,
                    user_id=runtime.context.user_id if runtime.context else "default",
                    thread_id=current_thread_id(),
                    kind="subagent_summary",
                    summary=handoff.reason,
                    agents=[handoff.source],
                )
            run_now = bool(handoff.pending_request)
        else:
            run_now = True

        if run_now and hops <= max_handoffs_per_turn:
            return Command(goto=handoff.target, update=update)
        if run_now:
            logger.warning("Handoff limit (%d) reached in one turn", max_handoffs_per_turn)
        return Command(goto=RECORD_EPISODE, update=update)

    async def record_turn(state: ConversationState, runtime: Runtime[AppContext]) -> dict[str, Any]:
        if runtime.store is None:
            return {}
        last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        last_reply = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        if last_user is None:
            return {}
        summary = f"User: {message_text(last_user)}"
        if last_reply is not None:
            summary += f"\nReply ({last_reply.name or 'assistant'}): {message_text(last_reply)}"
        await record_episode(
            runtime.store,
            user_id=runtime.context.user_id if runtime.context else "default",
            thread_id=current_thread_id(),
            kind="turn",
            summary=summary,
            agents=state.get("turn_agents", []),
        )
        return {}

    builder = StateGraph(ConversationState, context_schema=AppContext)
    builder.add_node("start_turn", start_turn, destinations=tuple(agent_names))
    for name in agent_names:
        builder.add_node(name, make_agent_node(name), destinations=(*agent_names, RECORD_EPISODE))
    builder.add_node(RECORD_EPISODE, record_turn)
    builder.add_edge(START, "start_turn")
    builder.add_edge(RECORD_EPISODE, END)
    return builder.compile(checkpointer=checkpointer, store=store, name="conversation")
