"""Conversation handoffs between the Orchestrator and the specialist agents.

Two delegation modes exist side by side:

* **task** (built into deepagents): the Orchestrator sends a one-shot, isolated task to a
  sub-agent and gets a result back; the user never talks to the sub-agent.
* **handoff** (this module): the Orchestrator passes *the conversation* to a specialist.
  The specialist talks to the user directly across turns (clarify scope, agree on the
  approach, deliver, summarize) and finally returns control with a summary.

Handoff tools are `return_direct`, so the agent's loop ends right after calling one; the
conversation graph then reads the call from the transcript and routes to the target.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

ORCHESTRATOR = "orchestrator"
HANDOFF_TOOL = "handoff_to_agent"
RETURN_TOOL = "return_to_orchestrator"
HANDOFF_TOOL_NAMES = {HANDOFF_TOOL, RETURN_TOOL}


@dataclass(frozen=True)
class Handoff:
    source: str
    target: str
    reason: str
    # Set on returns: something the user asked that the Orchestrator must answer now.
    pending_request: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "reason": self.reason,
            "pending_request": self.pending_request,
        }


def make_handoff_tool(specialists: dict[str, str]) -> BaseTool:
    """Orchestrator tool. `specialists` maps agent name -> one-line description."""
    names = tuple(specialists)
    roster = "\n".join(f"- {name}: {desc}" for name, desc in specialists.items())

    def handoff_to_agent(agent: str, reason: str) -> str:
        if agent not in specialists:
            return f"Unknown agent '{agent}'. Choose one of: {', '.join(names)}."
        return f"Conversation handed over to '{agent}'."

    return StructuredTool.from_function(
        func=handoff_to_agent,
        name=HANDOFF_TOOL,
        description=(
            "Hand the conversation over to a specialist who will talk to the user directly "
            "(clarify scope, agree the approach, deliver, summarize) until the work is done.\n"
            "Use for multi-step or ambiguous requests in the specialist's domain. For a quick "
            "self-contained question, delegate with `task` instead.\n"
            "Call this tool alone, as the last action of your turn. `reason` must carry "
            f"everything the specialist needs: the user's goal and known constraints.\n{roster}"
        ),
        args_schema=_handoff_schema(names),
        return_direct=True,
    )


def _handoff_schema(names: Sequence[str]) -> type:
    from pydantic import BaseModel, Field, create_model

    agent_type = Literal[tuple(names)]  # type: ignore[valid-type]
    return create_model(
        "HandoffArgs",
        __base__=BaseModel,
        agent=(agent_type, Field(description="Specialist to take over the conversation.")),
        reason=(str, Field(description="User's goal, context and constraints for the specialist.")),
    )


def make_return_tool() -> BaseTool:
    def return_to_orchestrator(summary: str, pending_request: str | None = None) -> str:
        return "Conversation returned to the orchestrator."

    return StructuredTool.from_function(
        func=return_to_orchestrator,
        name=RETURN_TOOL,
        description=(
            "Give the conversation back to the Orchestrator. Call it alone, when your work is "
            "delivered and summarized to the user, or when the user asks for something outside "
            "your domain.\n"
            "- summary: concise record of what was asked, decided and delivered (kept in "
            "episodic memory).\n"
            "- pending_request: ONLY if the user is waiting for an answer you cannot give "
            "(out of your domain); the Orchestrator will answer it immediately."
        ),
        return_direct=True,
    )


def find_handoff(agent_name: str, new_messages: Sequence[AnyMessage]) -> Handoff | None:
    """Return the last handoff call executed in `new_messages`, if any."""
    executed = {m.tool_call_id for m in new_messages if isinstance(m, ToolMessage)}
    for message in reversed(new_messages):
        if not isinstance(message, AIMessage):
            continue
        for call in reversed(message.tool_calls):
            if call["name"] not in HANDOFF_TOOL_NAMES or call["id"] not in executed:
                continue
            args = call["args"]
            if call["name"] == HANDOFF_TOOL:
                return Handoff(agent_name, args.get("agent", ""), args.get("reason", ""))
            return Handoff(agent_name, ORCHESTRATOR, args.get("summary", ""), args.get("pending_request"))
    return None


# --- Prompt injection of the conversation role -------------------------------------


class ConversationRoleState(AgentState):
    conversation_note: NotRequired[str]


class ConversationRoleMiddleware(AgentMiddleware):
    """Appends a per-turn note (who holds the conversation and why) to the system prompt."""

    state_schema = ConversationRoleState

    def _with_note(self, request: ModelRequest) -> ModelRequest:
        note = request.state.get("conversation_note")
        if not note:
            return request
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        blocks.append({"type": "text", "text": f"\n\n<conversation_role>\n{note}\n</conversation_role>"})
        return request.override(system_message=SystemMessage(content_blocks=blocks))

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        return handler(self._with_note(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._with_note(request))
