"""Per-invocation runtime context shared by the conversation graph and every agent."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.config import get_config


@dataclass(frozen=True)
class AppContext:
    """Passed as `context=` on every invocation; scopes long-term memory per user."""

    user_id: str = "default"


def current_thread_id(default: str = "no-thread") -> str:
    """Thread id of the running graph (the conversation/session id)."""
    try:
        return str(get_config().get("configurable", {}).get("thread_id") or default)
    except RuntimeError:
        return default
