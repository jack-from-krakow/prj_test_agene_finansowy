"""Store namespaces. All long-term memory is scoped by user id."""

from __future__ import annotations

import re
from typing import Any

from agent_konwersacyjny.context import AppContext, current_thread_id

_UNSAFE = re.compile(r"[^A-Za-z0-9\-_.@+:~]")


def safe(component: str) -> str:
    """Make a value usable as a store namespace component."""
    return _UNSAFE.sub("_", component) or "_"


def user_id_from_runtime(runtime: Any, default: str = "default") -> str:
    ctx = getattr(runtime, "context", None)
    if isinstance(ctx, AppContext):
        return ctx.user_id
    if isinstance(ctx, dict) and ctx.get("user_id"):
        return str(ctx["user_id"])
    return default


def semantic_ns(user_id: str) -> tuple[str, ...]:
    return (safe(user_id), "semantic")


def episodic_ns(user_id: str) -> tuple[str, ...]:
    return (safe(user_id), "episodic")


def procedural_fs_ns(runtime: Any) -> tuple[str, ...]:
    """Backs `/memories/` - persistent across all of the user's sessions."""
    return (safe(user_id_from_runtime(runtime)), "fs", "memories")


def workspace_fs_ns(runtime: Any) -> tuple[str, ...]:
    """Backs `/workspace/` - persistent within one session (thread)."""
    return (safe(user_id_from_runtime(runtime)), "fs", "workspace", safe(current_thread_id()))
