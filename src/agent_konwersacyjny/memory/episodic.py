"""Episodic memory: what happened in past turns/sessions, searchable later."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import BaseStore

from agent_konwersacyjny.memory.namespaces import episodic_ns

MAX_FIELD_CHARS = 1500


def _clip(text: str) -> str:
    return text if len(text) <= MAX_FIELD_CHARS else text[: MAX_FIELD_CHARS - 1] + "…"


async def record_episode(
    store: BaseStore,
    *,
    user_id: str,
    thread_id: str,
    kind: str,
    summary: str,
    agents: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Persist one episode. `kind` is e.g. `turn` or `subagent_summary`."""
    key = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    value = {
        "kind": kind,
        "thread_id": thread_id,
        "summary": _clip(summary),
        "agents": agents or [],
        "created_at": datetime.now(UTC).isoformat(),
        **(extra or {}),
    }
    await store.aput(episodic_ns(user_id), key, value)
    return key
