"""Store search that works with or without a vector index configured on the store."""

from __future__ import annotations

import re
from typing import Any

from langgraph.store.base import BaseStore, SearchItem

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 2}


def _keyword_score(query: str, value: dict[str, Any]) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    doc = _tokens(" ".join(str(v) for v in value.values()))
    return len(q & doc) / len(q)


async def search_memories(
    store: BaseStore, namespace: tuple[str, ...], query: str | None, limit: int = 5
) -> list[SearchItem]:
    """Semantic search when the store has an embedding index, keyword overlap otherwise."""
    if not query:
        return await store.asearch(namespace, limit=limit)
    if getattr(store, "index_config", None):
        return await store.asearch(namespace, query=query, limit=limit)
    candidates = await store.asearch(namespace, limit=1000)
    scored = [(item, _keyword_score(query, item.value)) for item in candidates]
    scored = [(item, s) for item, s in scored if s > 0]
    scored.sort(key=lambda pair: (pair[1], pair[0].updated_at), reverse=True)
    for item, s in scored:
        item.score = s
    return [item for item, _ in scored[:limit]]
