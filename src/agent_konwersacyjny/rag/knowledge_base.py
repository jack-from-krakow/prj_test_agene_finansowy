"""Hybrid retrieval: BM25 (free, local) plus optional embeddings, fused with RRF."""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from agent_konwersacyjny.rag.loader import Chunk, load_chunks

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+", re.UNICODE)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    # Crude prefix stemming helps with Polish inflection (przychód/przychody/przychodów).
    return [t[:6] for t in _WORD.findall(text.lower()) if len(t) > 1]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class KnowledgeBase:
    def __init__(
        self, root: Path, chunk_size: int = 1200, overlap: int = 150, embeddings: Embeddings | None = None
    ) -> None:
        self.root = root
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embeddings = embeddings
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._tokens: list[set[str]] = []
        self._bm25: BM25Okapi | None = None
        self._vectors: np.ndarray | None = None
        self._signature: tuple | None = None

    def _current_signature(self) -> tuple:
        if not self.root.exists():
            return ()
        return tuple(sorted((str(p), p.stat().st_mtime) for p in self.root.rglob("*") if p.is_file()))

    def reindex(self, force: bool = False) -> int:
        with self._lock:
            signature = self._current_signature()
            if not force and signature == self._signature:
                return len(self._chunks)
            self._chunks = load_chunks(self.root, self.chunk_size, self.overlap)
            corpus = [tokenize(f"{c.heading or ''} {c.text}") for c in self._chunks]
            self._tokens = [set(doc) for doc in corpus]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._vectors = None
            if self.embeddings and self._chunks:
                vectors = np.array(self.embeddings.embed_documents([c.text for c in self._chunks]))
                self._vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            self._signature = signature
            logger.info("Indexed %d chunks from %s", len(self._chunks), self.root)
            return len(self._chunks)

    @property
    def sources(self) -> list[str]:
        self.reindex()
        return sorted({c.source for c in self._chunks})

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.reindex()  # cheap no-op unless files changed
        if not self._chunks or self._bm25 is None:
            return []
        rankings: list[list[int]] = []
        query_tokens = tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)
        # BM25 IDF can be ~0 or negative in tiny corpora, so relevance = shares a query term.
        matching = [i for i, tokens in enumerate(self._tokens) if not tokens.isdisjoint(query_tokens)]
        rankings.append(sorted(matching, key=lambda i: bm25_scores[i], reverse=True))
        if self._vectors is not None and self.embeddings is not None:
            q = np.array(self.embeddings.embed_query(query))
            sims = self._vectors @ (q / np.linalg.norm(q))
            rankings.append(list(np.argsort(-sims)[: max(k * 4, 20)]))
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(self._chunks[i], score) for i, score in best]
