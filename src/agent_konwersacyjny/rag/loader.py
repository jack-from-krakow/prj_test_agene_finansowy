"""Load documents from a folder and split them into overlapping chunks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class Chunk:
    source: str  # path relative to the docs folder
    index: int
    text: str
    heading: str | None = None

    @property
    def id(self) -> str:
        return f"{self.source}#{self.index}"


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="replace")


_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def _sections(text: str) -> list[tuple[str | None, list[str]]]:
    """Group paragraphs into markdown sections: (heading, paragraphs incl. the heading line)."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if m := _HEADING.match(para.splitlines()[0]):
            sections.append((m.group(1).strip(), []))
        sections[-1][1].append(para)
    return [(h, paras) for h, paras in sections if paras]


def split_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, str | None]]:
    """Paragraph-aware splitting within markdown sections, with character overlap.

    Returns (chunk_text, section_heading) pairs. Chunks never span two sections.
    """
    step = max(1, chunk_size - overlap)
    chunks: list[tuple[str, str | None]] = []
    for heading, paragraphs in _sections(text):
        pieces = [p[i : i + chunk_size] for p in paragraphs for i in range(0, len(p), step)]
        current: list[str] = []
        fresh = False  # whether `current` holds anything beyond the overlap tail
        for piece in pieces:
            if fresh and sum(len(c) for c in current) + len(piece) > chunk_size:
                chunks.append(("\n\n".join(current), heading))
                tail = current[-1][-overlap:] if overlap else ""
                current, fresh = ([tail] if tail else []), False
            current.append(piece)
            fresh = True
        if fresh:
            chunks.append(("\n\n".join(current), heading))
    return chunks


def load_chunks(root: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    if not root.exists():
        return []
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            text = read_document(path)
        except Exception:  # noqa: BLE001 - one broken file must not break indexing
            logger.exception("Failed to read %s", path)
            continue
        rel = path.relative_to(root).as_posix()
        for i, (piece, heading) in enumerate(split_text(text, chunk_size, overlap)):
            chunks.append(Chunk(source=rel, index=i, text=piece, heading=heading))
    return chunks
