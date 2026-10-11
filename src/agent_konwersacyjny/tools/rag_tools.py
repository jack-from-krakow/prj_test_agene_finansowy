"""Tools exposing the local knowledge base to the RAG agent."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from agent_konwersacyjny.rag.knowledge_base import KnowledgeBase


def make_rag_tools(kb: KnowledgeBase) -> list[BaseTool]:
    @tool
    def search_knowledge_base(query: str, k: int = 5) -> str:
        """Search the local document base. Returns passages with [source#chunk] citations.

        Run several focused queries (synonyms, Polish and English terms) rather than one
        broad query. Cite sources in the answer using the bracketed ids.
        """
        hits = kb.search(query, k=max(1, min(k, 15)))
        if not hits:
            return "No relevant passages found."
        return "\n\n---\n\n".join(
            f"[{h.chunk.id}]"
            + (f" (section: {h.chunk.heading})" if h.chunk.heading else "")
            + f"\n{h.chunk.text}"
            for h in hits
        )

    @tool
    def list_documents() -> str:
        """List documents available in the knowledge base."""
        sources = kb.sources
        return "\n".join(f"- {s}" for s in sources) if sources else "The knowledge base is empty."

    @tool
    def reindex_knowledge_base() -> str:
        """Rebuild the search index (use after documents were added or changed)."""
        return f"Indexed {kb.reindex(force=True)} chunks."

    return [search_knowledge_base, list_documents, reindex_knowledge_base]
