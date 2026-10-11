from agent_konwersacyjny.rag.knowledge_base import KnowledgeBase
from agent_konwersacyjny.rag.loader import split_text
from agent_konwersacyjny.tools.rag_tools import make_rag_tools


def test_split_respects_size_and_tracks_headings():
    text = "# A\n\n" + "\n\n".join(f"akapit {i} " * 20 for i in range(10))
    chunks = split_text(text, chunk_size=300, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 30 + 4 for c, _ in chunks)
    assert all(h == "A" for _, h in chunks)


def test_search_finds_relevant_passage(settings):
    kb = KnowledgeBase(settings.docs_dir, chunk_size=80, overlap=0)
    hits = kb.search("kto akceptuje podróże zagraniczne", k=1)
    assert hits and "CFO" in hits[0].chunk.text
    assert hits[0].chunk.source == "polityka.md"


def test_index_refreshes_when_files_change(settings):
    kb = KnowledgeBase(settings.docs_dir)
    assert kb.sources == ["polityka.md"]
    (settings.docs_dir / "nowy.txt").write_text("Leasing samochodów", encoding="utf-8")
    assert "nowy.txt" in kb.sources


def test_tool_output_has_citations(settings):
    tools = {t.name: t for t in make_rag_tools(KnowledgeBase(settings.docs_dir))}
    out = tools["search_knowledge_base"].invoke({"query": "zarząd zatwierdza wydatki"})
    assert "[polityka.md#0]" in out
    assert tools["search_knowledge_base"].invoke({"query": "kryptowaluty"}) == "No relevant passages found."
