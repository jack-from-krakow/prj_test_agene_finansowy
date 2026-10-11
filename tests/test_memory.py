from langgraph.store.memory import InMemoryStore

from agent_konwersacyjny.memory.episodic import record_episode
from agent_konwersacyjny.memory.namespaces import episodic_ns, safe, semantic_ns
from agent_konwersacyjny.memory.search import search_memories


async def test_keyword_search_without_index():
    store = InMemoryStore()
    await store.aput(semantic_ns("u1"), "a", {"fact": "Użytkownik raportuje w EUR", "category": "preference"})
    await store.aput(semantic_ns("u1"), "b", {"fact": "Firma ma 3 działy", "category": "domain_fact"})
    hits = await search_memories(store, semantic_ns("u1"), "raportuje EUR")
    assert [h.key for h in hits] == ["a"]
    assert await search_memories(store, semantic_ns("u2"), "EUR") == []


async def test_episodes_are_scoped_per_user():
    store = InMemoryStore()
    await record_episode(
        store, user_id="u1", thread_id="t", kind="turn", summary="analiza budżetu marketingu"
    )
    assert len(await store.asearch(episodic_ns("u1"))) == 1
    assert await store.asearch(episodic_ns("u2")) == []


def test_namespace_sanitising():
    assert safe("jan kowalski/../x") == "jan_kowalski_.._x"
