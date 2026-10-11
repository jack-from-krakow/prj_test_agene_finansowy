"""End-to-end conversation flows with scripted models (no API calls)."""

from __future__ import annotations

import pytest
from fakes import ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agent_konwersacyjny.agents.factory import build_team
from agent_konwersacyjny.app import create_app
from agent_konwersacyjny.context import AppContext
from agent_konwersacyjny.graph import build_conversation_graph
from agent_konwersacyjny.memory.namespaces import episodic_ns, semantic_ns


class Harness:
    def __init__(self, settings, max_handoffs: int = 3):
        self.models = {name: ScriptedChatModel() for name in ("orchestrator", "rag", "finance")}
        self.store = InMemoryStore()
        team = build_team(settings, self.store, model_factory=lambda name: self.models[name])
        self.graph = build_conversation_graph(
            team, max_handoffs_per_turn=max_handoffs, checkpointer=InMemorySaver(), store=self.store
        )

    def script(self, agent: str, *steps):
        self.models[agent].steps.extend(steps)

    async def say(self, text: str, thread: str = "t1"):
        return await self.graph.ainvoke(
            {"messages": [("user", text)]},
            {"configurable": {"thread_id": thread}},
            context=AppContext(user_id="u1"),
        )


def system_text(model: ScriptedChatModel, call: int = -1) -> str:
    return str(model.calls[call][0].content)


def visible(state) -> list[tuple[str | None, str]]:
    return [(m.name, m.text) for m in state["messages"] if isinstance(m, AIMessage)]


@pytest.fixture
def h(settings):
    return Harness(settings)


async def test_orchestrator_answers_directly(h):
    h.script("orchestrator", AIMessage(content="Cześć!"))
    state = await h.say("hej")
    assert visible(state) == [("orchestrator", "Cześć!")]
    assert state["active_agent"] == "orchestrator"


async def test_specialist_takes_over_conversation_and_returns(h):
    # Turn 1: orchestrator hands over; finance clarifies scope in the same turn.
    h.script(
        "orchestrator",
        tool_call("handoff_to_agent", {"agent": "finance", "reason": "Analiza budżetu"}, "h1", "Przekazuję."),
    )
    h.script("finance", AIMessage(content="Jaki okres mam przeanalizować?"))
    state = await h.say("Przeanalizuj budżet")
    assert visible(state)[-2:] == [
        ("orchestrator", "Przekazuję."),
        ("finance", "Jaki okres mam przeanalizować?"),
    ]
    assert state["active_agent"] == "finance"
    assert "Analiza budżetu" in system_text(h.models["finance"])
    assert "<conversation_role>" in system_text(h.models["finance"])

    # Turn 2: the user's answer goes straight to finance, which works with its tools.
    h.script(
        "finance",
        tool_call("describe_table", {"file": "budzet.xlsx"}, "d1"),
        AIMessage(content="Styczeń: wykonanie 165 zł przy planie 150 zł."),
    )
    state = await h.say("Styczeń")
    assert visible(state)[-1] == ("finance", "Styczeń: wykonanie 165 zł przy planie 150 zł.")
    tool_results = [m for m in h.models["finance"].calls[-1] if isinstance(m, ToolMessage)]
    assert "Rows: 4" in tool_results[0].text
    assert len(h.models["orchestrator"].calls) == 1  # orchestrator was not involved

    # Turn 3: finance summarizes and returns control without a pending request.
    h.script(
        "finance",
        tool_call(
            "return_to_orchestrator",
            {"summary": "Analiza stycznia: +10% vs plan"},
            "r1",
            "Gotowe, podsumowanie wyżej.",
        ),
    )
    state = await h.say("Dzięki, to wszystko")
    assert state["active_agent"] == "orchestrator"
    assert visible(state)[-1] == ("finance", "Gotowe, podsumowanie wyżej.")
    assert len(h.models["orchestrator"].calls) == 1  # nothing pending, orchestrator not woken
    episodes = [i.value for i in await h.store.asearch(episodic_ns("u1"), limit=50)]
    assert any(e["kind"] == "subagent_summary" and "stycznia" in e["summary"] for e in episodes)

    # Turn 4: back to the orchestrator; the old handoff note is gone.
    h.script("orchestrator", AIMessage(content="W czym jeszcze pomóc?"))
    await h.say("Co dalej?")
    assert "<conversation_role>" not in system_text(h.models["orchestrator"])


async def test_return_with_pending_request_wakes_orchestrator(h):
    h.script("orchestrator", tool_call("handoff_to_agent", {"agent": "rag", "reason": "polityka"}, "h1"))
    h.script("rag", AIMessage(content="O którą politykę chodzi?"))
    await h.say("Pytanie o politykę")

    h.script(
        "rag",
        tool_call(
            "return_to_orchestrator",
            {"summary": "Nic nie ustalono", "pending_request": "Ile wydaliśmy na marketing?"},
            "r1",
        ),
    )
    h.script("orchestrator", AIMessage(content="Sprawdzam to w danych finansowych."))
    state = await h.say("Właściwie, ile wydaliśmy na marketing?")
    assert visible(state)[-1] == ("orchestrator", "Sprawdzam to w danych finansowych.")
    assert "Ile wydaliśmy na marketing?" in system_text(h.models["orchestrator"])
    assert state["active_agent"] == "orchestrator"


async def test_handoff_loop_is_bounded(settings):
    h = Harness(settings, max_handoffs=1)
    h.script("orchestrator", tool_call("handoff_to_agent", {"agent": "rag", "reason": "x"}, "h1"))
    h.script("rag", tool_call("return_to_orchestrator", {"summary": "s", "pending_request": "p"}, "r1"))
    state = await h.say("ping-pong")
    # Second hop exceeds the limit: control is recorded but the orchestrator is not re-run.
    assert state["active_agent"] == "orchestrator"
    assert len(h.models["orchestrator"].calls) == 1


async def test_orchestrator_delegates_one_shot_task(h):
    h.script(
        "orchestrator",
        tool_call("task", {"subagent_type": "finance", "description": "Suma planu"}, "t1"),
        lambda msgs: AIMessage(content=f"Wynik: {msgs[-1].text}"),
    )
    h.script("finance", AIMessage(content="300 zł"))
    state = await h.say("Ile wynosi plan?")
    assert visible(state)[-1] == ("orchestrator", "Wynik: 300 zł")
    assert state["active_agent"] == "orchestrator"
    assert "Delegated task mode" in system_text(h.models["finance"])


async def test_memory_tools_and_procedural_memory(h):
    h.script(
        "orchestrator",
        tool_call("remember_fact", {"fact": "Raporty w EUR", "category": "preference"}, "m1"),
        tool_call("write_file", {"file_path": "/memories/AGENTS.md", "content": "- Raportuj w EUR"}, "m2"),
        AIMessage(content="Zapamiętane."),
    )
    await h.say("Zawsze raportuj w EUR")
    facts = await h.store.asearch(semantic_ns("u1"))
    assert facts[0].value["fact"] == "Raporty w EUR"

    # Procedural memory is loaded into every agent's prompt on the next run.
    h.script("orchestrator", AIMessage(content="ok"))
    await h.say("Nowe pytanie", thread="t2")
    assert "Raportuj w EUR" in system_text(h.models["orchestrator"])
    assert "Waluta: PLN." in system_text(h.models["orchestrator"])  # static /config/AGENTS.md


async def test_source_data_is_read_only(h, settings):
    h.script(
        "orchestrator",
        tool_call("write_file", {"file_path": "/finance/hack.md", "content": "x"}, "w1"),
        AIMessage(content="ok"),
    )
    await h.say("zapisz")
    denied = [m for m in h.models["orchestrator"].calls[-1] if isinstance(m, ToolMessage)]
    assert "denied" in denied[0].text.lower() or "permission" in denied[0].text.lower()
    assert not (settings.finance_dir / "hack.md").exists()


async def test_app_persists_history_across_restarts(settings):
    models = {name: ScriptedChatModel() for name in ("orchestrator", "rag", "finance")}
    models["orchestrator"].steps.append(AIMessage(content="Pierwsza odpowiedź"))
    async with create_app(settings, model_factory=lambda n: models[n], load_mcp=False) as app:
        result = await app.chat("Pierwsze pytanie", thread_id="th", user_id="u1")
        assert [r.text for r in result.replies] == ["Pierwsza odpowiedź"]

    models["orchestrator"].steps.append(AIMessage(content="Druga odpowiedź"))
    async with create_app(settings, model_factory=lambda n: models[n], load_mcp=False) as app:
        result = await app.chat("Drugie pytanie", thread_id="th", user_id="u1")
        assert [r.text for r in result.replies] == ["Druga odpowiedź"]
        history = await app.history("th")
    assert [speaker for speaker, _ in history] == ["user", "orchestrator", "user", "orchestrator"]
    # The model saw the earlier exchange restored from the SQLite checkpoint.
    assert "Pierwsze pytanie" in str(models["orchestrator"].calls[-1])
