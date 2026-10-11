# Architektura

## Przegląd

```
                ┌──────────────────────── conversation graph (LangGraph) ────────────────────────┐
 user message → │ start_turn ──► [active_agent] ──handoff──► [target] ──► … ──► record_episode ──► │ → odpowiedzi
                └─────────────────────────────────────────────────────────────────────────────────┘
                         │                    │                     │
                   orchestrator              rag                 finance        (każdy = create_deep_agent)
                         │
                       task ──► rag / finance / general-purpose  (jednorazowe zadania w izolacji)
```

Graf rozmowy (`graph.py`) trzyma **publiczny transkrypt** (wiadomości użytkownika i widoczne
odpowiedzi agentów) oraz `active_agent`, czyli agenta, który dostanie następną wiadomość
użytkownika. Każdy agent to pełny Deep Agent (`create_deep_agent`) z planowaniem (`write_todos`),
systemem plików, podsumowywaniem kontekstu, skills i pamięcią AGENTS.md. Wewnętrzny ruch
narzędzi zostaje w przebiegu agenta, więc agenci z różnymi narzędziami nie widzą nawzajem swoich
wywołań.

## Dwa tryby delegowania

| | `task` (wbudowany w deepagents) | `handoff_to_agent` (ten projekt) |
|---|---|---|
| kto rozmawia z użytkownikiem | Orkiestrator | specjalista, bezpośrednio, przez wiele tur |
| kontekst specjalisty | tylko opis zadania | cały publiczny transkrypt + powód przekazania |
| kiedy | szybkie, dobrze określone pytania; równoległe zadania | zadania wieloetapowe, niejasne, wymagające dialogu |
| koniec | wynik wraca do Orkiestratora | `return_to_orchestrator(summary, pending_request?)` |

Po przejęciu rozmowy specjalista dostaje w prompcie notę `<conversation_role>` z fazami pracy:
doprecyzowanie zakresu → uzgodnienie podejścia → realizacja → podsumowanie → zwrot.
Podsumowanie trafia do pamięci epizodycznej. Jeśli użytkownik czeka na coś spoza domeny
specjalisty (`pending_request`), Orkiestrator odpowiada w tej samej turze. Liczba przekazań
w jednej turze jest ograniczona (`AGENT_MAX_HANDOFFS_PER_TURN`), co chroni przed pętlą.

Narzędzia przekazania mają `return_direct=True`: pętla agenta kończy się zaraz po wywołaniu,
a graf rozmowy odczytuje wywołanie z transkryptu i kieruje turę do celu.

## Wirtualny system plików

`CompositeBackend` z deepagents, wspólny dla wszystkich agentów (`backends.py`):

| ścieżka | backend | dostęp |
|---|---|---|
| `/` | `StateBackend` | brudnopis jednego przebiegu |
| `/workspace/` | `StoreBackend` (użytkownik + wątek) | pamięć sesji, odczyt/zapis |
| `/memories/` | `StoreBackend` (użytkownik) | pamięć proceduralna, odczyt/zapis |
| `/finance/`, `/docs/`, `/skills/`, `/config/` | `FilesystemBackend` | tylko odczyt (`FilesystemPermission`) |

## Pamięć

- **Historia rozmowy**: checkpointer (`AsyncSqliteSaver`); `thread_id` = sesja. W produkcji
  podmień na `langgraph-checkpoint-postgres`.
- **Sesja**: stan wątku (`active_agent`, `handoff`) i pliki `/workspace/`.
- **Semantyczna**: `remember_fact`, `recall_facts`, `forget_fact` w Store, przestrzeń
  `(user, "semantic")`.
- **Epizodyczna**: węzeł `record_episode` zapisuje każdą turę, a powroty specjalistów zapisują
  podsumowania; `search_episodes` przeszukuje je. Przestrzeń `(user, "episodic")`.
- **Proceduralna**: `MemoryMiddleware` deepagents ładuje `config/AGENTS.md` (repozytorium)
  i `/memories/AGENTS.md` (uczony przez agentów) do promptu każdego agenta. `skills/` zawiera
  procedury krok po kroku (`SkillsMiddleware`).

Wyszukiwanie w pamięci używa indeksu wektorowego Store, gdy skonfigurowano embeddingi, a w
przeciwnym razie dopasowania słów kluczowych.

## Bezpieczeństwo danych finansowych

Narzędzia finansowe nie wykonują kodu: model składa zapytania z ustalonego słownika (filtry,
agregacje), a `calculate` liczy na drzewie AST bez `eval`. Ścieżki są ograniczone do folderu
danych, a dane źródłowe są tylko do odczytu.

## Rozszerzanie

- **Nowy specjalista**: dodaj `SpecialistSpec` w `agents/factory.py::specialist_specs` (prompt
  i narzędzia). Automatycznie dostaje tryb rozmowy, tryb `task`, pamięć i MCP.
- **Narzędzia MCP**: `mcp_servers.json`, pole `agents`.
- **Human-in-the-loop**: `interrupt_on` w `create_deep_agent` (wymaga checkpointera, który już jest).
- **Nowe procedury**: katalog w `skills/` z plikiem `SKILL.md`.

## Znane ograniczenia / dalsze kroki

- Publiczny transkrypt rośnie bez ograniczeń. Każdy przebieg agenta korzysta z
  `SummarizationMiddleware`, ale trwałe streszczanie długich wątków to następny krok.
- Strumieniowanie odpowiedzi w CLI (`astream` z `stream_mode="messages"`).
- API HTTP (np. FastAPI albo LangGraph Server) i uwierzytelnianie użytkowników.
- Ewaluacje (LangSmith) na zestawie pytań finansowych i RAG.
