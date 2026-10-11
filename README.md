# Agent konwersacyjny (Deep Agent)

Konwersacyjny system wieloagentowy zbudowany na [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
i LangGraph. **Orkiestrator** prowadzi rozmowę i koordynuje pracę sub-agentów. Każdy specjalista
może **przejąć rozmowę**: doprecyzować zakres, omówić sposób realizacji, wykonać zadanie,
podsumować wynik i oddać rozmowę Orkiestratorowi.

Specjaliści na start:

| Agent | Zakres | Narzędzia |
|---|---|---|
| `orchestrator` | rozmowa, planowanie, routing, pamięć | `handoff_to_agent`, `task`, `write_todos`, pamięć, `calculate`, MCP |
| `rag` | wiedza z dokumentów (`data/docs`: md/txt/pdf) | `search_knowledge_base` (BM25 + opcjonalnie embeddingi), `read_file` |
| `finance` | dane finansowe (`data/finance`: xlsx/csv/txt/md) | `describe_table`, `query_table`, `aggregate_table`, `calculate`, `read_file` |

Szczegóły projektu: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Szybki start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # uzupełnij ANTHROPIC_API_KEY
agent-chat                    # albo: python -m agent_konwersacyjny
```

W czacie: `/new` (nowy wątek), `/history`, `/thread`, `/exit`. Rozmowę można wznowić po restarcie:
`agent-chat --thread <id>`. Pamięć długoterminowa jest per użytkownik: `--user jan`.

Przykładowe pytania na danych demo:

- „Jakie są progi akceptacji wydatków?” (RAG)
- „Ile wyniosły koszty marketingu w III kwartale w porównaniu z planem?” (Finanse)
- „Przygotuj analizę odchyleń budżetu za 2026” (przejęcie rozmowy przez agenta finansowego)

### LangGraph Studio

```bash
pip install "langgraph-cli[inmem]"
langgraph dev
```

## Model

Model jest konfigurowalny (`AGENT_MODEL`, format `provider:model` z `init_chat_model`). Domyślnie
tani `anthropic:claude-haiku-5-5`. Każdemu agentowi można ustawić inny model, np. mocniejszy
Orkiestrator: `AGENT_ORCHESTRATOR_MODEL=anthropic:claude-sonnet-5-5`. Inni dostawcy działają po
doinstalowaniu integracji (np. `pip install -e ".[openai]"` i `AGENT_MODEL=openai:<nazwa-modelu>`).

## Dane

- `data/finance/` – Excel/CSV i notatki. Przykładowy skoroszyt generuje
  `python scripts/generate_sample_data.py` (dane fikcyjne).
- `data/docs/` – dokumenty dla RAG. Indeks odświeża się sam po zmianie plików.
- Własne foldery: `AGENT_FINANCE_DIR`, `AGENT_DOCS_DIR`. Dane źródłowe są dla agentów **tylko do odczytu**.

## Pamięć

| Rodzaj | Gdzie | Do czego |
|---|---|---|
| historia rozmowy | checkpointer SQLite (`.agent_data/checkpoints.sqlite`) | pełny stan wątku, wznowienie rozmowy |
| sesja (krótkotrwała) | stan wątku + pliki `/workspace/` | aktywny agent, kontekst przekazania, robocze wyniki |
| semantyczna | LangGraph Store (`semantic`) | fakty o użytkowniku i firmie: `remember_fact` / `recall_facts` |
| epizodyczna | LangGraph Store (`episodic`) | przebieg wcześniejszych tur i podsumowania specjalistów: `search_episodes` |
| proceduralna | `/memories/AGENTS.md` (edytowalny przez agentów), `config/AGENTS.md`, `skills/` | jak pracować: preferencje, konwencje, procedury |

Ustaw `AGENT_EMBEDDINGS_MODEL`, aby włączyć wyszukiwanie wektorowe w RAG i pamięci semantycznej.
Bez tego działa BM25 i wyszukiwanie po słowach kluczowych, bez kosztów.

## MCP

Skopiuj `mcp_servers.example.json` do `mcp_servers.json` i dopisz serwery. Pole `agents` decyduje,
który agent dostaje narzędzia serwera (`orchestrator`, `rag`, `finance`, domyślnie wszyscy).
Serwer, który nie wystartuje, jest pomijany z ostrzeżeniem.

## Testy

```bash
pytest          # testy offline na atrapie modelu, bez kosztów API
ruff check . && ruff format --check .
```
