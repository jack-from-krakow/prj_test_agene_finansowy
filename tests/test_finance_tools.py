import re

import pytest

from agent_konwersacyjny.tools.finance_tools import FinanceData, FinanceDataError, make_finance_tools


@pytest.fixture
def tools(settings):
    return {t.name: t for t in make_finance_tools(FinanceData(settings.finance_dir))}


def test_list_files(tools):
    out = tools["list_financial_files"].invoke({})
    assert "budzet.xlsx" in out and "Koszty" in out and "notatki.md" in out


def test_describe(tools):
    out = tools["describe_table"].invoke({"file": "budzet.xlsx"})
    assert "Rows: 4" in out and "Wykonanie" in out


def test_query_with_filters_and_sort(tools):
    out = tools["query_table"].invoke(
        {
            "file": "budzet.xlsx",
            "filters": [{"column": "Kategoria", "op": "==", "value": "Marketing"}],
            "sort_by": "Wykonanie",
            "descending": True,
            "columns": ["Miesiąc", "Wykonanie"],
        }
    )
    lines = [line for line in out.splitlines() if "2026" in line]
    assert len(lines) == 2 and "120" in lines[0]


def test_aggregate_grouped(tools):
    out = tools["aggregate_table"].invoke(
        {
            "file": "budzet.xlsx",
            "metrics": [{"column": "Plan", "agg": "sum"}, {"column": "Wykonanie", "agg": "sum"}],
            "group_by": ["Kategoria"],
        }
    )
    assert re.search(r"Marketing\s*\|\s*200(\.00)?\s*\|\s*210(\.00)?\s*\|", out)
    assert re.search(r"IT\s*\|\s*100(\.00)?\s*\|\s*100(\.00)?\s*\|", out)


def test_errors_are_returned_to_the_model(tools):
    assert tools["describe_table"].invoke({"file": "brak.xlsx"}).startswith("Error")
    assert "Unknown column" in tools["query_table"].invoke(
        {"file": "budzet.xlsx", "filters": [{"column": "X", "op": "==", "value": 1}]}
    )


def test_path_traversal_blocked(settings):
    with pytest.raises(FinanceDataError):
        FinanceData(settings.finance_dir).resolve("../config/AGENTS.md")
