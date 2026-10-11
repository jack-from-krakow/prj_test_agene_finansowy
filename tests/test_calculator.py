import pytest

from agent_konwersacyjny.tools.calculator import calculate, safe_eval


def test_arithmetic():
    assert safe_eval("(125000 - 98000) / 98000 * 100") == pytest.approx(27.5510204)
    assert safe_eval("round(1234.5678, 2)") == 1234.57
    assert safe_eval("max(1, 2, 3) + abs(-4)") == 7


def test_polish_decimal_comma():
    assert safe_eval("1234,5 * 2") == 2469.0


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "a + 1", "2 ** 100000", "[1, 2]"])
def test_rejects_non_arithmetic(expr):
    with pytest.raises(ValueError):
        safe_eval(expr)


def test_tool_reports_errors_as_text():
    assert calculate.invoke({"expression": "1/0"}).startswith("Error")
    assert calculate.invoke({"expression": "2*21"}) == "2*21 = 42"
