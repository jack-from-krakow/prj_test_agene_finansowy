"""A safe arithmetic evaluator (no `eval`), for exact financial calculations."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *xs: sum(xs),
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
}
_MAX_EXPONENT = 1000


def safe_eval(expression: str) -> float | int:
    """Evaluate an arithmetic expression. Raises `ValueError` on anything else."""

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
        raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")

    try:
        tree = ast.parse(
            expression.replace(",", ".") if _looks_decimal_comma(expression) else expression, mode="eval"
        )
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc.msg}") from exc
    return _eval(tree)


def _looks_decimal_comma(expression: str) -> bool:
    # "1234,56 * 2" (Polish notation) but not "max(1, 2)".
    return "," in expression and "(" not in expression


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. `(125000 - 98000) / 98000 * 100`.

    Supports + - * / // % **, parentheses and abs, round, min, max, sum, sqrt, log, exp.
    Always use this instead of mental arithmetic for financial numbers.
    """
    try:
        result = safe_eval(expression)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        return f"Error: {exc}"
    if isinstance(result, float):
        result = round(result, 6)
    return f"{expression} = {result}"
