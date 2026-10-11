"""Read-only, structured access to tabular financial data in a local folder.

The agent never runs arbitrary code: it composes filters/aggregations from a fixed
vocabulary, so a prompt-injected spreadsheet cannot execute anything.
"""

from __future__ import annotations

import functools
import threading
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

TABULAR_SUFFIXES = {".xlsx", ".xlsm", ".xls", ".csv"}
TEXT_SUFFIXES = {".md", ".txt"}
MAX_ROWS = 200

FilterOp = Literal["==", "!=", ">", ">=", "<", "<=", "contains", "in"]
AggFunc = Literal["sum", "mean", "median", "min", "max", "count", "nunique"]


class Filter(BaseModel):
    column: str
    op: FilterOp
    value: Any = Field(description="Comparison value; a list for `in`.")


class Metric(BaseModel):
    column: str
    agg: AggFunc


class FinanceDataError(ValueError):
    pass


class FinanceData:
    """Loads and caches tables from `root`, keyed by path + mtime."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._cache: dict[tuple[str, str | None, int, float], pd.DataFrame] = {}
        self._lock = threading.Lock()

    def resolve(self, file: str) -> Path:
        path = (self.root / file.lstrip("/")).resolve()
        if not path.is_relative_to(self.root):
            raise FinanceDataError(f"'{file}' is outside the finance data folder.")
        if not path.is_file():
            raise FinanceDataError(f"File '{file}' not found. Use list_financial_files.")
        return path

    def files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(
            p
            for p in self.root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in TABULAR_SUFFIXES | TEXT_SUFFIXES
            and not p.name.startswith(("~$", "."))
        )

    def sheets(self, path: Path) -> list[str]:
        if path.suffix.lower() == ".csv":
            return []
        return list(pd.ExcelFile(path).sheet_names)

    def load(self, file: str, sheet: str | None = None, header_row: int = 0) -> pd.DataFrame:
        path = self.resolve(file)
        suffix = path.suffix.lower()
        if suffix not in TABULAR_SUFFIXES:
            raise FinanceDataError(f"'{file}' is not tabular; read it with read_file instead.")
        key = (str(path), sheet, header_row, path.stat().st_mtime)
        with self._lock:
            if key not in self._cache:
                if suffix == ".csv":
                    df = pd.read_csv(path, header=header_row, sep=None, engine="python")
                else:
                    sheets = self.sheets(path)
                    if sheet is None:
                        sheet = sheets[0]
                    elif sheet not in sheets:
                        raise FinanceDataError(f"Sheet '{sheet}' not in {sheets}.")
                    df = pd.read_excel(path, sheet_name=sheet, header=header_row)
                df.columns = [str(c).strip() for c in df.columns]
                self._cache[key] = df.dropna(how="all")
            return self._cache[key]


def _apply_filters(df: pd.DataFrame, filters: list[Filter] | None) -> pd.DataFrame:
    for f in filters or []:
        if f.column not in df.columns:
            raise FinanceDataError(f"Unknown column '{f.column}'. Columns: {list(df.columns)}")
        col = df[f.column]
        match f.op:
            case "==":
                mask = col == f.value
            case "!=":
                mask = col != f.value
            case ">":
                mask = col > f.value
            case ">=":
                mask = col >= f.value
            case "<":
                mask = col < f.value
            case "<=":
                mask = col <= f.value
            case "contains":
                mask = col.astype(str).str.contains(str(f.value), case=False, na=False, regex=False)
            case "in":
                values = f.value if isinstance(f.value, list) else [f.value]
                mask = col.isin(values)
        df = df[mask]
    return df


def _check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise FinanceDataError(f"Unknown columns {missing}. Columns: {list(df.columns)}")


def _to_markdown(df: pd.DataFrame, limit: int) -> str:
    total = len(df)
    shown = df.head(min(limit, MAX_ROWS))
    table = shown.to_markdown(index=False, floatfmt=",.2f")
    suffix = f"\n\n({len(shown)} of {total} rows shown)" if total > len(shown) else f"\n\n({total} rows)"
    return table + suffix


def make_finance_tools(data: FinanceData) -> list[BaseTool]:
    def _guard(fn):  # noqa: ANN001, ANN202
        """Return data errors to the model as text so it can correct the call."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            try:
                return fn(*args, **kwargs)
            except (FinanceDataError, KeyError, TypeError, ValueError) as exc:
                return f"Error: {exc}"

        return wrapper

    @tool
    @_guard
    def list_financial_files() -> str:
        """List financial data files (Excel/CSV tables, txt/md notes) with their sheets."""
        files = data.files()
        if not files:
            return f"No files in the finance folder ({data.root})."
        lines = []
        for p in files:
            rel = p.relative_to(data.root).as_posix()
            info = f"- {rel} ({p.stat().st_size / 1024:.1f} KB)"
            if p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
                info += f" sheets={data.sheets(p)}"
            lines.append(info)
        return "\n".join(lines)

    @tool
    @_guard
    def describe_table(file: str, sheet: str | None = None, header_row: int = 0) -> str:
        """Show a table's shape, columns with types, sample rows and numeric statistics.

        Call this before querying a table you have not seen yet. `file` is relative to
        the finance folder; `sheet` defaults to the first sheet.
        """
        df = data.load(file, sheet, header_row)
        dtypes = "\n".join(f"- {c}: {t}" for c, t in df.dtypes.astype(str).items())
        numeric = df.select_dtypes("number")
        stats = numeric.describe().T[["count", "mean", "min", "max"]] if not numeric.empty else None
        parts = [
            f"Rows: {len(df)}, columns: {len(df.columns)}",
            f"Columns:\n{dtypes}",
            f"Sample:\n{df.head(5).to_markdown(index=False)}",
        ]
        if stats is not None:
            parts.append(f"Numeric summary:\n{stats.to_markdown(floatfmt=',.2f')}")
        return "\n\n".join(parts)

    @tool
    @_guard
    def query_table(
        file: str,
        sheet: str | None = None,
        columns: list[str] | None = None,
        filters: list[Filter] | None = None,
        sort_by: str | None = None,
        descending: bool = False,
        limit: int = 50,
        header_row: int = 0,
    ) -> str:
        """Select rows from a table with optional column projection, filters and sorting."""
        df = _apply_filters(data.load(file, sheet, header_row), filters)
        if sort_by:
            _check_columns(df, [sort_by])
            df = df.sort_values(sort_by, ascending=not descending)
        if columns:
            _check_columns(df, columns)
            df = df[columns]
        return _to_markdown(df, limit)

    @tool
    @_guard
    def aggregate_table(
        file: str,
        metrics: list[Metric],
        group_by: list[str] | None = None,
        sheet: str | None = None,
        filters: list[Filter] | None = None,
        header_row: int = 0,
    ) -> str:
        """Aggregate a table (sum/mean/median/min/max/count/nunique), optionally grouped.

        Example: totals per category -> metrics=[{"column": "Kwota", "agg": "sum"}],
        group_by=["Kategoria"].
        """
        df = _apply_filters(data.load(file, sheet, header_row), filters)
        _check_columns(df, [m.column for m in metrics] + (group_by or []))
        spec = {f"{m.column}_{m.agg}": (m.column, m.agg) for m in metrics}
        if group_by:
            result = df.groupby(group_by, dropna=False).agg(**spec).reset_index()
        else:
            result = pd.DataFrame([{name: df[col].agg(agg) for name, (col, agg) in spec.items()}])
        return _to_markdown(result, MAX_ROWS)

    return [list_financial_files, describe_table, query_table, aggregate_table]
