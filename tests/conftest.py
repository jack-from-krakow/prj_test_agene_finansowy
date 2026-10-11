from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from agent_konwersacyjny.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    finance = tmp_path / "finance"
    docs = tmp_path / "docs"
    config = tmp_path / "config"
    for d in (finance, docs, config):
        d.mkdir()
    pd.DataFrame(
        {
            "Miesiąc": ["2026-01", "2026-01", "2026-02", "2026-02"],
            "Kategoria": ["Marketing", "IT", "Marketing", "IT"],
            "Plan": [100.0, 50.0, 100.0, 50.0],
            "Wykonanie": [120.0, 45.0, 90.0, 55.0],
        }
    ).to_excel(finance / "budzet.xlsx", sheet_name="Koszty", index=False)
    (finance / "notatki.md").write_text("Marketing przekroczony celowo.", encoding="utf-8")
    (docs / "polityka.md").write_text(
        "# Progi akceptacji\n\nWydatki powyżej 50 000 zł zatwierdza zarząd.\n\n"
        "# Podróże\n\nPodróże zagraniczne akceptuje CFO.",
        encoding="utf-8",
    )
    (config / "AGENTS.md").write_text("Waluta: PLN.", encoding="utf-8")
    return Settings(
        finance_dir=finance,
        docs_dir=docs,
        config_dir=config,
        skills_dir=tmp_path / "skills",
        storage_dir=tmp_path / "storage",
        mcp_config=tmp_path / "mcp_servers.json",
    )
