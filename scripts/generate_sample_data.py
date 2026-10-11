"""Generate the sample Excel workbook in data/finance (fictional company, deterministic)."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "finance" / "budzet_2026.xlsx"
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
COSTS = {
    ("Wynagrodzenia", "Operacje"): 180_000,
    ("Wynagrodzenia", "Sprzedaż"): 95_000,
    ("Marketing", "Sprzedaż"): 40_000,
    ("IT i oprogramowanie", "Operacje"): 28_000,
    ("Najem biura", "Administracja"): 22_000,
    ("Podróże służbowe", "Sprzedaż"): 12_000,
    ("Szkolenia", "Administracja"): 6_000,
}
PRODUCTS = {"Abonament SaaS": 310_000, "Wdrożenia": 120_000, "Szkolenia": 25_000}


def main() -> None:
    rng = random.Random(2026)
    costs = []
    for month in MONTHS:
        for (category, dept), plan in COSTS.items():
            actual = round(plan * rng.uniform(0.85, 1.2), 2)
            costs.append(
                {"Miesiąc": month, "Kategoria": category, "Dział": dept, "Plan": plan, "Wykonanie": actual}
            )
    revenue = []
    for i, month in enumerate(MONTHS):
        for product, base in PRODUCTS.items():
            plan = round(base * (1 + 0.02 * i), 2)
            revenue.append(
                {
                    "Miesiąc": month,
                    "Produkt": product,
                    "Plan": plan,
                    "Wykonanie": round(plan * rng.uniform(0.8, 1.15), 2),
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT) as writer:
        pd.DataFrame(costs).to_excel(writer, sheet_name="Koszty", index=False)
        pd.DataFrame(revenue).to_excel(writer, sheet_name="Przychody", index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
