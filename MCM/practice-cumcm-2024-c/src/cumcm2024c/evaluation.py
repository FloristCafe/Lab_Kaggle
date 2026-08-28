from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import CropPlanningData


def read_solution_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    df["season"] = df["season"].astype(int)
    df["area_mu"] = pd.to_numeric(df["area_mu"], errors="coerce").fillna(0.0)
    return df[df["area_mu"] > 1e-9].copy()


def evaluate_solution_profit(
    solution: pd.DataFrame,
    data: CropPlanningData,
    *,
    excess_price_factor: float = 0.5,
) -> dict[str, object]:
    production: dict[tuple[str, int], float] = {}
    planting_cost = 0.0
    for row in solution.itertuples(index=False):
        key = (row.crop, int(row.year))
        prod = float(row.area_mu) * data.yield_per_mu[row.plot, row.crop, int(row.year), int(row.season)]
        production[key] = production.get(key, 0.0) + prod
        planting_cost += float(row.area_mu) * data.cost_per_mu[row.plot, row.crop, int(row.year), int(row.season)]

    normal_revenue = 0.0
    excess_revenue = 0.0
    normal_sales: dict[tuple[str, int], float] = {}
    excess_sales: dict[tuple[str, int], float] = {}
    for crop in data.crops:
        for year in data.years:
            key = (crop, year)
            produced = production.get(key, 0.0)
            normal = min(produced, data.demand[crop, year])
            excess = max(0.0, produced - normal)
            normal_sales[key] = normal
            excess_sales[key] = excess
            price = data.price[crop, year]
            normal_revenue += normal * price
            excess_revenue += excess * excess_price_factor * price

    return {
        "profit": float(normal_revenue + excess_revenue - planting_cost),
        "normal_revenue": float(normal_revenue),
        "excess_revenue": float(excess_revenue),
        "planting_cost": float(planting_cost),
        "production": production,
        "normal_sales": normal_sales,
        "excess_sales": excess_sales,
    }

