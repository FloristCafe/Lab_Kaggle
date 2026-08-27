from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - handled at runtime for machines without Gurobi.
    gp = None
    GRB = None

from .data import CropPlanningData


@dataclass(frozen=True)
class SolveResult:
    status: int
    objective_value: float | None
    planted_area: dict[tuple[str, str, int, int], float]
    normal_sales: dict[tuple[str, int], float]
    excess_sales: dict[tuple[str, int], float]


def solve_crop_plan(
    data: CropPlanningData,
    *,
    excess_price_factor: float = 0.5,
    min_area_if_planted: float = 0.0,
    time_limit: float | None = None,
    mip_gap: float | None = 0.01,
    output_flag: int = 1,
    price_modifiers: dict[tuple[str, int], float] | None = None,
) -> SolveResult:
    if gp is None or GRB is None:
        raise RuntimeError(
            "gurobipy is not installed. Install requirements and configure a Gurobi license first."
        )

    price_modifiers = price_modifiers or {}
    model = gp.Model("cumcm_2024_problem_c")
    model.Params.OutputFlag = output_flag
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    feasible_keys = [
        (i, j, t, s)
        for i in data.plots
        for j in data.crops
        for t in data.years
        for s in data.seasons
        if data.feasible.get((i, j, s), False)
    ]
    sales_keys = [(j, t) for j in data.crops for t in data.years]

    x = model.addVars(feasible_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVars(feasible_keys, vtype=GRB.BINARY, name="y")
    w = model.addVars(sales_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="normal_sales")
    e = model.addVars(sales_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="excess_sales")

    feasible_set = set(feasible_keys)
    _add_area_constraints(model, data, x, feasible_set)
    _add_big_m_constraints(model, data, x, y, feasible_keys, min_area_if_planted)
    _add_no_replant_constraints(model, data, y, feasible_set)
    _add_legume_rotation_constraints(model, data, y, feasible_set)
    _add_sales_balance_constraints(model, data, x, w, e, feasible_set)

    for j, t in sales_keys:
        model.addConstr(w[j, t] <= data.demand[j, t], name=f"demand_cap[{j},{t}]")

    revenue = gp.quicksum(
        w[j, t] * _price(data, j, t, price_modifiers)
        + e[j, t] * excess_price_factor * _price(data, j, t, price_modifiers)
        for j, t in sales_keys
    )
    cost = gp.quicksum(
        x[i, j, t, s] * data.cost_per_mu[i, j, t, s]
        for i, j, t, s in feasible_keys
    )
    model.setObjective(revenue - cost, GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0:
        return SolveResult(
            status=int(model.Status),
            objective_value=None,
            planted_area={},
            normal_sales={},
            excess_sales={},
        )

    return SolveResult(
        status=int(model.Status),
        objective_value=float(model.ObjVal),
        planted_area={
            key: float(var.X)
            for key, var in x.items()
            if abs(float(var.X)) > 1e-7
        },
        normal_sales={
            key: float(var.X)
            for key, var in w.items()
            if abs(float(var.X)) > 1e-7
        },
        excess_sales={
            key: float(var.X)
            for key, var in e.items()
            if abs(float(var.X)) > 1e-7
        },
    )


def _add_area_constraints(model, data: CropPlanningData, x, feasible_set: set[tuple[str, str, int, int]]) -> None:
    for i in data.plots:
        for t in data.years:
            for s in data.seasons:
                keys = [(i, j, t, s) for j in data.crops if (i, j, t, s) in feasible_set]
                if keys:
                    model.addConstr(
                        gp.quicksum(x[key] for key in keys) <= data.area[i],
                        name=f"area[{i},{t},{s}]",
                    )


def _add_big_m_constraints(
    model,
    data: CropPlanningData,
    x,
    y,
    feasible_keys: Iterable[tuple[str, str, int, int]],
    min_area_if_planted: float,
) -> None:
    for i, j, t, s in feasible_keys:
        model.addConstr(x[i, j, t, s] <= data.area[i] * y[i, j, t, s], name=f"big_m[{i},{j},{t},{s}]")
        if min_area_if_planted > 0:
            model.addConstr(
                x[i, j, t, s] >= min_area_if_planted * y[i, j, t, s],
                name=f"min_area[{i},{j},{t},{s}]",
            )


def _add_no_replant_constraints(model, data: CropPlanningData, y, feasible_set: set[tuple[str, str, int, int]]) -> None:
    first_year = min(data.years)
    for i in data.plots:
        for j in data.crops:
            if (i, j, first_year, 1) in feasible_set and any(data.y_2023.get((i, j, s), 0) for s in data.seasons):
                model.addConstr(y[i, j, first_year, 1] <= 0, name=f"no_replant_2023[{i},{j}]")
            for t in data.years:
                if (i, j, t, 1) in feasible_set and (i, j, t, 2) in feasible_set:
                    model.addConstr(
                        y[i, j, t, 1] + y[i, j, t, 2] <= 1,
                        name=f"no_replant_same_year[{i},{j},{t}]",
                    )
            for t1, t2 in zip(data.years[:-1], data.years[1:]):
                if (i, j, t1, 2) in feasible_set and (i, j, t2, 1) in feasible_set:
                    model.addConstr(
                        y[i, j, t1, 2] + y[i, j, t2, 1] <= 1,
                        name=f"no_replant_cross_year[{i},{j},{t1}]",
                    )
                for s in data.seasons:
                    if (i, j, t1, s) in feasible_set and (i, j, t2, s) in feasible_set:
                        model.addConstr(
                            y[i, j, t1, s] + y[i, j, t2, s] <= 1,
                            name=f"no_replant_same_season[{i},{j},{t1},{s}]",
                        )


def _add_legume_rotation_constraints(
    model,
    data: CropPlanningData,
    y,
    feasible_set: set[tuple[str, str, int, int]],
) -> None:
    legume_crops = [j for j in data.crops if data.is_legume[j]]
    planning_years = data.years
    for i in data.plots:
        for window_start in range(2023, max(planning_years) - 1 + 1):
            terms = []
            for year in (window_start, window_start + 1, window_start + 2):
                for j in legume_crops:
                    for s in data.seasons:
                        if year == 2023:
                            if data.y_2023.get((i, j, s), 0):
                                terms.append(1)
                        elif year in planning_years and (i, j, year, s) in feasible_set:
                            terms.append(y[i, j, year, s])
            if terms:
                model.addConstr(gp.quicksum(terms) >= 1, name=f"legume_rotation[{i},{window_start}]")


def _add_sales_balance_constraints(
    model,
    data: CropPlanningData,
    x,
    w,
    e,
    feasible_set: set[tuple[str, str, int, int]],
) -> None:
    for j in data.crops:
        for t in data.years:
            production = gp.quicksum(
                x[i, j, t, s] * data.yield_per_mu[i, j, t, s]
                for i in data.plots
                for s in data.seasons
                if (i, j, t, s) in feasible_set
            )
            model.addConstr(w[j, t] + e[j, t] == production, name=f"sales_balance[{j},{t}]")


def _price(
    data: CropPlanningData,
    crop: str,
    year: int,
    price_modifiers: dict[tuple[str, int], float],
) -> float:
    return data.price[crop, year] * price_modifiers.get((crop, year), price_modifiers.get((crop, 0), 1.0))


def total_crop_area(result: SolveResult, crop: str) -> float:
    return sum(area for (_, j, _, _), area in result.planted_area.items() if j == crop)


def write_solution_csv(result: SolveResult, output_path: Path) -> None:
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "plot": i,
            "crop": j,
            "year": t,
            "season": s,
            "area_mu": area,
        }
        for (i, j, t, s), area in sorted(result.planted_area.items())
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
