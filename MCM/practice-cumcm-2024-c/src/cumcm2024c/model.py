from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from pyscipopt import Model, quicksum
except ImportError:  # pragma: no cover - handled at runtime for machines without SCIP.
    Model = None
    quicksum = None

from .data import CropPlanningData


@dataclass(frozen=True)
class SolveResult:
    status: str
    objective_value: float | None
    planted_area: dict[tuple[str, str, int, int], float]
    normal_sales: dict[tuple[str, int], float]
    excess_sales: dict[tuple[str, int], float]
    metadata: dict[str, object] | None = None


def solve_crop_plan(
    data: CropPlanningData,
    *,
    excess_price_factor: float = 0.5,
    min_area_if_planted: float = 0.1,
    time_limit: float | None = None,
    mip_gap: float | None = 0.01,
    output_flag: int = 1,
    price_modifiers: dict[tuple[str, int], float] | None = None,
    log_file: Path | None = None,
) -> SolveResult:
    if Model is None or quicksum is None:
        raise RuntimeError(
            "PySCIPOpt is not installed. Run `pip install pyscipopt` in your project environment first."
        )

    price_modifiers = price_modifiers or {}
    model = Model("cumcm_2024_problem_c")
    if output_flag == 0:
        model.hideOutput()
    if time_limit is not None:
        model.setRealParam("limits/time", float(time_limit))
    if mip_gap is not None:
        model.setRealParam("limits/gap", float(mip_gap))
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        model.setLogfile(str(log_file))

    feasible_keys = [
        (i, j, t, s)
        for i in data.plots
        for j in data.crops
        for t in data.years
        for s in data.seasons
        if data.feasible.get((i, j, s), False)
    ]
    sales_keys = [(j, t) for j in data.crops for t in data.years]

    x = _add_vars(model, feasible_keys, lb=0.0, vtype="C", prefix="x")
    y = _add_vars(model, feasible_keys, lb=0.0, ub=1.0, vtype="B", prefix="y")
    w = _add_vars(model, sales_keys, lb=0.0, vtype="C", prefix="normal_sales")
    e = _add_vars(model, sales_keys, lb=0.0, vtype="C", prefix="excess_sales")

    feasible_set = set(feasible_keys)
    _add_area_constraints(model, data, x, feasible_set)
    _add_big_m_constraints(model, data, x, y, feasible_keys, min_area_if_planted)
    _add_no_replant_constraints(model, data, y, feasible_set)
    _add_legume_rotation_constraints(model, data, y, feasible_set)
    _add_sales_balance_constraints(model, data, x, w, e, feasible_set)

    for j, t in sales_keys:
        model.addCons(w[j, t] <= data.demand[j, t], name=f"demand_cap[{j},{t}]")

    profit = _profit_expression(
        data,
        x=x,
        w=w,
        e=e,
        feasible_keys=feasible_keys,
        sales_keys=sales_keys,
        excess_price_factor=excess_price_factor,
        price_modifiers=price_modifiers,
    )
    model.setObjective(profit, "maximize")
    model.optimize()

    status = str(model.getStatus())
    if model.getNSols() == 0:
        return SolveResult(
            status=status,
            objective_value=None,
            planted_area={},
            normal_sales={},
            excess_sales={},
            metadata={"log_file": str(log_file) if log_file is not None else None},
        )

    return SolveResult(
        status=status,
        objective_value=float(model.getObjVal()),
        planted_area={
            key: float(model.getVal(var))
            for key, var in x.items()
            if abs(float(model.getVal(var))) > 1e-7
        },
        normal_sales={
            key: float(model.getVal(var))
            for key, var in w.items()
            if abs(float(model.getVal(var))) > 1e-7
        },
        excess_sales={
            key: float(model.getVal(var))
            for key, var in e.items()
            if abs(float(model.getVal(var))) > 1e-7
        },
        metadata={"log_file": str(log_file) if log_file is not None else None},
    )


def solve_crop_plan_miqp(
    data: CropPlanningData,
    *,
    cov_dict: dict[tuple[str, str], float],
    risk_lambda: float,
    excess_price_factor: float = 0.5,
    min_area_if_planted: float = 0.1,
    output_flag: int = 1,
    price_modifiers: dict[tuple[str, int], float] | None = None,
    log_file: Path | None = None,
) -> SolveResult:
    if Model is None or quicksum is None:
        raise RuntimeError(
            "PySCIPOpt is not installed. Run `pip install pyscipopt` in your project environment first."
        )

    price_modifiers = price_modifiers or {}
    model = Model("cumcm_2024_problem_c_miqp")
    if output_flag == 0:
        model.hideOutput()
    model.setRealParam("limits/time", 180.0)
    model.setRealParam("limits/gap", 0.01)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        model.setLogfile(str(log_file))

    feasible_keys = [
        (i, j, t, s)
        for i in data.plots
        for j in data.crops
        for t in data.years
        for s in data.seasons
        if data.feasible.get((i, j, s), False)
    ]
    sales_keys = [(j, t) for j in data.crops for t in data.years]

    x = _add_vars(model, feasible_keys, lb=0.0, vtype="C", prefix="x")
    y = _add_vars(model, feasible_keys, lb=0.0, ub=1.0, vtype="B", prefix="y")
    w = _add_vars(model, sales_keys, lb=0.0, vtype="C", prefix="normal_sales")
    e = _add_vars(model, sales_keys, lb=0.0, vtype="C", prefix="excess_sales")
    total_area = _add_vars(model, data.crops, lb=0.0, vtype="C", prefix="total_area")

    feasible_set = set(feasible_keys)
    _add_area_constraints(model, data, x, feasible_set)
    _add_big_m_constraints(model, data, x, y, feasible_keys, min_area_if_planted)
    _add_no_replant_constraints(model, data, y, feasible_set)
    _add_legume_rotation_constraints(model, data, y, feasible_set)
    _add_sales_balance_constraints(model, data, x, w, e, feasible_set)

    for j, t in sales_keys:
        model.addCons(w[j, t] <= data.demand[j, t], name=f"demand_cap[{j},{t}]")

    for j in data.crops:
        model.addCons(
            total_area[j] == quicksum(
                x[i, j, t, s]
                for i in data.plots
                for t in data.years
                for s in data.seasons
                if (i, j, t, s) in feasible_set
            ),
            name=f"total_area[{j}]",
        )

    expected_profit = _profit_expression(
        data,
        x=x,
        w=w,
        e=e,
        feasible_keys=feasible_keys,
        sales_keys=sales_keys,
        excess_price_factor=excess_price_factor,
        price_modifiers=price_modifiers,
    )
    risk_expr = quicksum(
        total_area[j] * total_area[k] * float(cov_dict.get((j, k), 0.0))
        for j in data.crops
        for k in data.crops
    )
    if risk_lambda > 0:
        risk_penalty = model.addVar(name="portfolio_risk", lb=0.0, vtype="C")
        model.addCons(risk_penalty >= risk_expr, name="portfolio_risk_epigraph")
        model.setObjective(expected_profit - risk_lambda * risk_penalty, "maximize")
    else:
        model.setObjective(expected_profit, "maximize")
    model.optimize()

    status = str(model.getStatus())
    if model.getNSols() == 0:
        return SolveResult(
            status=status,
            objective_value=None,
            planted_area={},
            normal_sales={},
            excess_sales={},
            metadata={
                "risk_lambda": risk_lambda,
                "log_file": str(log_file) if log_file is not None else None,
            },
        )

    planted_area = {
        key: float(model.getVal(var))
        for key, var in x.items()
        if abs(float(model.getVal(var))) > 1e-7
    }
    normal_sales = {
        key: float(model.getVal(var))
        for key, var in w.items()
        if abs(float(model.getVal(var))) > 1e-7
    }
    excess_sales = {
        key: float(model.getVal(var))
        for key, var in e.items()
        if abs(float(model.getVal(var))) > 1e-7
    }
    expected_profit_value = _compute_expected_profit_value(
        data,
        planted_area=planted_area,
        normal_sales=normal_sales,
        excess_sales=excess_sales,
        excess_price_factor=excess_price_factor,
        price_modifiers=price_modifiers,
    )
    portfolio_risk_value = _compute_portfolio_risk_value(data, planted_area, cov_dict)

    return SolveResult(
        status=status,
        objective_value=float(model.getObjVal()),
        planted_area=planted_area,
        normal_sales=normal_sales,
        excess_sales=excess_sales,
        metadata={
            "risk_lambda": risk_lambda,
            "expected_profit": expected_profit_value,
            "portfolio_risk": portfolio_risk_value,
            "log_file": str(log_file) if log_file is not None else None,
        },
    )


def _add_area_constraints(model, data: CropPlanningData, x, feasible_set: set[tuple[str, str, int, int]]) -> None:
    for i in data.plots:
        for t in data.years:
            for s in data.seasons:
                keys = [(i, j, t, s) for j in data.crops if (i, j, t, s) in feasible_set]
                if keys:
                    model.addCons(
                        quicksum(x[key] for key in keys) <= data.area[i],
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
        model.addCons(x[i, j, t, s] <= data.area[i] * y[i, j, t, s], name=f"big_m[{i},{j},{t},{s}]")
        if min_area_if_planted > 0:
            model.addCons(
                x[i, j, t, s] >= min_area_if_planted * y[i, j, t, s],
                name=f"min_area[{i},{j},{t},{s}]",
            )


def _add_no_replant_constraints(model, data: CropPlanningData, y, feasible_set: set[tuple[str, str, int, int]]) -> None:
    first_year = min(data.years)
    for i in data.plots:
        for j in data.crops:
            if (i, j, first_year, 1) in feasible_set and any(data.y_2023.get((i, j, s), 0) for s in data.seasons):
                model.addCons(y[i, j, first_year, 1] <= 0, name=f"no_replant_2023[{i},{j}]")
            for t in data.years:
                if (i, j, t, 1) in feasible_set and (i, j, t, 2) in feasible_set:
                    model.addCons(
                        y[i, j, t, 1] + y[i, j, t, 2] <= 1,
                        name=f"no_replant_same_year[{i},{j},{t}]",
                    )
            for t1, t2 in zip(data.years[:-1], data.years[1:]):
                if (i, j, t1, 2) in feasible_set and (i, j, t2, 1) in feasible_set:
                    model.addCons(
                        y[i, j, t1, 2] + y[i, j, t2, 1] <= 1,
                        name=f"no_replant_cross_year[{i},{j},{t1}]",
                    )
                for s in data.seasons:
                    if (i, j, t1, s) in feasible_set and (i, j, t2, s) in feasible_set:
                        model.addCons(
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
                model.addCons(quicksum(terms) >= 1, name=f"legume_rotation[{i},{window_start}]")


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
            production = quicksum(
                x[i, j, t, s] * data.yield_per_mu[i, j, t, s]
                for i in data.plots
                for s in data.seasons
                if (i, j, t, s) in feasible_set
            )
            model.addCons(w[j, t] + e[j, t] == production, name=f"sales_balance[{j},{t}]")


def _price(
    data: CropPlanningData,
    crop: str,
    year: int,
    price_modifiers: dict[tuple[str, int], float],
) -> float:
    return data.price[crop, year] * price_modifiers.get((crop, year), price_modifiers.get((crop, 0), 1.0))


def _profit_expression(
    data: CropPlanningData,
    *,
    x,
    w,
    e,
    feasible_keys: list[tuple[str, str, int, int]],
    sales_keys: list[tuple[str, int]],
    excess_price_factor: float,
    price_modifiers: dict[tuple[str, int], float],
):
    revenue = quicksum(
        w[j, t] * _price(data, j, t, price_modifiers)
        + e[j, t] * excess_price_factor * _price(data, j, t, price_modifiers)
        for j, t in sales_keys
    )
    cost = quicksum(
        x[i, j, t, s] * data.cost_per_mu[i, j, t, s]
        for i, j, t, s in feasible_keys
    )
    return revenue - cost


def _add_vars(model, keys, *, lb: float = 0.0, ub: float | None = None, vtype: str = "C", prefix: str):
    variables = {}
    for key in keys:
        kwargs = {"name": f"{prefix}[{_key_name(key)}]", "lb": lb, "vtype": vtype}
        if ub is not None:
            kwargs["ub"] = ub
        variables[key] = model.addVar(**kwargs)
    return variables


def _key_name(key) -> str:
    if isinstance(key, tuple):
        return ",".join(str(part) for part in key)
    return str(key)


def _compute_expected_profit_value(
    data: CropPlanningData,
    *,
    planted_area: dict[tuple[str, str, int, int], float],
    normal_sales: dict[tuple[str, int], float],
    excess_sales: dict[tuple[str, int], float],
    excess_price_factor: float,
    price_modifiers: dict[tuple[str, int], float],
) -> float:
    revenue = sum(
        normal_sales.get((crop, year), 0.0) * _price(data, crop, year, price_modifiers)
        + excess_sales.get((crop, year), 0.0) * excess_price_factor * _price(data, crop, year, price_modifiers)
        for crop in data.crops
        for year in data.years
    )
    cost = sum(
        area * data.cost_per_mu[plot, crop, year, season]
        for (plot, crop, year, season), area in planted_area.items()
    )
    return float(revenue - cost)


def _compute_portfolio_risk_value(
    data: CropPlanningData,
    planted_area: dict[tuple[str, str, int, int], float],
    cov_dict: dict[tuple[str, str], float],
) -> float:
    total_area = {
        crop: sum(area for (_plot, planted_crop, _year, _season), area in planted_area.items() if planted_crop == crop)
        for crop in data.crops
    }
    return float(
        sum(
            total_area[j] * total_area[k] * float(cov_dict.get((j, k), 0.0))
            for j in data.crops
            for k in data.crops
        )
    )


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
