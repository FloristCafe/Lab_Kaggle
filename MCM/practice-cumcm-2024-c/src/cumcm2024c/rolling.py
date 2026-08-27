from __future__ import annotations

from collections import defaultdict

from .data import CropPlanningData
from .model import GRB, SolveResult, gp


def solve_crop_plan_rolling(
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
    planted_area: dict[tuple[str, str, int, int], float] = {}
    normal_sales: dict[tuple[str, int], float] = defaultdict(float)
    excess_sales: dict[tuple[str, int], float] = defaultdict(float)
    objective = 0.0
    history = _initial_history(data)

    for year in data.years:
        remaining_demand = {crop: data.demand[crop, year] for crop in data.crops}
        for season in data.seasons:
            sub = _solve_one_period(
                data,
                year=year,
                season=season,
                history=history,
                remaining_demand=remaining_demand,
                excess_price_factor=excess_price_factor,
                min_area_if_planted=min_area_if_planted,
                time_limit=time_limit,
                mip_gap=mip_gap,
                output_flag=output_flag,
                price_modifiers=price_modifiers,
            )
            if sub.objective_value is None:
                return SolveResult(
                    status=sub.status,
                    objective_value=None,
                    planted_area=planted_area,
                    normal_sales=dict(normal_sales),
                    excess_sales=dict(excess_sales),
                )
            objective += sub.objective_value
            for key, value in sub.planted_area.items():
                planted_area[key] = value
                plot, crop, _, period = key
                if value > 1e-7:
                    history[plot, crop, year, period] = 1
            for (crop, period_year), value in sub.normal_sales.items():
                normal_sales[crop, period_year] += value
                remaining_demand[crop] = max(0.0, remaining_demand[crop] - value)
            for key, value in sub.excess_sales.items():
                excess_sales[key] += value

    return SolveResult(
        status=2,
        objective_value=objective,
        planted_area=planted_area,
        normal_sales=dict(normal_sales),
        excess_sales=dict(excess_sales),
    )


def _solve_one_period(
    data: CropPlanningData,
    *,
    year: int,
    season: int,
    history: dict[tuple[str, str, int, int], int],
    remaining_demand: dict[str, float],
    excess_price_factor: float,
    min_area_if_planted: float,
    time_limit: float | None,
    mip_gap: float | None,
    output_flag: int,
    price_modifiers: dict[tuple[str, int], float],
) -> SolveResult:
    model = gp.Model(f"cumcm_2024_problem_c_rolling_{year}_{season}")
    model.Params.OutputFlag = output_flag
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    feasible_keys = [
        (plot, crop, year, season)
        for plot in data.plots
        for crop in data.crops
        if data.feasible.get((plot, crop, season), False)
    ]
    x = model.addVars(feasible_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVars(feasible_keys, vtype=GRB.BINARY, name="y")
    w = model.addVars(data.crops, lb=0.0, vtype=GRB.CONTINUOUS, name="normal_sales")
    e = model.addVars(data.crops, lb=0.0, vtype=GRB.CONTINUOUS, name="excess_sales")

    for plot in data.plots:
        keys = [key for key in feasible_keys if key[0] == plot]
        if keys:
            model.addConstr(gp.quicksum(x[key] for key in keys) <= data.area[plot], name=f"area[{plot}]")

    for plot, crop, _, _ in feasible_keys:
        key = (plot, crop, year, season)
        model.addConstr(x[key] <= data.area[plot] * y[key], name=f"big_m[{plot},{crop}]")
        if min_area_if_planted > 0:
            model.addConstr(x[key] >= min_area_if_planted * y[key], name=f"min_area[{plot},{crop}]")
        if _was_immediately_previous_planted(history, plot, crop, year, season):
            model.addConstr(y[key] <= 0, name=f"no_replant[{plot},{crop}]")

    legume_crops = [crop for crop in data.crops if data.is_legume[crop]]
    for plot in data.plots:
        if _needs_legume_now(data, history, plot, year, season, data.years[-1]):
            keys = [
                (plot, crop, year, season)
                for crop in legume_crops
                if (plot, crop, year, season) in x
            ]
            if keys:
                model.addConstr(gp.quicksum(y[key] for key in keys) >= 1, name=f"legume_due[{plot}]")
            else:
                raise ValueError(f"No feasible legume crop for plot={plot}, year={year}, season={season}")

    for crop in data.crops:
        production = gp.quicksum(
            x[plot, crop, year, season] * data.yield_per_mu[plot, crop, year, season]
            for plot in data.plots
            if (plot, crop, year, season) in x
        )
        model.addConstr(w[crop] + e[crop] == production, name=f"sales_balance[{crop}]")
        model.addConstr(w[crop] <= remaining_demand[crop], name=f"demand_cap[{crop}]")

    revenue = gp.quicksum(
        w[crop] * _price(data, crop, year, price_modifiers)
        + e[crop] * excess_price_factor * _price(data, crop, year, price_modifiers)
        for crop in data.crops
    )
    cost = gp.quicksum(
        x[plot, crop, year, season] * data.cost_per_mu[plot, crop, year, season]
        for plot, crop, _, _ in feasible_keys
    )
    model.setObjective(revenue - cost, GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0:
        return SolveResult(int(model.Status), None, {}, {}, {})

    return SolveResult(
        status=int(model.Status),
        objective_value=float(model.ObjVal),
        planted_area={
            key: float(var.X)
            for key, var in x.items()
            if abs(float(var.X)) > 1e-7
        },
        normal_sales={
            (crop, year): float(var.X)
            for crop, var in w.items()
            if abs(float(var.X)) > 1e-7
        },
        excess_sales={
            (crop, year): float(var.X)
            for crop, var in e.items()
            if abs(float(var.X)) > 1e-7
        },
    )


def _initial_history(data: CropPlanningData) -> dict[tuple[str, str, int, int], int]:
    return {
        (plot, crop, 2023, season): planted
        for (plot, crop, season), planted in data.y_2023.items()
        if planted
    }


def _was_immediately_previous_planted(
    history: dict[tuple[str, str, int, int], int],
    plot: str,
    crop: str,
    year: int,
    season: int,
) -> bool:
    if season == 2:
        return bool(history.get((plot, crop, year, 1), 0))
    return any(
        history.get((plot, crop, year - 1, prev_season), 0)
        for prev_season in (1, 2)
    )


def _needs_legume_now(
    data: CropPlanningData,
    history: dict[tuple[str, str, int, int], int],
    plot: str,
    year: int,
    season: int,
    last_year: int,
) -> bool:
    if season != _last_legume_opportunity(data, plot):
        return False
    legume_crops = {crop for crop in data.crops if data.is_legume[crop]}
    for window_start in range(2023, last_year - 1 + 1):
        window = (window_start, window_start + 1, window_start + 2)
        if year != window[-1]:
            continue
        has_legume = any(
            planted
            for (p, crop, hist_year, _season), planted in history.items()
            if p == plot and crop in legume_crops and hist_year in window
        )
        if not has_legume:
            return True
    return False


def _last_legume_opportunity(data: CropPlanningData, plot: str) -> int:
    feasible_legume_seasons = [
        season
        for season in data.seasons
        for crop in data.crops
        if data.is_legume[crop] and data.feasible.get((plot, crop, season), False)
    ]
    if not feasible_legume_seasons:
        raise ValueError(f"No feasible legume crop for plot={plot}")
    return max(feasible_legume_seasons)


def _price(
    data: CropPlanningData,
    crop: str,
    year: int,
    price_modifiers: dict[tuple[str, int], float],
) -> float:
    return data.price[crop, year] * price_modifiers.get((crop, year), price_modifiers.get((crop, 0), 1.0))
