from __future__ import annotations

import random
from .data import CropPlanningData


def make_problem2_scenario(
    data: CropPlanningData,
) -> CropPlanningData:
    """Build the deterministic worst-case scenario required by Problem 2."""
    first_year = min(data.years)
    wheat_corn = {"Wheat", "Corn"}
    new_demand = {}
    new_price = {}
    new_cost = {}
    new_yield = {}

    for crop in data.crops:
        crop_type = data.crop_type[crop].lower()
        for year in data.years:
            horizon = year - first_year
            base_demand = data.demand[crop, year]
            if crop in wheat_corn:
                new_demand[crop, year] = base_demand * (1.05 ** (horizon + 1))
            else:
                new_demand[crop, year] = base_demand * 0.95

            base_price = data.price[crop, year]
            if "vegetable" in crop_type:
                new_price[crop, year] = base_price * (1.05 ** (horizon + 1))
            elif "edible fungi" in crop_type:
                new_price[crop, year] = base_price * (0.95 ** (horizon + 1))
            else:
                new_price[crop, year] = base_price

    for (plot, crop, year, season), cost in data.cost_per_mu.items():
        horizon = year - first_year
        new_cost[plot, crop, year, season] = cost * (1.05 ** horizon)

    for key, value in data.yield_per_mu.items():
        new_yield[key] = value * 0.90

    return _replace_parameters(
        data,
        demand=new_demand,
        price=new_price,
        cost_per_mu=new_cost,
        yield_per_mu=new_yield,
    )


def make_worst_case_problem2_scenario(data: CropPlanningData) -> CropPlanningData:
    return make_problem2_scenario(data)


def make_random_problem2_scenario(
    data: CropPlanningData,
    *,
    seed: int = 2024,
) -> CropPlanningData:
    rng = random.Random(seed)
    wheat_corn = {"Wheat", "Corn"}
    new_demand = {}
    new_price = {}
    new_cost = {}
    for crop in data.crops:
        crop_type = data.crop_type[crop].lower()
        for year in data.years:
            horizon = year - min(data.years) + 1
            base_demand = data.demand[crop, year]
            if crop in wheat_corn:
                growth = rng.uniform(0.05, 0.10)
                new_demand[crop, year] = base_demand * ((1.0 + growth) ** horizon)
            else:
                new_demand[crop, year] = base_demand * rng.uniform(0.95, 1.05)

            base_price = data.price[crop, year]
            if "vegetable" in crop_type:
                new_price[crop, year] = base_price * (1.05 ** horizon)
            elif "edible fungi" in crop_type:
                decline = 0.05 if crop == "Morel" else rng.uniform(0.01, 0.05)
                new_price[crop, year] = base_price * ((1.0 - decline) ** horizon)
            else:
                new_price[crop, year] = base_price

    for (plot, crop, year, season), cost in data.cost_per_mu.items():
        horizon = year - min(data.years) + 1
        new_cost[plot, crop, year, season] = cost * (1.05 ** horizon)

    return _replace_parameters(
        data,
        demand=new_demand,
        price=new_price,
        cost_per_mu=new_cost,
        yield_per_mu={
            key: value * rng.uniform(0.90, 1.10)
            for key, value in data.yield_per_mu.items()
        },
    )


def _replace_parameters(
    data: CropPlanningData,
    *,
    demand: dict[tuple[str, int], float],
    price: dict[tuple[str, int], float],
    cost_per_mu: dict[tuple[str, str, int, int], float],
    yield_per_mu: dict[tuple[str, str, int, int], float],
) -> CropPlanningData:
    return CropPlanningData(
        plots=data.plots,
        crops=data.crops,
        years=data.years,
        seasons=data.seasons,
        area=data.area,
        plot_type=data.plot_type,
        crop_id=data.crop_id,
        crop_name_by_id=data.crop_name_by_id,
        crop_type=data.crop_type,
        is_legume=data.is_legume,
        feasible=data.feasible,
        yield_per_mu=yield_per_mu,
        cost_per_mu=cost_per_mu,
        price=price,
        demand=demand,
        planted_2023=data.planted_2023,
        y_2023=data.y_2023,
    )
