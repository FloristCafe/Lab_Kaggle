from __future__ import annotations

import random
from .data import CropPlanningData


def make_problem2_scenario(
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
        yield_per_mu={
            key: value * rng.uniform(0.90, 1.10)
            for key, value in data.yield_per_mu.items()
        },
        cost_per_mu=new_cost,
        price=new_price,
        demand=new_demand,
        planted_2023=data.planted_2023,
        y_2023=data.y_2023,
    )
