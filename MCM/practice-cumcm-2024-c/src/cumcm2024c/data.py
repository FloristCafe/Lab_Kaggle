from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_DATA_DIR, SEASONS, YEARS


@dataclass(frozen=True)
class CropPlanningData:
    plots: list[str]
    crops: list[str]
    years: list[int]
    seasons: list[int]
    area: dict[str, float]
    plot_type: dict[str, str]
    crop_id: dict[str, int]
    crop_name_by_id: dict[int, str]
    crop_type: dict[str, str]
    is_legume: dict[str, bool]
    feasible: dict[tuple[str, str, int], bool]
    yield_per_mu: dict[tuple[str, str, int, int], float]
    cost_per_mu: dict[tuple[str, str, int, int], float]
    price: dict[tuple[str, int], float]
    demand: dict[tuple[str, int], float]
    planted_2023: dict[tuple[str, str, int], float]
    y_2023: dict[tuple[str, str, int], int]


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\u3000", " ").split()).strip()


def parse_price_midpoint(value: Any) -> float:
    text = clean_text(value)
    if not text:
        return 0.0
    if "-" in text:
        lo, hi = (float(part.strip()) for part in text.split("-", maxsplit=1))
        return (lo + hi) / 2.0
    return float(text)


def season_to_int(value: Any) -> int:
    text = clean_text(value).lower()
    if "second" in text or text in {"2", "2.0"}:
        return 2
    return 1


def normalize_plot_type(value: Any) -> str:
    return clean_text(value)


def normalize_crop_name(value: Any) -> str:
    return clean_text(value)


def load_raw_tables(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    annex1 = data_dir / "Annex1.xlsx"
    annex2 = data_dir / "Annex2.xlsx"
    if not annex1.exists() or not annex2.exists():
        raise FileNotFoundError(f"Cannot find Annex1.xlsx and Annex2.xlsx under {data_dir}")

    return {
        "plots": pd.read_excel(annex1, sheet_name="Existing Cultivated Land"),
        "crops": pd.read_excel(annex1, sheet_name="Cultivate Crops"),
        "planting_2023": pd.read_excel(annex2, sheet_name="Rural Crop Planting in 2023"),
        "stats_2023": pd.read_excel(annex2, sheet_name="Statistical data from 2023"),
    }


def build_crop_planning_data(
    data_dir: Path = DEFAULT_DATA_DIR,
    years: tuple[int, ...] = YEARS,
) -> CropPlanningData:
    tables = load_raw_tables(data_dir)
    plots_df = _clean_plots(tables["plots"])
    crops_df = _clean_crops(tables["crops"])
    planting_df = _clean_planting_2023(tables["planting_2023"])
    stats_df = _clean_stats_2023(tables["stats_2023"])

    plots = plots_df["plot"].tolist()
    crops = crops_df["crop"].tolist()
    years_list = list(years)
    seasons = list(SEASONS)

    area_dict = plots_df.set_index("plot")["area"].astype(float).to_dict()
    plot_type_dict = plots_df.set_index("plot")["plot_type"].to_dict()
    crop_id_dict = crops_df.set_index("crop")["crop_id"].astype(int).to_dict()
    crop_name_by_id = crops_df.set_index("crop_id")["crop"].to_dict()
    crop_type_dict = crops_df.set_index("crop")["crop_type"].to_dict()
    is_legume = {
        crop: "legume" in crop_type.lower()
        for crop, crop_type in crop_type_dict.items()
    }

    stat_maps = _build_stat_maps(stats_df)
    feasible: dict[tuple[str, str, int], bool] = {}
    base_yield: dict[tuple[str, str, int], float] = {}
    base_cost: dict[tuple[str, str, int], float] = {}
    base_price: dict[str, float] = {}

    for plot in plots:
        ptype = plot_type_dict[plot]
        for crop in crops:
            cid = crop_id_dict[crop]
            for season in seasons:
                key = (cid, ptype, season)
                ok = key in stat_maps["yield"]
                feasible[(plot, crop, season)] = ok
                if ok:
                    base_yield[(plot, crop, season)] = float(stat_maps["yield"][key])
                    base_cost[(plot, crop, season)] = float(stat_maps["cost"][key])
                    base_price[crop] = float(stat_maps["price"][key])
                else:
                    base_yield[(plot, crop, season)] = 0.0
                    base_cost[(plot, crop, season)] = 0.0

    demand_2023 = _estimate_2023_demand(planting_df, stats_df, crop_name_by_id, plot_type_dict)
    demand_dict = {
        (crop, year): float(demand_2023.get(crop, 0.0))
        for crop in crops
        for year in years_list
    }
    price_dict = {
        (crop, year): float(base_price.get(crop, 0.0))
        for crop in crops
        for year in years_list
    }
    cost_dict = {
        (plot, crop, year, season): float(base_cost[(plot, crop, season)])
        for plot in plots
        for crop in crops
        for year in years_list
        for season in seasons
    }
    yield_dict = {
        (plot, crop, year, season): float(base_yield[(plot, crop, season)])
        for plot in plots
        for crop in crops
        for year in years_list
        for season in seasons
    }

    planted_2023 = {
        (row.plot, row.crop, int(row.season)): float(row.area)
        for row in planting_df.itertuples(index=False)
    }
    y_2023 = {
        (plot, crop, season): int(planted_2023.get((plot, crop, season), 0.0) > 1e-9)
        for plot in plots
        for crop in crops
        for season in seasons
    }

    return CropPlanningData(
        plots=plots,
        crops=crops,
        years=years_list,
        seasons=seasons,
        area=area_dict,
        plot_type=plot_type_dict,
        crop_id=crop_id_dict,
        crop_name_by_id=crop_name_by_id,
        crop_type=crop_type_dict,
        is_legume=is_legume,
        feasible=feasible,
        yield_per_mu=yield_dict,
        cost_per_mu=cost_dict,
        price=price_dict,
        demand=demand_dict,
        planted_2023=planted_2023,
        y_2023=y_2023,
    )


def _clean_plots(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(
        columns={
            "Plot Name": "plot",
            "Plot Type": "plot_type",
            "Plot Area/mu": "area",
        }
    )[["plot", "plot_type", "area"]].copy()
    out["plot"] = out["plot"].map(clean_text)
    out["plot_type"] = out["plot_type"].map(normalize_plot_type)
    out["area"] = pd.to_numeric(out["area"], errors="coerce")
    out = out.dropna(subset=["plot", "area"])
    out = out[out["plot"] != ""].drop_duplicates("plot")
    return out.reset_index(drop=True)


def _clean_crops(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(
        columns={
            "Crop ID ": "crop_id",
            "Crop ID": "crop_id",
            "Crop Name": "crop",
            " Crop Type": "crop_type",
            "Crop Type": "crop_type",
        }
    )[["crop_id", "crop", "crop_type"]].copy()
    out["crop_id"] = pd.to_numeric(out["crop_id"], errors="coerce").astype("Int64")
    out["crop"] = out["crop"].map(normalize_crop_name)
    out["crop_type"] = out["crop_type"].map(clean_text)
    out = out.dropna(subset=["crop_id"])
    out = out[out["crop"] != ""].drop_duplicates("crop_id")
    out["crop_id"] = out["crop_id"].astype(int)
    return out.reset_index(drop=True)


def _clean_planting_2023(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(
        columns={
            "Planting Plot": "plot",
            "Crop ID": "crop_id",
            "Crop Name": "crop",
            "Crop Type": "crop_type",
            "Planted Area /mu": "area",
            "Planting Season": "season",
        }
    )[["plot", "crop_id", "crop", "crop_type", "area", "season"]].copy()
    out["plot"] = out["plot"].map(clean_text)
    out["crop"] = out["crop"].map(normalize_crop_name)
    out["crop_type"] = out["crop_type"].map(clean_text)
    out["crop_id"] = pd.to_numeric(out["crop_id"], errors="coerce").astype("Int64")
    out["area"] = pd.to_numeric(out["area"], errors="coerce").fillna(0.0)
    out["season"] = out["season"].map(season_to_int)
    out = out.dropna(subset=["crop_id"])
    out["crop_id"] = out["crop_id"].astype(int)
    return out[out["plot"] != ""].reset_index(drop=True)


def _clean_stats_2023(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(
        columns={
            "Crop ID": "crop_id",
            "Crop Name": "crop",
            "Plot Type": "plot_type",
            "Planting Season": "season",
            "Yield per Mu/jin": "yield_per_mu",
            "Planting Cost/(yuan/mu)": "cost_per_mu",
            "Sales Price/(yuan/jin)": "price_range",
        }
    )[["crop_id", "crop", "plot_type", "season", "yield_per_mu", "cost_per_mu", "price_range"]].copy()
    out["crop_id"] = pd.to_numeric(out["crop_id"], errors="coerce").astype("Int64")
    out["crop"] = out["crop"].map(normalize_crop_name)
    out["plot_type"] = out["plot_type"].map(normalize_plot_type)
    out["season"] = out["season"].map(season_to_int)
    out["yield_per_mu"] = pd.to_numeric(out["yield_per_mu"], errors="coerce").fillna(0.0)
    out["cost_per_mu"] = pd.to_numeric(out["cost_per_mu"], errors="coerce").fillna(0.0)
    out["price"] = out["price_range"].map(parse_price_midpoint)
    out = out.dropna(subset=["crop_id"])
    out["crop_id"] = out["crop_id"].astype(int)
    return out.reset_index(drop=True)


def _build_stat_maps(stats_df: pd.DataFrame) -> dict[str, dict[tuple[int, str, int], float]]:
    keys = ["crop_id", "plot_type", "season"]
    grouped = stats_df.groupby(keys, as_index=False).agg(
        yield_per_mu=("yield_per_mu", "mean"),
        cost_per_mu=("cost_per_mu", "mean"),
        price=("price", "mean"),
    )
    return {
        "yield": {
            (int(row.crop_id), row.plot_type, int(row.season)): float(row.yield_per_mu)
            for row in grouped.itertuples(index=False)
        },
        "cost": {
            (int(row.crop_id), row.plot_type, int(row.season)): float(row.cost_per_mu)
            for row in grouped.itertuples(index=False)
        },
        "price": {
            (int(row.crop_id), row.plot_type, int(row.season)): float(row.price)
            for row in grouped.itertuples(index=False)
        },
    }


def _estimate_2023_demand(
    planting_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    crop_name_by_id: dict[int, str],
    plot_type_by_plot: dict[str, str],
) -> dict[str, float]:
    yield_lookup = {
        (int(row.crop_id), row.plot_type, int(row.season)): float(row.yield_per_mu)
        for row in stats_df.itertuples(index=False)
    }
    rows = []
    for row in planting_df.itertuples(index=False):
        plot_type = plot_type_by_plot.get(row.plot, "")
        yield_mu = yield_lookup.get((int(row.crop_id), plot_type, int(row.season)), 0.0)
        rows.append((crop_name_by_id[int(row.crop_id)], float(row.area) * yield_mu))
    demand = pd.DataFrame(rows, columns=["crop", "production"]).groupby("crop")["production"].sum()
    return demand.to_dict()


def to_tuple_keyed_dicts(data: CropPlanningData) -> dict[str, dict]:
    return {
        "area_dict": data.area,
        "plot_type_dict": data.plot_type,
        "crop_type_dict": data.crop_type,
        "is_legume_dict": data.is_legume,
        "feasible_dict": data.feasible,
        "yield_dict": data.yield_per_mu,
        "cost_dict": data.cost_per_mu,
        "price_dict": data.price,
        "demand_dict": data.demand,
        "planted_2023_dict": data.planted_2023,
        "y_2023_dict": data.y_2023,
    }
