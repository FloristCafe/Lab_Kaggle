from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm2024c.config import DEFAULT_DATA_DIR
from cumcm2024c.data import build_crop_planning_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check generated CUMCM 2024 C solution CSV files.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_crop_planning_data(args.data_dir)
    df = pd.read_csv(args.csv)
    required = {"plot", "crop", "year", "season", "area_mu"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df["year"] = df["year"].astype(int)
    df["season"] = df["season"].astype(int)
    df["area_mu"] = pd.to_numeric(df["area_mu"], errors="coerce").fillna(0.0)
    df = df[df["area_mu"] > args.tol].copy()

    checks = {
        "rows": len(df),
        "total_area_mu": float(df["area_mu"].sum()),
        "area_violations": _area_violations(df, data, args.tol),
        "feasibility_violations": _feasibility_violations(df, data),
        "immediate_replant_violations": _immediate_replant_violations(df, data),
        "legume_rotation_violations": _legume_rotation_violations(df, data),
    }

    print(f"file={args.csv}")
    print(f"rows={checks['rows']}")
    print(f"total_area_mu={checks['total_area_mu']:.4f}")
    for key in [
        "area_violations",
        "feasibility_violations",
        "immediate_replant_violations",
        "legume_rotation_violations",
    ]:
        violations = checks[key]
        print(f"{key}={len(violations)}")
        for item in violations[:10]:
            print(f"  {item}")


def _area_violations(df: pd.DataFrame, data, tol: float) -> list[tuple]:
    used = df.groupby(["plot", "year", "season"], as_index=False)["area_mu"].sum()
    out = []
    for row in used.itertuples(index=False):
        limit = data.area.get(row.plot)
        if limit is None or row.area_mu > limit + tol:
            out.append((row.plot, int(row.year), int(row.season), float(row.area_mu), limit))
    return out


def _feasibility_violations(df: pd.DataFrame, data) -> list[tuple]:
    out = []
    for row in df.itertuples(index=False):
        if not data.feasible.get((row.plot, row.crop, int(row.season)), False):
            out.append((row.plot, row.crop, int(row.year), int(row.season)))
    return out


def _immediate_replant_violations(df: pd.DataFrame, data) -> list[tuple]:
    planted = {
        (row.plot, row.crop, int(row.year), int(row.season))
        for row in df.itertuples(index=False)
    }
    out = []
    for plot, crop, year, season in sorted(planted):
        if season == 1:
            previous = any(
                (plot, crop, year - 1, s) in planted
                or (year == 2024 and data.y_2023.get((plot, crop, s), 0))
                for s in data.seasons
            )
        else:
            previous = (plot, crop, year, 1) in planted
        if previous:
            out.append((plot, crop, year, season))
    return out


def _legume_rotation_violations(df: pd.DataFrame, data) -> list[tuple]:
    legume_crops = {crop for crop, is_legume in data.is_legume.items() if is_legume}
    planted = {
        (row.plot, row.crop, int(row.year), int(row.season))
        for row in df.itertuples(index=False)
        if row.crop in legume_crops
    }
    planted |= {
        (plot, crop, 2023, season)
        for (plot, crop, season), used in data.y_2023.items()
        if used and crop in legume_crops
    }

    out = []
    for plot in data.plots:
        for start in range(2023, 2029):
            years = {start, start + 1, start + 2}
            if not any(p == plot and year in years for p, _crop, year, _season in planted):
                out.append((plot, start, start + 2))
    return out


if __name__ == "__main__":
    main()
