from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm2024c.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from cumcm2024c.data import CropPlanningData, build_crop_planning_data
from cumcm2024c.model import solve_crop_plan, total_crop_area
from cumcm2024c.rolling import solve_crop_plan_rolling


def solve_model(
    price_modifier: float,
    *,
    target_crops: set[str],
    tracked_crop: str,
    data: CropPlanningData,
    excess_price_factor: float = 0.5,
    time_limit: float | None = None,
    mip_gap: float | None = 0.01,
    rolling: bool = False,
) -> tuple[float, float]:
    modifiers = {
        (crop, 0): price_modifier
        for crop in target_crops
    }
    solver = solve_crop_plan_rolling if rolling else solve_crop_plan
    result = solver(
        data,
        excess_price_factor=excess_price_factor,
        time_limit=time_limit,
        mip_gap=mip_gap,
        output_flag=0,
        price_modifiers=modifiers,
    )
    if result.objective_value is None:
        raise RuntimeError(f"Gurobi did not find a solution; status={result.status}")
    return result.objective_value, total_crop_area(result, tracked_crop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run price sensitivity analysis for CUMCM 2024 C.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "sensitivity")
    parser.add_argument("--target-crops", nargs="+", default=["Wheat", "Soybean"])
    parser.add_argument("--tracked-crop", default="Soybean")
    parser.add_argument("--excess-price-factor", type=float, default=0.5)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--rolling", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_crop_planning_data(args.data_dir)

    records = []
    for pct in range(-20, 21, 5):
        modifier = 1.0 + pct / 100.0
        profit, tracked_area = solve_model(
            modifier,
            target_crops=set(args.target_crops),
            tracked_crop=args.tracked_crop,
            data=data,
            excess_price_factor=args.excess_price_factor,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            rolling=args.rolling,
        )
        records.append(
            {
                "price_change_pct": pct,
                "price_modifier": modifier,
                "profit": profit,
                f"{args.tracked_crop}_area_mu": tracked_area,
            }
        )
        print(f"price_change={pct:+d}% profit={profit:.2f} {args.tracked_crop}_area={tracked_area:.2f}")

    df = pd.DataFrame(records)
    csv_path = args.output_dir / "sensitivity_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _plot_profit(df, args.output_dir / "profit_sensitivity.png")
    _plot_area(df, args.tracked_crop, args.output_dir / "tracked_crop_area_sensitivity.png")
    print(f"wrote={csv_path}")


def _setup_chinese_font() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_profit(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.plot(df["price_change_pct"], df["profit"], marker="o", linewidth=2, label="最大总利润")
    ax.set_xlabel("核心作物价格波动率/%")
    ax.set_ylabel("总利润/元")
    ax.set_title("价格扰动对总利润的影响")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_area(df: pd.DataFrame, tracked_crop: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_chinese_font()
    col = f"{tracked_crop}_area_mu"
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.bar(df["price_change_pct"], df[col], width=3.5, label=f"{tracked_crop} 种植面积")
    ax.set_xlabel("核心作物价格波动率/%")
    ax.set_ylabel("种植面积/亩")
    ax.set_title(f"价格扰动对 {tracked_crop} 种植面积的影响")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
