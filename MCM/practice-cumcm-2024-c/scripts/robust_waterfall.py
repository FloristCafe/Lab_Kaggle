from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm2024c.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from cumcm2024c.data import CropPlanningData, build_crop_planning_data
from cumcm2024c.evaluation import evaluate_solution_profit, read_solution_csv
from cumcm2024c.scenarios import make_problem2_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Q2 robust-cost waterfall chart.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--q1-solution", type=Path, default=DEFAULT_OUTPUT_DIR / "result1_2_solution.csv")
    parser.add_argument("--q2-solution", type=Path, default=DEFAULT_OUTPUT_DIR / "result2_solution.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "waterfall")
    parser.add_argument("--excess-price-factor", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = build_crop_planning_data(args.data_dir)
    robust = make_problem2_scenario(base)
    q1_solution = read_solution_csv(args.q1_solution)
    q2_solution = read_solution_csv(args.q2_solution)

    q1_profit = evaluate_solution_profit(q1_solution, base, excess_price_factor=args.excess_price_factor)["profit"]
    q2_profit = evaluate_solution_profit(q2_solution, robust, excess_price_factor=args.excess_price_factor)["profit"]

    base_on_q2 = evaluate_solution_profit(q2_solution, base, excess_price_factor=args.excess_price_factor)["profit"]
    yield_only = evaluate_solution_profit(
        q2_solution,
        _mix_scenario(base, robust, use_yield=True),
        excess_price_factor=args.excess_price_factor,
    )["profit"]
    cost_only = evaluate_solution_profit(
        q2_solution,
        _mix_scenario(base, robust, use_cost=True),
        excess_price_factor=args.excess_price_factor,
    )["profit"]
    price_only = evaluate_solution_profit(
        q2_solution,
        _mix_scenario(base, robust, use_price=True),
        excess_price_factor=args.excess_price_factor,
    )["profit"]
    wheat_corn_demand_only = evaluate_solution_profit(
        q2_solution,
        _wheat_corn_demand_scenario(base, robust),
        excess_price_factor=args.excess_price_factor,
    )["profit"]

    raw = {
        "产量下降损失": max(0.0, base_on_q2 - yield_only),
        "成本上升损失": max(0.0, base_on_q2 - cost_only),
        "价格恶化损失": max(0.0, base_on_q2 - price_only),
        "小麦玉米销量增加对冲": max(0.0, wheat_corn_demand_only - base_on_q2),
    }
    robust_cost = q1_profit - q2_profit
    raw_net_loss = raw["产量下降损失"] + raw["成本上升损失"] + raw["价格恶化损失"] - raw["小麦玉米销量增加对冲"]
    scale = robust_cost / raw_net_loss if abs(raw_net_loss) > 1e-9 else 1.0
    allocated = {
        "产量下降损失": raw["产量下降损失"] * scale,
        "成本上升损失": raw["成本上升损失"] * scale,
        "价格恶化损失": raw["价格恶化损失"] * scale,
        "小麦玉米销量增加对冲": raw["小麦玉米销量增加对冲"] * scale,
    }

    table = pd.DataFrame(
        [
            {"item": "Q1确定性总利润", "type": "start", "raw_yuan": q1_profit, "allocated_yuan": q1_profit},
            {"item": "产量下降损失", "type": "loss", "raw_yuan": raw["产量下降损失"], "allocated_yuan": -allocated["产量下降损失"]},
            {"item": "成本上升损失", "type": "loss", "raw_yuan": raw["成本上升损失"], "allocated_yuan": -allocated["成本上升损失"]},
            {"item": "价格恶化损失", "type": "loss", "raw_yuan": raw["价格恶化损失"], "allocated_yuan": -allocated["价格恶化损失"]},
            {"item": "小麦玉米销量增加对冲", "type": "gain", "raw_yuan": raw["小麦玉米销量增加对冲"], "allocated_yuan": allocated["小麦玉米销量增加对冲"]},
            {"item": "Q2鲁棒保底总利润", "type": "end", "raw_yuan": q2_profit, "allocated_yuan": q2_profit},
        ]
    )
    table["allocated_wan"] = table["allocated_yuan"] / 10000.0
    table["raw_wan"] = table["raw_yuan"] / 10000.0
    table.to_csv(args.output_dir / "robust_waterfall_components.csv", index=False, encoding="utf-8-sig")

    summary = {
        "q1_profit_yuan": q1_profit,
        "q2_profit_yuan": q2_profit,
        "robust_cost_yuan": robust_cost,
        "base_profit_of_q2_solution_yuan": base_on_q2,
        "raw_component_yuan": raw,
        "raw_net_loss_yuan": raw_net_loss,
        "allocation_scale": scale,
        "allocated_component_yuan": allocated,
    }
    (args.output_dir / "robust_waterfall_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_waterfall(table, args.output_dir / "robust_waterfall.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote={args.output_dir / 'robust_waterfall.png'}")


def _mix_scenario(
    base: CropPlanningData,
    robust: CropPlanningData,
    *,
    use_yield: bool = False,
    use_cost: bool = False,
    use_price: bool = False,
    use_demand: bool = False,
) -> CropPlanningData:
    return CropPlanningData(
        plots=base.plots,
        crops=base.crops,
        years=base.years,
        seasons=base.seasons,
        area=base.area,
        plot_type=base.plot_type,
        crop_id=base.crop_id,
        crop_name_by_id=base.crop_name_by_id,
        crop_type=base.crop_type,
        is_legume=base.is_legume,
        feasible=base.feasible,
        yield_per_mu=robust.yield_per_mu if use_yield else base.yield_per_mu,
        cost_per_mu=robust.cost_per_mu if use_cost else base.cost_per_mu,
        price=robust.price if use_price else base.price,
        demand=robust.demand if use_demand else base.demand,
        planted_2023=base.planted_2023,
        y_2023=base.y_2023,
    )


def _wheat_corn_demand_scenario(base: CropPlanningData, robust: CropPlanningData) -> CropPlanningData:
    demand = dict(base.demand)
    for crop in ("Wheat", "Corn"):
        for year in base.years:
            demand[crop, year] = robust.demand[crop, year]
    mixed = _mix_scenario(base, robust)
    return CropPlanningData(
        plots=mixed.plots,
        crops=mixed.crops,
        years=mixed.years,
        seasons=mixed.seasons,
        area=mixed.area,
        plot_type=mixed.plot_type,
        crop_id=mixed.crop_id,
        crop_name_by_id=mixed.crop_name_by_id,
        crop_type=mixed.crop_type,
        is_legume=mixed.is_legume,
        feasible=mixed.feasible,
        yield_per_mu=mixed.yield_per_mu,
        cost_per_mu=mixed.cost_per_mu,
        price=mixed.price,
        demand=demand,
        planted_2023=mixed.planted_2023,
        y_2023=mixed.y_2023,
    )


def _plot_waterfall(table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = table["item"].tolist()
    values = table["allocated_wan"].tolist()
    running = [values[0]]
    for value in values[1:-1]:
        running.append(running[-1] + value)
    starts = [0.0]
    starts.extend(running[:-1])
    starts.append(0.0)

    colors = []
    heights = []
    bottoms = []
    for idx, row in table.iterrows():
        value = float(row["allocated_wan"])
        if row["type"] in {"start", "end"}:
            colors.append("#386cb0")
            heights.append(value)
            bottoms.append(0.0)
        elif value < 0:
            colors.append("#d95f5f")
            heights.append(abs(value))
            bottoms.append(starts[idx] + value)
        else:
            colors.append("#4daf4a")
            heights.append(value)
            bottoms.append(starts[idx])

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
    ax.bar(range(len(labels)), heights, bottom=bottoms, color=colors, edgecolor="#333333", linewidth=0.8)
    for idx, (bottom, height, value) in enumerate(zip(bottoms, heights, values)):
        y = bottom + height
        text = f"{value:+.2f}" if idx not in {0, len(values) - 1} else f"{value:.2f}"
        ax.text(idx, y, text, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("金额/万元")
    ax.set_title("Q2恶劣情景鲁棒代价瀑布图")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
