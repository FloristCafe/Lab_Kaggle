from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm2024c.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from cumcm2024c.data import build_crop_planning_data
from cumcm2024c.model import solve_crop_plan_miqp, write_solution_csv
from cumcm2024c.risk import simulate_price_covariance
from cumcm2024c.scenarios import make_problem2_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIQP risk-aversion frontier for CUMCM 2024 C.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "pareto")
    parser.add_argument("--base-scenario", choices=["p1", "p2"], default="p2")
    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=[0, 10, 50, 100, 200, 500, 1000],
        help="Explicit risk-aversion coefficients to test.",
    )
    parser.add_argument("--lambda-min", type=float, default=None)
    parser.add_argument("--lambda-max", type=float, default=None)
    parser.add_argument("--lambda-step", type=float, default=None)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--excess-price-factor", type=float, default=0.5)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-solutions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "scip_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    data = build_crop_planning_data(args.data_dir)
    if args.base_scenario == "p2":
        data = make_problem2_scenario(data)

    risk_data = simulate_price_covariance(data, n_samples=args.samples, seed=args.seed)
    risk_data.sample_covariance.to_csv(args.output_dir / "sample_price_return_covariance.csv", encoding="utf-8-sig")
    risk_data.correlation.to_csv(args.output_dir / "category_correlation_matrix.csv", encoding="utf-8-sig")

    records = []
    for risk_lambda in _lambda_values(args):
        log_file = log_dir / f"miqp_lambda_{risk_lambda:.2f}.log"
        result = solve_crop_plan_miqp(
            data,
            cov_dict=risk_data.cov_dict,
            risk_lambda=risk_lambda,
            excess_price_factor=args.excess_price_factor,
            output_flag=0 if args.quiet else 1,
            log_file=log_file,
        )
        metadata = result.metadata or {}
        row = {
            "risk_lambda": risk_lambda,
            "status": result.status,
            "status_name": _status_name(result.status),
            "objective_value": result.objective_value,
            "expected_profit": metadata.get("expected_profit"),
            "portfolio_risk": metadata.get("portfolio_risk"),
            "nonzero_planting_decisions": len(result.planted_area),
            "log_file": str(log_file),
        }
        records.append(row)
        print(
            f"lambda={risk_lambda:.2f} status={row['status_name']} "
            f"expected_profit={row['expected_profit']} risk={row['portfolio_risk']}"
        )
        if args.save_solutions and result.objective_value is not None:
            write_solution_csv(result, args.output_dir / f"solution_lambda_{risk_lambda:.2f}.csv")

    frontier = pd.DataFrame(records)
    frontier_path = args.output_dir / "pareto_frontier.csv"
    frontier.to_csv(frontier_path, index=False, encoding="utf-8-sig")
    _plot_frontier(frontier, args.output_dir / "pareto_frontier.png")

    summary = {
        "base_scenario": args.base_scenario,
        "samples": args.samples,
        "seed": args.seed,
        "lambda_values": _lambda_values(args),
        "frontier_csv": str(frontier_path),
        "frontier_png": str(args.output_dir / "pareto_frontier.png"),
    }
    (args.output_dir / "pareto_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote={frontier_path}")


def _lambda_grid(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + idx * step, 10) for idx in range(count + 1)]


def _lambda_values(args: argparse.Namespace) -> list[float]:
    if args.lambda_min is not None or args.lambda_max is not None or args.lambda_step is not None:
        if args.lambda_min is None or args.lambda_max is None or args.lambda_step is None:
            raise ValueError("--lambda-min, --lambda-max, and --lambda-step must be provided together.")
        return _lambda_grid(args.lambda_min, args.lambda_max, args.lambda_step)
    return [float(value) for value in args.lambda_values]


def _plot_frontier(frontier: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    valid = frontier.dropna(subset=["expected_profit", "portfolio_risk"])
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        valid["portfolio_risk"],
        valid["expected_profit"],
        marker="o",
        linewidth=2,
        label="帕累托有效前沿",
    )
    for row in valid.itertuples(index=False):
        ax.annotate(f"{row.risk_lambda:.1f}", (row.portfolio_risk, row.expected_profit), fontsize=8)
    ax.set_xlabel("组合风险")
    ax.set_ylabel("预期利润/元")
    ax.set_title("预期利润-组合风险帕累托前沿")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _status_name(status: str) -> str:
    return str(status).upper()


if __name__ == "__main__":
    main()
