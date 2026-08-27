from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm2024c.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from cumcm2024c.data import build_crop_planning_data
from cumcm2024c.export import write_solution_to_template
from cumcm2024c.model import solve_crop_plan, write_solution_csv
from cumcm2024c.rolling import solve_crop_plan_rolling
from cumcm2024c.scenarios import make_problem2_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve CUMCM 2024 Problem C crop planning MILP.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "solution.csv")
    parser.add_argument("--xlsx-output", type=Path, default=None)
    parser.add_argument("--template-name", default=None)
    parser.add_argument("--scenario", choices=["p1", "p2"], default="p1")
    parser.add_argument("--excess-price-factor", type=float, default=0.5)
    parser.add_argument("--min-area-if-planted", type=float, default=0.0)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Solve year-season subproblems sequentially to fit size-limited Gurobi licenses.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_crop_planning_data(args.data_dir)
    if args.scenario == "p2":
        data = make_problem2_scenario(data, seed=args.seed)

    solver = solve_crop_plan_rolling if args.rolling else solve_crop_plan
    result = solver(
        data,
        excess_price_factor=args.excess_price_factor,
        min_area_if_planted=args.min_area_if_planted,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        output_flag=0 if args.quiet else 1,
    )
    write_solution_csv(result, args.output)
    if args.xlsx_output is not None:
        template_name = args.template_name
        if template_name is None:
            template_name = "result2.xlsx" if args.scenario == "p2" else "result1_2.xlsx"
            if args.excess_price_factor == 0:
                template_name = "result1_1.xlsx"
        write_solution_to_template(
            result,
            template_name=template_name,
            output_path=args.xlsx_output,
            data_dir=args.data_dir,
        )
    print(f"status={result.status}")
    print(f"objective_value={result.objective_value}")
    print(f"nonzero_planting_decisions={len(result.planted_area)}")
    print(f"wrote={args.output}")
    if args.xlsx_output is not None:
        print(f"wrote_xlsx={args.xlsx_output}")


if __name__ == "__main__":
    main()
