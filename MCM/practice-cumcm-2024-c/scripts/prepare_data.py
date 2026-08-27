from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm2024c.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from cumcm2024c.data import build_crop_planning_data, to_tuple_keyed_dicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean CUMCM 2024 C Excel data into tuple-keyed dicts.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "data_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_crop_planning_data(args.data_dir)
    dicts = to_tuple_keyed_dicts(data)

    summary = {
        "num_plots": len(data.plots),
        "num_crops": len(data.crops),
        "years": data.years,
        "seasons": data.seasons,
        "sample_area_dict": dict(list(dicts["area_dict"].items())[:5]),
        "sample_demand_dict": {
            str(key): value for key, value in list(dicts["demand_dict"].items())[:8]
        },
        "sample_yield_dict": {
            str(key): value for key, value in list(dicts["yield_dict"].items())[:8]
        },
        "sample_cost_dict": {
            str(key): value for key, value in list(dicts["cost_dict"].items())[:8]
        },
        "sample_price_dict": {
            str(key): value for key, value in list(dicts["price_dict"].items())[:8]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote summary to {args.output}")


if __name__ == "__main__":
    main()

