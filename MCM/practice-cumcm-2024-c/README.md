# CUMCM 2024 Problem C

This workspace contains a modular baseline for CUMCM 2024 Problem C.

## Data

The default official data path is:

```text
D:\MCM\CUMCM-2024\problems\CUMCM2024ProblemsE\ProblemC
```

The data layer reads:

- `Annex1.xlsx`
- `Annex2.xlsx`

and converts cleaned tables into tuple-keyed dictionaries for Gurobi, including:

- `area_dict = {plot: area}`
- `demand_dict = {(crop, year): demand}`
- `yield_dict = {(plot, crop, year, season): yield_per_mu}`
- `cost_dict = {(plot, crop, year, season): cost_per_mu}`
- `price_dict = {(crop, year): price}`

## Commands

Install dependencies in your Python environment:

```bash
pip install -r requirements.txt
```

Preview cleaned tuple-keyed data:

```bash
python scripts/prepare_data.py
```

Solve Problem 1 situation 1, where excess output is wasted:

```bash
python scripts/solve_plan.py --scenario p1 --excess-price-factor 0 --output outputs/result1_1_solution.csv --xlsx-output outputs/result1_1.xlsx
```

Solve Problem 1 situation 2, where excess output is sold at a 50% discount:

```bash
python scripts/solve_plan.py --scenario p1 --excess-price-factor 0.5 --output outputs/result1_2_solution.csv --xlsx-output outputs/result1_2.xlsx
```

If your Gurobi license reports `Model too large for size-limited license`, add `--rolling` to solve year-season subproblems sequentially:

```bash
python scripts/solve_plan.py --rolling --scenario p1 --excess-price-factor 0.5 --output outputs/result1_2_solution.csv --xlsx-output outputs/result1_2.xlsx
```

Run a simulated Problem 2 scenario:

```bash
python scripts/solve_plan.py --scenario p2 --excess-price-factor 0.5 --output outputs/result2_solution.csv --xlsx-output outputs/result2.xlsx
```

Run sensitivity analysis:

```bash
python scripts/sensitivity.py --target-crops Wheat Soybean --tracked-crop Soybean
```

## Notes

The MILP includes area limits, big-M planting logic, no consecutive replanting, three-year legume rotation, and production-sales linearization with normal and discounted excess sales.
