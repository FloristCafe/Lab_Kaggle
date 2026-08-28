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

and converts cleaned tables into tuple-keyed dictionaries for PySCIPOpt, including:

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

Run a simulated Problem 2 scenario:

```bash
python scripts/solve_plan.py --scenario p2 --excess-price-factor 0.5 --output outputs/result2_solution.csv --xlsx-output outputs/result2.xlsx
```

Problem 2 uses a deterministic worst-case scenario generated from 2023 baselines:

- Wheat and corn demand grows by `5%` each year from 2023.
- Other crop demand is fixed at `95%` of 2023 demand.
- All yields are fixed at `90%` of 2023 yield.
- Costs compound by `5%` per year with 2024 as the base year.
- Grain prices stay fixed; vegetable prices grow by `5%` per year; edible fungi prices decline by `5%` per year.

Run the Problem 3 MIQP risk-aversion frontier:

```bash
python scripts/pareto_frontier.py --base-scenario p2 --lambda-values 0 10 50 100 200 500 1000 --quiet
```

The MIQP builds a simulated crop price-return covariance matrix from economic category correlations, then solves the same global seven-year planning model with objective `Expected Profit - lambda * Portfolio Risk`.

SCIP logs are written under `outputs/scip_logs` for `solve_plan.py` and under `outputs/pareto/scip_logs` for `pareto_frontier.py`.

Draw the Q2 robust-cost waterfall chart:

```bash
python scripts/robust_waterfall.py --q1-solution outputs/result1_2_solution.csv --q2-solution outputs/result2_solution.csv
```

The waterfall chart evaluates the Q2 planting plan under partial scenarios for yield loss, cost growth, price deterioration, and wheat/corn demand hedging, then writes `outputs/waterfall/robust_waterfall.png` and the component table.

Run sensitivity analysis:

```bash
python scripts/sensitivity.py --target-crops Wheat Soybean --tracked-crop Soybean
```

## Notes

The MILP includes area limits, big-M planting logic, no consecutive replanting, three-year legume rotation, and production-sales linearization with normal and discounted excess sales. The seven-year planning horizon is modeled as one global MILP and solved with a single `model.optimize()` call.

The default minimum planted area is `0.1` mu whenever a binary planting decision is active. This prevents `y=1, x=0` from satisfying rotation constraints without producing a real planting area.
