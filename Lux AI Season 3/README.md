# Lux AI Season 3 research simulation

This project simulates the NeurIPS 2024 Lux AI Season 3 competition locally. It uses the official environment at commit `c6d1e665d12a467b299fb586b3f876e47847746b` and keeps the official Python starter unchanged in `agents/starter`.

## Environment

Run all commands in Ubuntu-26.04, on the WSL ext4 filesystem. The official runner starts agents with `python` from `PATH`, so activate this project's environment before matches.

```bash
cd "/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3"
git clone https://github.com/Lux-AI-Challenge/Lux-Design-S3.git vendor/lux-design-s3
git -C vendor/lux-design-s3 checkout c6d1e665d12a467b299fb586b3f876e47847746b
uv python install 3.11
uv venv --python 3.11 .venv
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install --python .venv/bin/python -e ./vendor/lux-design-s3/src
source .venv/bin/activate
```

The warning about a missing CUDA-enabled `jaxlib` is expected: the current simulation uses CPU.

## Agents

- `agents/starter`: official Python starter kit, including its `lux` helpers.
- `agents/v0`: observation-only planning baseline. It remembers visible terrain and relics across matches, explores unobserved tiles, routes around known asteroids, spreads units over relic areas, and waits if energy is insufficient to move. It does not infer scoring tiles or use sap actions yet.
- `agents/v1`: rule teacher for macro-action research. An observation-only relic solver estimates scoring tiles; a dispatcher builds all unit/target scores, globally assigns distinct targets, then executes one path step. The agent does not use sap. Its teacher labels are hypotheses for supervised learning, not proven optimal actions.

Both agents use the official `main.py` interface. Only `agent.py` differs in `v0`.

## Smoke checks

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
luxai-s3 agents/v0/main.py agents/starter/main.py --seed 101 --output replays/v0_vs_starter_101.json
luxai-s3 agents/starter/main.py agents/v0/main.py --seed 101 --output replays/starter_vs_v0_101.json
```

A single seed is an interface check, not evidence of strategy strength. Formal paired-seed evaluation and MLflow reporting are later stages. Replays and the downloaded official source are excluded from Git; the official commit above is the version pin.

## V1 rule teacher

The environment is run by this project's Python 3.11/JAX environment. The schedule script is run by `/home/issue/ml-workspace/.venv/bin/python` so it can record results in the existing MLflow database. Both run in WSL on ext4. It does not execute matches until you call it.

The first `v1_dev` smoke check (seeds 102 and 103, both sides) finished without crashes but lost all four full games to `v0`. Its traces exposed repeated target stacking, so they must not be pooled with data from the corrected dispatcher. The corrected `v1_injective_dev` check used five paired seeds, or ten games. Its saved manifest pins the pre-decay code; the current agent cannot be rerun into that suite.

The dispatcher then gained one-step marginal information decay: a tile occupied in the previous observation receives no information bonus, while its expected scoring value remains. The `v1_marginal_info_dev` suite used the same seeds and sides:

```bash
# [WSL Bash]
cd "/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3"
/home/issue/ml-workspace/.venv/bin/python scripts/run_v1_schedule.py \
  --seed-start 102 --seed-count 5 --opponents v0 --trace \
  --suite v1_marginal_info_dev
```

Regenerate the paired comparison from saved results and traces with:

```bash
# [WSL Bash]
cd "/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3"
.venv/bin/python scripts/compare_v1_suites.py \
  v1_injective_dev v1_marginal_info_dev \
  --output outputs/v1_marginal_info_dev/comparison.md
```

The comparison found 1/10 full-game wins in both suites. Early match wins changed from 0/20 to 1/20, while late match wins changed from 19/20 to 18/20. Match-1 cumulative visited tiles rose from 199.5 to 223.2 on average, but first confirmed scoring tiles appeared later on average (episode step 298.3 to 304.2); neither suite confirmed any in the first two matches. These five paired seeds are descriptive, and the 50-seed final holdout remains untouched. Do not promote these traces to SL expert labels based on footprint alone.

The trace writer is enabled only by `LUX_V1_TRACE_DIR`, which the schedule script sets with `--trace`. The marginal-information suite records the observation-derived 22-channel map, including previous occupancy, plus per-unit relative enemy and previous-target features, path cost and reachability maps, and the rule-assigned target. It never reads raw replay state. Convert the durable per-step frames to compressed NPZ after a completed run:

```bash
# [WSL Bash]
cd "/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3"
.venv/bin/python scripts/pack_v1_traces.py artifacts/v1_traces/v1_marginal_info_dev/*.frames
```

For a fixed-target path probe, set `LUX_V1_FIXED_TARGET=x,y` when invoking `luxai-s3` manually. It forces the first active unit toward that tile and is for path diagnostics only; do not use its trace as an SL expert label. A target can be reachable on the known map but fail after unknown terrain, energy drift, or simultaneous opponent moves. Record arrival, stalls, deaths, and invalid actions rather than treating 100% arrival as a prerequisite.

Important training boundary: the current teacher may change its target every environment step. A heatmap is a macro target representation, but it only shortens the RL decision horizon when the later policy commits to targets or makes event-triggered decisions. The NPZ data supplies behavior-cloning labels; it does not by itself justify IMPALA or prove the teacher is strong.
