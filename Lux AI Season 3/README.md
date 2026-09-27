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

Both agents use the official `main.py` interface. Only `agent.py` differs in `v0`.

## Smoke checks

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
luxai-s3 agents/v0/main.py agents/starter/main.py --seed 101 --output replays/v0_vs_starter_101.json
luxai-s3 agents/starter/main.py agents/v0/main.py --seed 101 --output replays/starter_vs_v0_101.json
```

A single seed is an interface check, not evidence of strategy strength. Formal paired-seed evaluation and MLflow reporting are later stages. Replays and the downloaded official source are excluded from Git; the official commit above is the version pin.

