# V3 fixed-batch behavior-cloning check

`network.py` implements the requested 21-channel, 64-hidden-channel, four-block
residual network. Every convolution has stride 1. The 16 output planes are raw
macro-target logits indexed by replay unit slot; there is no unit query channel.
This model is currently a tensor/optimization sanity check, not a deployed agent.

`scripts/train_bc.py` samples 32 frames without replacement from existing NPZ
files using seed 42, requiring at least one valid intent per frame. It loads
that batch once and performs 500 Adam updates on exactly those same tensors.
There is no augmentation, dropout, validation set, or strategy modification.
The CUDA default is BF16 convolution with FP32 cross-entropy. Invalid unit slots
have zero loss and zero gradients; an entirely invalid batch skips the update.
Spatial class indices follow the parser's `[x, y]` convention: `class = x*24+y`.

Run in the existing ML workspace interpreter; no installation in the match-runner
virtual environment is needed. Choose an empty output directory for every run.

[WSL Bash]

```bash
cd "/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3"
/home/issue/ml-workspace/.venv/bin/python -m unittest discover -s tests -p test_bc.py -v
/home/issue/ml-workspace/.venv/bin/python scripts/train_bc.py \
  --batch-size 32 --epochs 500 --seed 42 --lr 0.001 \
  --device cuda --precision bf16 --output-dir outputs/bc_single_batch_seed42_reproduction
```

Artifacts include the 501-row loss/accuracy CSV (initial state plus 500 updates),
curve, checkpoint, summary, and manifest containing exact batch indices,
channel order, data/source hashes, runtime, Git state and MLflow identity.
The first run's report is at `outputs/bc_single_batch_seed42/report.md`.

Memorizing this batch cannot validate replay action/observation timing, expert
intent semantics, a consistent global x/y transpose, or unseen unit-slot identity.
Those remain separate questions before broader imitation learning.
