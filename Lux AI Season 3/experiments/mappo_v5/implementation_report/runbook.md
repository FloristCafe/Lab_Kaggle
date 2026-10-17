# MAPPO v5：用户执行手册

全部在 VS Code 的 Ubuntu-26.04 终端执行。按顺序逐项运行；Agent不会自动接续。
当前只交付100个完整大局的pilot，未提供1000局正式训练配置。每组使用训练seed42，完整大局包含五小局。
E0没有训练，评估4个dev环境seed、双方交换阵营，共8局；E1/E2各训练100局，再用相同8局评估。

先设置目录和日志。`pipefail`让终端保留真实退出状态；显存打印不是完成信号，完成以`summary.json`为准。

[WSL Bash]
```bash
cd '/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3'
mkdir -p outputs/mappo_v5
set -o pipefail
```

## 1. 原始Frozen IL基线 E0

[WSL Bash]
```bash
/home/issue/ml-workspace/.venv/bin/python -u scripts/train_mappo_v5.py \
  --config configs/mappo_v5/E0_frozen.yaml \
  --output-dir experiments/mappo_v5/E0_frozen_seed42 \
  --seed 42 --run-pilot 2>&1 | tee outputs/mappo_v5/E0_frozen_seed42.log
```

## 2. 匹配确定性Gate的Frozen诊断

它保留相同IL权重，用于区分执行Gate改变与RL更新收益。

[WSL Bash]
```bash
/home/issue/ml-workspace/.venv/bin/python -u scripts/train_mappo_v5.py \
  --config configs/mappo_v5/E0_gate.yaml \
  --output-dir experiments/mappo_v5/E0_gate_seed42 \
  --seed 42 --run-pilot 2>&1 | tee outputs/mappo_v5/E0_gate_seed42.log
```

## 3. E1：确定性Gate，学习Mode/Goal

[WSL Bash]
```bash
/home/issue/ml-workspace/.venv/bin/python -u scripts/train_mappo_v5.py \
  --config configs/mappo_v5/E1_GM_pilot.yaml \
  --output-dir experiments/mappo_v5/E1_GM_seed42 \
  --seed 42 --run-pilot 2>&1 | tee outputs/mappo_v5/E1_GM_seed42.log
```

## 4. E2主实验：中性R初始化，联合学习R/Mode/Goal

仅重置R输出层的weight/bias，初始自由R概率0.5；其他IL权重不变。R LR=1e-4。

[WSL Bash]
```bash
/home/issue/ml-workspace/.venv/bin/python -u scripts/train_mappo_v5.py \
  --config configs/mappo_v5/E2_RMG_pilot.yaml \
  --output-dir experiments/mappo_v5/E2_RMG_seed42 \
  --seed 42 --run-pilot 2>&1 | tee outputs/mappo_v5/E2_RMG_seed42.log
```

## 5. 原始R初始化对照

保留原始IL R，R LR=3e-5。该组与主E2同时改变初始化和R LR，不可将差异纯归因于重初始化。
如需分离两者的因果效应，另行设计匹配LR的对照，本轮不自动增加实验。

[WSL Bash]
```bash
/home/issue/ml-workspace/.venv/bin/python -u scripts/train_mappo_v5.py \
  --config configs/mappo_v5/E2_original_pilot.yaml \
  --output-dir experiments/mappo_v5/E2_original_seed42 \
  --seed 42 --run-pilot 2>&1 | tee outputs/mappo_v5/E2_original_seed42.log
```

额外准备了`configs/mappo_v5/E0_neutral.yaml`冻结诊断。中性R的greedy阈值将0.5平局判为Replan，因此不能把greedy行为未变化解释为中性化无效；训练使用随机R采样。

## 完成后只读汇总

[WSL Bash]
```bash
/home/issue/ml-workspace/.venv/bin/python scripts/report_mappo_v5_pilot.py \
  --run-dirs experiments/mappo_v5/E0_frozen_seed42 \
             experiments/mappo_v5/E0_gate_seed42 \
             experiments/mappo_v5/E1_GM_seed42 \
             experiments/mappo_v5/E2_RMG_seed42 \
             experiments/mappo_v5/E2_original_seed42 \
  --output-dir experiments/mappo_v5/pilot_comparison_seed42
```

## 断点恢复

仅在完整大局边界保存`last.pth`。中途退出会从上一个保存的大局重新开始；环境隐藏状态不序列化。
恢复必须保持相同配置、seed、源码、IL权重与输出目录，且还没有完成summary。

[WSL Bash]
```bash
/home/issue/ml-workspace/.venv/bin/python -u scripts/train_mappo_v5.py \
  --config configs/mappo_v5/E2_RMG_pilot.yaml \
  --output-dir experiments/mappo_v5/E2_RMG_seed42 \
  --seed 42 --run-pilot --resume 2>&1 | tee -a outputs/mappo_v5/E2_RMG_seed42.log
```

每个run保存：`manifest.json`、`config.yaml`、`seed_plan.json`、`source/`、`initial.pth`、`last.pth`、
`update_metrics.json`、`training_metrics.json`、`evaluation_metrics.json`、首末大局的小份真实rollout、`worker.log`和`attempt_N.json`。
`last.pth`包含Actor、Critic、优化器、RNG与已完成进度；旧IL checkpoint保持不变。
每个训练update另存`rollouts/attemptN/update_*.npz`，包含全部真实state/next-state、previous intent、动作、概率、压位掩码、tau、奖励和终止标记；ZLIB无损压缩，不额外量化输入。
每run约2GiB rollout预算、4GiB可用空间保留线；达到保护线会停止，保留最后完整大局checkpoint。中断局的旧journal也计入空间预算。
评估逐决策另存`evaluation_trace_*.json`；其log density只是潜在随机策略的诊断量，不能作greedy行为的PPO old probability，更新入口会拒绝greedy样本。
MLflow实验名`lux-s3-mappo-v5`，每次续跑独立attempt，不能只用最后一次attempt的wall统计总成本。

## Pilot判据

- 工程：所有训练为完整505步，tau为实际步数，匹配奖励与终止；更新前概率误差≤1e-4，loss/梯度/参数有限，无目标冲突。
- 价值：检查非零终局回报、固定target的critic MSE前后、explained variance与曲线；短测试只有零官方奖励，不能证明critic已学到游戏价值。
- 行为：分开看forced与free R、Continue、mode分布、目标距离/持久性/完成、原地停滞和战术覆盖；不能靠R1 F1或总体mode占比判断成功。
- 效果：E1相对匹配Gate E0，E2相对E1；按相同dev seed和阵营配对。4个dev seed不足以证明最终强度，单训练seed不足以证明稳健性。
- 若有坍缩或Critic/奖励异常，先定位；不得自动开始1000局。通过pilot科研复核后，再配置约1000局及多个训练seed。
