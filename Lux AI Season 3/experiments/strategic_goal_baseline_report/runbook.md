# 复现与产物导航

正式结果见 `research_report.md`。所有命令在项目 WSL 目录执行；输出目录必须是新的，以免覆盖已完成的实验。

当前数据与实验全部保存在 WSL ext4。此交付包只复制报告、源码、模型和日志，不复制 725 场原始 JSON 或训练 NPZ。`dataset/manifest.json` 保存 400 场 NPZ 与原始 JSON 的内容哈希。

## 产物

- `experiments/A_waypoint_reproduction`：旧最佳 checkpoint 的实际复评。
- `experiments/B_strategic_goal_il`：400 场、320/80 分组、6 epoch 训练，最佳第 5 epoch。
- `experiments/CDE_goal_mappo_v2`：修正生命周期统计后的最终三组在线对照及权重。
- `experiments/CDE_goal_mappo`：初轮对照，保留作审计；不用于最终战术统计。
- `experiments/strategic_audit`：数据哈希、模式分布、因果截断检查、标签示例、初轮精确执行源码。
- `source`：当前新增实现、测试与配置。
- `logs`：建集、IL、两轮 RL、测试、报告日志。

## 命令

下列仅供复现，不会自动执行。当前冻结版存在报告中列明的 SAP 尝试标签污染和 EXPLORE 失败；扩大训练之前应先修复与重新版本化标签。

[WSL Bash]
```bash
cd '/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3'
/home/issue/ml-workspace/.venv/bin/python -m unittest discover -s tests -p 'test_strategic*.py' -v
.venv/bin/python scripts/check_official_combat.py
```

从已有 400 场 NPZ 复现 Goal IL（选择新的输出名）：

[WSL Bash]
```bash
/home/issue/ml-workspace/.venv/bin/python -u scripts/train_strategic_il.py \
  --data-dir data/dataset_v2_goal \
  --output-dir experiments/B_strategic_goal_il_reproduction \
  --epochs 6 --min-games 400 --fold 0 --seed 42
```

以指定 IL checkpoint 复现小预算对照；预算与种子从保存的配置读取：

[WSL Bash]
```bash
/home/issue/ml-workspace/.venv/bin/python -u scripts/train_goal_mappo.py \
  --il-checkpoint experiments/B_strategic_goal_il/best_model.pth \
  --config configs/strategic_baseline.yaml \
  --output-dir experiments/CDE_goal_mappo_reproduction \
  --arms random_mappo il_mappo il_only --seed 42
```

从本地原始 JSON 重新建集，使用新目录；绝不删除 JSON：

[WSL Bash]
```bash
/home/issue/ml-workspace/.venv/bin/python -u scripts/build_strategic_dataset.py \
  --limit 400 --output-dir data/dataset_v2_goal_reproduction
```

源/配置哈希改变时，builder 会拒绝将结果混入旧数据目录。运行中的日志可在 `outputs/` 中保存；完成后以实验目录中的 `summary.json`、`validation_log.json`、MLflow FINISHED 状态为准，单条 CUDA 显存输出不能证明整个实验完成。

模型文件是由本项目产生的 PyTorch checkpoint，应与同版本源码、配置、数据清单一同使用。新 factorized Actor 与旧 35-channel waypoint 网络不兼容。
