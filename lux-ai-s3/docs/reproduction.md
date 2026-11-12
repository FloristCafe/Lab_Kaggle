# 复现说明与资产边界

## 先区分三个层次

1. **仓库内可独立复算**：证据SHA、G05/G06/A01/A02/D12地图配对点估计、BC选模、A训练1200局官方胜数记录。无需模型或环境。
2. **当前机器可加载与预检**：数据/权重在外部artifact目录；环境依赖已存在；可验证历史BC与A最终checkpoint。
3. **完整历史实验复跑**：旧Runner、全部原始日志、IL旧来源等保留外部归档。不能把仅有本仓库解释成所有研究的一键复现，旧400场来源缺失仍然存在。

本轮没有重启训练或官方环境，只恢复与静态检查。以下长程命令是方法记录，必须由人另行决定执行，不是当前推荐继续训练的计划。关闭的Goal Selector、Credit、评分调参和复审优化路线不自动恢复。

## A. 无环境结果复算

[WSL Bash]

```bash
cd "$HOME/Kaggle/Lab_Kaggle/lux-ai-s3"
export LUX_PYTHON="$HOME/ml-workspace/.venv/bin/python"
"$LUX_PYTHON" -B tools/recompute_results.py
"$LUX_PYTHON" -B -m pytest -q -p no:cacheprovider tests
```

结果工具独立重算保留记录的点估计与计数，区间读取原冻结输出而不是重新选择随机种子制造一致。完整逐局原日志仍在外部，source_index.json记录不可变SHA与相对来源。

## B. 官方环境和对手依赖

当前工作站：

[WSL Bash]

```bash
export LUX_ARTIFACTS="$HOME/ml-workspace/artifacts/lux-s3-expert-preparation-20261111"
export LUX_LEGACY_ARCHIVE="$HOME/ml-workspace/archives/lux-s3-retired-20261112_C01/legacy_project"
export LUX_ENGINE_PYTHON="$LUX_LEGACY_ARCHIVE/.venv/bin/python"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MLFLOW_DISABLE_TELEMETRY=true MLFLOW_ENABLE_ASYNC_LOGGING=false
export LUX_OPPONENT_REGISTRY="$LUX_ARTIFACTS/local_reproduction/relocated_opponents.json"
"$LUX_PYTHON" -B tools/prepare_local_assets.py \
  --source "$LUX_ARTIFACTS/admitted_opponents.json" \
  --output "$LUX_OPPONENT_REGISTRY"
"$LUX_PYTHON" -B tools/preflight.py
```

官方环境代码已经回到vendor，归档中的Python环境只是当前机器已有解释器；进程强制从本项目vendor导入Lux环境。上述命令不调用reset/step。

源工程从独立仓库移到Lab_Kaggle后，旧外部Agent注册表的wrapper路径失效。prepare_local_assets仅生成派生登记：替换代码根路径、验证所有原SHA值不变；它不修改原注册表、不放行变更权重、不启动对战。

在新机器上，需分别安装requirements.txt、requirements-engine.txt、requirements-opponents.txt；第三方CPU环境和权重按reports/teacher_catalog.json恢复，并完成原有准入，不能凭旧本机路径宣称已验收。

## C. 数据构建方法（仅手动）

当前已处理数据可直接核验，无需重建：

[WSL Bash]

```bash
"$LUX_PYTHON" -B -m training.expert.macro_dataset \
  --verify --output "$LUX_ARTIFACTS/macro_dataset01"
```

完整构建入口均保留：

[WSL Bash]

```bash
rebuild="$LUX_ARTIFACTS/manual_rebuild_$(date +%Y%m%d_%H%M%S)"
mkdir "$rebuild"
"$LUX_PYTHON" -B -m training.expert.replays --audit \
  --replay-root "$LUX_LEGACY_ARCHIVE/data/raw_replays" --output "$rebuild/raw"
"$LUX_PYTHON" -B tools/index_expert_versions.py \
  --manifest "$rebuild/raw/manifest.json" \
  --metadata "$LUX_LEGACY_ARCHIVE/data/kaggle_downloads/86411" --output "$rebuild/versions.json"
"$LUX_PYTHON" -B -m training.expert.macro_dataset \
  --source "$LUX_LEGACY_ARCHIVE/outputs/mappo_v5_2/v2_goal_adapter_mode_gate_run576_20261029_G06" \
  --output "$rebuild/macro"
"$LUX_PYTHON" -B -m training.expert.primitive_dataset \
  --manifest "$rebuild/versions.json" --teacher 'Flat Neurons' \
  --submission-id 43368177 --output "$rebuild/primitive"
```

## D. BC、RL、恢复的实际入口

以下示例只说明已实现的方法，不自动执行。所有运行必须用新目录，不能覆盖历史pretrain_r02或formal。

[WSL Bash]

```bash
"$LUX_PYTHON" -B -m training.pretrain --model a --seed 3101 \
  --dataset "$LUX_ARTIFACTS/macro_dataset01" \
  --teacher G06_FULL_S2_B0_RMode_V2_adapted_Goal \
  --primitive-dataset "$LUX_ARTIFACTS/primitive_flat02" \
  --run-dir "$LUX_ARTIFACTS/manual_reproduction/bc_a"
"$LUX_PYTHON" -B -m training.bc_check --model a \
  --checkpoint "$LUX_ARTIFACTS/manual_reproduction/bc_a/selected.json"
"$LUX_PYTHON" -B -m training.train --model a --seed 1313969260 \
  --bc-checkpoint "$LUX_ARTIFACTS/manual_reproduction/bc_a/selected.json" \
  --opponent-registry "$LUX_OPPONENT_REGISTRY" \
  --run-dir "$LUX_ARTIFACTS/manual_reproduction/rl_a"
```

BC只训练对应Actor和训练用辅助头；进入RL加载Actor，Critic按原规则初始化，RL优化器为空。BC按Calibration复合NLL选模，之后才读取Exploration。

BC恢复在同一pretrain命令上加--resume；新RL恢复在同一train命令上加--resume latest。完整checkpoint包含AdamW、RNG、训练步数和PFSP状态。半局/半epoch不伪装成已提交进度；失败attempt保留。不能更改源码/数据/注册表身份后强行续接。

历史1200局A结果已冻结。若复现其最终开发评估，可使用下节入口；这项评估尚未执行，不能在报告里补造结果。

## E. 官方配对评估与监控

[WSL Bash]

```bash
"$LUX_PYTHON" -B -m training.evaluate \
  --checkpoint "$LUX_ARTIFACTS/formal/a-1313969260/batch_000299/checkpoint.pt" \
  --opponent v2 --output "$LUX_ARTIFACTS/manual_reproduction/eval_a_v2"
"$LUX_PYTHON" -B -m training.evaluate --summarize \
  --output "$LUX_ARTIFACTS/manual_reproduction/eval_a_v2"
```

评估24地图、双方阵营，共48局；官方team_wins独立重算，地图配对Bootstrap。只在人工另行决定运行后才有结果。

[WSL Bash]

```bash
"$LUX_PYTHON" -B -m training.inspect_run "$LUX_ARTIFACTS/formal/a-1313969260"
"$LUX_PYTHON" -m mlflow ui --backend-store-uri "sqlite:///$LUX_ARTIFACTS/mlflow.db" \
  --host 127.0.0.1 --port 5000
```

MLflow数据库、checkpoint、NPZ、完整JSON和外部Agent都不进入Git。当前代码来源锁保留原70个核心文件的哈希，因此路径整理不等于修改冻结算法。新文档、工具、证据和baseline快照不冒充原始实验的一部分。
