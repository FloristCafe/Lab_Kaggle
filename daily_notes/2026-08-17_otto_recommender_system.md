# 2026-08-17 OTTO Recommender System Daily Note

## Project

- 项目目录：`D:\Python\Kaggle\otto-recommender-system`
- Python 环境：`C:\Users\lenovo\miniconda3\envs\kg_env\python.exe`
- 本地数据：`data/test.parquet`
- 数据 schema：`session:int32, aid:int32, ts:int64, type:int8`
- 当前目标：先在本地沙盒验证可扩展的图召回流水线，再迁移到 Kaggle Notebook 上做更大规模测试。

## What We Built

今天把项目从 toy baseline 升级为 Polars-first 的图召回沙盒。

核心新增模块：

- `src/otto_recommender/polars_graph.py`
  - 标准化事件列类型。
  - 将 session 序列转成 COO 风格共现边表。
  - 边表 schema：`aid_x:int32, aid_y:int32, weight:float32`。
  - 支持按 session hash 分块、块内 Early Pruning、合并后二次聚合与最终 Top-K 截断。
  - 支持用边表生成 session-level Top-20 recommendation。
  - 支持本地 weighted Recall@K 评估。

核心新增脚本：

- `scripts/split_local.py`
  - 从 `data/test.parquet` 切出本地训练事件与验证标签。
- `scripts/build_covisitation_edges.py`
  - 构建分块共现边表并合并为紧凑 Top-20 边表。
- `scripts/run_graph_baseline.py`
  - 用共现边表做 graph retrieval baseline，并输出推荐结果与指标。

核心文档：

- `docs/local_graph_sandbox.md`
  - 记录本地沙盒命令、数据契约、内存策略、Kaggle 迁移注意事项。
- `configs/local_graph_baseline.yaml`
  - 记录当前本地图召回配置。

## Commands Run

```powershell
conda activate kg_env
cd D:\Python\Kaggle\otto-recommender-system
python -m pip install -e .
python scripts\split_local.py
python scripts\build_covisitation_edges.py --n-buckets 16 --topk-per-chunk 80 --final-topk-per-source 20
python scripts\run_graph_baseline.py
python -m pytest -q
```

## Artifacts

本地切分结果：

- `data/processed/local_train.parquet`
  - `6,006,419` training events
- `data/processed/local_valid_labels.parquet`
  - `921,704` validation label rows

图召回结果：

- `artifacts/candidates/edge_parts/`
  - 16 个分块边表。
- `artifacts/candidates/top20_edges.parquet`
  - 最终紧凑共现边表，约 `43 MB`。
- `artifacts/candidates/local_recommendations.parquet`
  - 本地 Top-20 推荐结果，约 `201 MB`。
- `artifacts/reports/edge_build_stats.csv`
  - 分块建边统计。
- `artifacts/reports/local_graph_metrics.csv`
  - 本地 Recall@20 指标。

## Local Metrics

Weighted Recall@20:

- `0.7504884`

Per-type Recall@20:

| type | meaning | total | hits | recall | metric_weight |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | clicks | 822,342 | 312,402 | 0.37989303 | 0.1 |
| 1 | carts | 83,911 | 68,526 | 0.816651 | 0.3 |
| 2 | orders | 15,451 | 12,039 | 0.7791729 | 0.6 |

## Current Modeling Status

当前项目不是 MDP 建模。

虽然数据天然是 session 序列，表面上可以联想到状态、动作、奖励和转移，但现在的实现是推荐系统里的图召回 baseline：

- state 没有被显式定义成马尔可夫状态。
- action 没有被建模为 agent 的策略输出。
- reward 没有被设计成 RL 奖励函数。
- transition dynamics 没有被学习或估计。
- 没有 policy evaluation / policy improvement。

当前也没有使用 Deep Learning agent。

目前使用的是：

- Polars DataFrame
- COO-style item-item co-visitation graph
- recency seed
- graph expansion
- weighted edge aggregation
- Top-K retrieval

这是一条高性能传统推荐召回链路，不是 DL，也不是 RL agent。

## Why This Structure Is Useful

这个结构适合后续迁移到 Kaggle 大数据：

- 避免巨大的 Python nested dict。
- 用列式边表管理 item-item graph。
- 每个 chunk 先截断高价值边，减少中间结果膨胀。
- 合并 chunk 后再次聚合和截断，得到紧凑的召回资产。
- `top20_edges.parquet` 可以作为后续 ranker、特征工程或 notebook baseline 的输入。

## Next Steps

1. 把 `top20_edges.parquet` 和 graph baseline 逻辑迁移到 Kaggle Notebook。
2. 增加 type-specific co-visitation edges，例如 clicks-only、carts/orders-weighted 两套图。
3. 加入更强的 session recency weighting 与 item popularity fallback。
4. 构建 candidate feature table，为后续 LightGBM / CatBoost ranker 做准备。
5. 如果要往量化金融/RL 方向迁移，再设计真正的 sequential decision formulation：
   - state: session history embedding
   - action: recommended item slate
   - reward: click/cart/order weighted feedback
   - policy: slate recommendation policy
   - evaluation: off-policy 或 logged bandit evaluation

## Stage 2 Update: Heuristic Co-visitation

新增阶段二多规则共现图：

- `click_to_click`
  - source: click
  - target: click
  - window: 24h
  - decay: exponential half-life, half-life = 6h
- `cart_order_to_cart_order`
  - source: cart/order
  - target: cart/order
  - clicks are strictly filtered out
  - window: 14 days
  - decay: soft long-tail decay
- `click_to_cart_order`
  - source: click
  - target: cart/order
  - window: 7 days
  - decay: conversion half-life, half-life = 2 days

新增文件：

- `src/otto_recommender/heuristic_covisitation.py`
- `scripts/build_heuristic_covisitation.py`
- `scripts/run_heuristic_retrieval.py`
- `docs/stage2_heuristic_covisitation.md`
- `tests/test_heuristic_covisitation.py`

Smoke 版运行结果：

- command: `python scripts\build_heuristic_covisitation.py --n-buckets 2`
- command: `python scripts\run_heuristic_retrieval.py --graphs-dir artifacts\candidates\heuristic_covisitation_smoke --recommendations-output artifacts\candidates\heuristic_recommendations_smoke.parquet --metrics-output artifacts\reports\heuristic_graph_metrics_smoke.csv`
- recommendations: `28,447,120`
- weighted Recall@20: `0.761238`

Click-to-Click 当前时间衰减与流行度惩罚公式：

```text
w(A -> B) = 21600 / (t + 21600) * 1 / (N(A) * N(B)) ^ 0.5
```

其中 `t` 是两次点击相隔秒数，只保留 `0 < t <= 86400` 的点击对。

2026-08-17 晚间公式修正：

- 移除指数衰减中的 `exp`，改用 Polars/Rust 向量化除法 `c / (t + c)`。
- 引入 degree penalty，也就是 `1 / (N(A) * N(B)) ^ alpha`。
- 默认 `alpha = 0.5`，等价于根号惩罚。
- 脚本参数：`python scripts\build_heuristic_covisitation.py --n-buckets 16 --degree-alpha 0.5`
- 召回收尾：最终候选池默认改为每个 session `Top-100`，评估仍看 `Recall@20`。
- 召回输出：`artifacts/candidates/heuristic_candidates_top100.parquet`

Degree penalty 全量运行结果：

| model | clicks recall | carts recall | orders recall | weighted Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| single graph baseline | 0.37989303 | 0.816651 | 0.7791729 | 0.7504884 |
| stage2 heuristic before degree penalty | 0.38281518 | 0.8372323 | 0.7862274 | 0.7611877 |
| stage2 heuristic with degree penalty | 0.36344126 | 0.88442516 | 0.89528185 | 0.83884084 |

观察：

- Click recall 下降，但 carts/orders recall 大幅提升。
- OTTO 评估权重更偏向 orders 和 carts，因此总分显著提升。
- Degree penalty 有效压制了爆款 hub item 的泛化噪声，让高意图转化图更加锋利。
