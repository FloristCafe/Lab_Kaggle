# 2026-08-18 OTTO Stage 2 Retrieval Closeout

## Goal

阶段二收尾：用 Polars lazy 频率统计、Hash Join degree penalty、矩阵式 Top-K 截断和分块候选池生成，闭环三路 heuristic co-visitation recall。

## Engineering Changes

核心文件：

- `src/otto_recommender/heuristic_covisitation.py`
- `scripts/run_heuristic_retrieval_bucketed.py`
- `scripts/run_heuristic_retrieval.py`
- `docs/stage2_heuristic_covisitation.md`

关键改动：

- `item_frequency()` 改成只读取 `aid` 列做 lazy frequency aggregation。
- 共现 pair 生成后，再对 `aid_x` 和 `aid_y` 做两次 Hash Join 接入全局频率。
- 最终边权：

```text
w_final = c / (t + c) * 1 / (N(A) * N(B)) ^ alpha
```

- 默认 `alpha = 0.5`，即根号 degree penalty。
- 不使用 `apply` 或逐行 Python 函数。
- 新增分块候选生成脚本，避免一次性全量 candidate join 卡住内存。

## Final Commands

```powershell
conda activate kg_env
cd D:\Python\Kaggle\otto-recommender-system
python scripts\build_heuristic_covisitation.py --n-buckets 16 --degree-alpha 0.5
python scripts\run_heuristic_retrieval_bucketed.py --n-buckets 16 --candidate-topk 100 --min-candidates 50 --eval-topk 20
python -m pytest -q
```

## Outputs

三路图：

- `artifacts/candidates/heuristic_covisitation/click_to_click_top20.parquet`
- `artifacts/candidates/heuristic_covisitation/cart_order_to_cart_order_top20.parquet`
- `artifacts/candidates/heuristic_covisitation/click_to_cart_order_top20.parquet`

候选池：

- `artifacts/candidates/heuristic_candidates_top100_parts/`
- part files: `16`
- total size: about `621 MB`

统计：

- `artifacts/reports/heuristic_candidates_top100_stats.csv`
- `artifacts/reports/heuristic_candidates_top100_stats_summary.csv`
- `artifacts/reports/heuristic_candidates_top100_metrics.csv`

## Candidate Pool Summary

| metric | value |
| --- | ---: |
| sessions | 1,671,803 |
| min candidates per session | 50 |
| mean candidates per session | 94.7435 |
| median candidates per session | 100 |
| max candidates per session | 100 |
| sessions with >= 50 candidates | 1,671,803 |
| sessions with exactly 100 candidates | 1,401,234 |

## Recall@20

| type | total | hits | recall | metric weight |
| --- | ---: | ---: | ---: | ---: |
| clicks | 822,342 | 292,095 | 0.35519892 | 0.1 |
| carts | 83,911 | 74,127 | 0.88340026 | 0.3 |
| orders | 15,451 | 13,834 | 0.8953466 | 0.6 |

Weighted Recall@20:

```text
0.83774793
```

## Interpretation

阶段二已经闭环：

- 三路召回图已经独立构建。
- 图边表使用 COO schema：`aid_x, aid_y, weight, graph`。
- 边权同时融合时间衰减与 degree penalty。
- 最终候选池达到每个 session 50-100 个 item。
- 评估仍保持高 carts/orders recall，适合进入后续 ranker 阶段。

下一阶段应转向候选特征表与排序模型：

- candidate-source features
- graph weights from each route
- item popularity / recency features
- session recency features
- LightGBM / CatBoost ranker baseline
