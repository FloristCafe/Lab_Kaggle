# Local Graph Sandbox

这个沙盒用于把本地 `data/test.parquet` 切成训练事件与验证标签，然后用 Polars DataFrame 构建紧凑的共现图召回。

## Data Contract

输入事件表：

| column | dtype | meaning |
| --- | --- | --- |
| `session` | `int32` | session id |
| `aid` | `int32` | item id |
| `ts` | `int64` | event timestamp |
| `type` | `int8` | event type id, usually `0=clicks, 1=carts, 2=orders` |

核心图结构是 COO 风格边表：

| column | dtype | meaning |
| --- | --- | --- |
| `aid_x` | `int32` | 源节点，用户先交互的商品 |
| `aid_y` | `int32` | 目标节点，用户后交互的商品 |
| `weight` | `float32` | 共现权重 |

## Pipeline Stages

1. `split_local.py`
   - 输入：`data/test.parquet`
   - 输出：`data/processed/local_train.parquet`
   - 输出：`data/processed/local_valid_labels.parquet`
   - 逻辑：每个 session 留最后 `N` 个事件做验证，其余作为训练事件。

2. `build_covisitation_edges.py`
   - 输入：`data/processed/local_train.parquet`
   - 中间输出：`artifacts/candidates/edge_parts/edges_part_*.parquet`
   - 最终输出：`artifacts/candidates/top20_edges.parquet`
   - 逻辑：按 `session % n_buckets` 分块，块内构建边表，先截断每个 `aid_x` 的高价值边，再合并所有块，重新聚合权重并再次截断。

3. `run_graph_baseline.py`
   - 输入：训练事件与 `top20_edges.parquet`
   - 输出：`artifacts/candidates/local_recommendations.parquet`
   - 输出：`artifacts/reports/local_graph_metrics.csv`
   - 逻辑：用最近交互商品做 seed，经共现边扩展，再按 session 聚合分数取 Top-20。

## PyCharm Terminal Commands

PowerShell:

```powershell
conda activate kg_env
cd D:\Python\Kaggle\otto-recommender-system
$env:PYTHONPATH="D:\Python\Kaggle\otto-recommender-system\src"
python -m pip install -e .
python scripts\split_local.py
python scripts\build_covisitation_edges.py --n-buckets 16 --topk-per-chunk 80 --final-topk-per-source 20
python scripts\run_graph_baseline.py
python scripts\run_heuristic_retrieval.py --candidate-topk 100 --eval-topk 20
python -m pytest -q
```

cmd:

```cmd
conda activate kg_env
cd /d D:\Python\Kaggle\otto-recommender-system
set PYTHONPATH=D:\Python\Kaggle\otto-recommender-system\src
python -m pip install -e .
python scripts\split_local.py
python scripts\build_covisitation_edges.py --n-buckets 16 --topk-per-chunk 80 --final-topk-per-source 20
python scripts\run_graph_baseline.py
python scripts\run_heuristic_retrieval.py --candidate-topk 100 --eval-topk 20
python -m pytest -q
```

## Memory Strategy

- 使用 Polars DataFrame 表达图，不把完整 item-item 图长期放进 Python 嵌套字典。
- 分块键是 `session % n_buckets`，可把大数据拆成多个 session 子集独立建边。
- 每个块先按 `aid_x` 保留 `topk_per_chunk` 条边，这是 Early Pruning。
- 合并所有块后再次 `group_by(aid_x, aid_y).sum(weight)`，然后保留 `final_topk_per_source`。
- Kaggle 大数据上可以增加 `n_buckets`，例如 `32/64/128`，用更小块换取更稳的内存。

## Kaggle Migration Notes

- notebook 中保持同样的数据契约：`session, aid, ts, type`。
- 如果 Kaggle 里内存紧张，先把 `n_buckets` 提高，再降低 `max_events_per_session` 或 `topk_per_chunk`。
- `top20_edges.parquet` 是可以复用的召回资产，后续 reranker 可以直接把它当图特征源。
