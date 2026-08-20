# Stage 3: Interaction Features

本阶段基于候选集构建 candidate-level 交互特征，主键是 `(session, aid)`。

## Inputs

- `data/processed/local_train.parquet`
- `artifacts/candidates/heuristic_candidates_top100_parts/candidates_part_*.parquet`
- `artifacts/candidates/heuristic_covisitation/*_top20.parquet`

## Outputs

正式输出目录：

```text
artifacts/features/interaction_features_parts/
```

Smoke 输出目录：

```text
artifacts/features/interaction_features_parts_smoke/
```

## Feature Columns

候选基础列：

- `session`
- `aid`
- `candidate_score`
- `candidate_rank`

局部交互统计：

- `local_interaction_count`
- `local_click_count`
- `local_cart_count`
- `local_order_count`
- `is_repeated_item`

时间差特征：

- `session_last_ts`
- `item_last_ts`
- `delta_t_nan`
- `delta_t_filled`

图网络信号：

- `graph_weight_sum`
- `graph_weight_max`
- `graph_source_count`
- `graph_w_click_to_click`
- `graph_w_cart_order_to_cart_order`
- `graph_w_click_to_cart_order`
- `graph_weight_total`
- `dominant_graph_source`

## Missing Value Strategy

意图衰减时间差：

```text
delta_t = session_last_ts - item_last_ts
```

对于当前 session 从未交互过的候选商品，`item_last_ts` 缺失，因此 `delta_t` 缺失。

保留两套字段用于后续排序层消融：

- `delta_t_nan`: 保留 null，让 XGBoost/LightGBM 学习缺失值默认分裂方向。
- `delta_t_filled`: 用 `9999999` 填充，表示物理意义上的“距离上次交互极远”。

后续 ranker 需要分别验证：

- A: 使用 `delta_t_nan`
- B: 使用 `delta_t_filled`

比较指标：

- Recall@20
- NDCG@20
- carts/orders weighted metric

## Commands

Smoke:

```powershell
python scripts\build_interaction_features_bucketed.py --bucket-start 0 --bucket-end 1 --output-dir artifacts\features\interaction_features_parts_smoke --stats-output artifacts\reports\interaction_features_stats_smoke.csv
```

Full:

```powershell
python scripts\build_interaction_features_bucketed.py --n-buckets 16 --output-dir artifacts\features\interaction_features_parts --stats-output artifacts\reports\interaction_features_stats.csv
```

Strict candidate chunking:

```powershell
python scripts\build_interaction_features_candidate_chunks.py --n-buckets 16 --candidate-chunks 10 --output-dir artifacts\features\interaction_features_candidate_chunks --stats-output artifacts\reports\interaction_features_candidate_chunks_stats.csv
```

这条路径会把每个 bucket 内的千万级候选再次均分为 `10` 个 chunk。每个 chunk 会：

- 只读取当前 chunk 的 candidate rows。
- 用 `pl.scan_parquet()` 唤醒训练事件。
- 用当前 chunk 的 `(session, aid)` keys 做局部 join。
- 立即 `sink_parquet()` 写入 chunk parquet。
- 显式 `del` 临时变量并 `gc.collect()`。

## Multi-route Weight Experiment

后续 ranker 做两组消融：

- Fused: 只使用 `graph_weight_total` / `graph_weight_sum`。
- Separate: 使用 `graph_w_click_to_click`、`graph_w_cart_order_to_cart_order`、`graph_w_click_to_cart_order` 三路独立权重。

预期：

- Fused 特征更简单，树深度压力更小。
- Separate 特征上限更高，因为树模型可以学习不同业务意图通道的非线性组合。
- 对比指标：Recall@20、NDCG@20、carts/orders weighted metric。

## Smoke Result

Bucket 0:

- rows: `9,906,798`
- file size: about `101 MB`
- repeated rows: `248,579`
- null `delta_t_nan` rows: `9,658,219`
- max graph weight: `117.246422`
