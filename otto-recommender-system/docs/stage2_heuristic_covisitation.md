# Stage 2: Heuristic Co-visitation

阶段二目标是构建三路工业级启发式共现图，并用列式边表管理召回资产。最终输出不只是 Top-20 评估结果，而是每个 session 的 50-100 个候选商品池，供后续 ranker 使用。

## Graph Outputs

默认输出目录：

```text
artifacts/candidates/heuristic_covisitation/
```

三路最终 Top-20 边表：

- `click_to_click_top20.parquet`
- `cart_order_to_cart_order_top20.parquet`
- `click_to_cart_order_top20.parquet`

边表统一 schema：

| column | dtype | meaning |
| --- | --- | --- |
| `aid_x` | `int32` | source item |
| `aid_y` | `int32` | target item |
| `weight` | `float32` | heuristic co-visitation weight |
| `graph` | `str` | graph rule name |

## Rule 1: Click-to-Click

业务逻辑：用户在同一次探索中点击的商品，常常是相似品或替代品。

过滤：

- source: `type == 0`
- target: `type == 0`
- direction: `ts_y > ts_x`
- time window: `0 < t <= 24h`

时间衰减与流行度惩罚：

```text
w(A -> B) = c / (t + c) * 1 / (N(A) * N(B)) ^ alpha
```

当前实现：

```text
c = 21,600 seconds
alpha = 0.5
t = (ts_y - ts_x) / 1000
N(A), N(B) = item global interaction frequency
```

直觉：

```text
time_decay = c / (t + c)
degree_penalty = 1 / sqrt(N(A) * N(B))
```

这样满足：

- 24 小时内有效，越近越强。
- 不使用逐行 Python `apply`。
- 对爆款 hub item 做惩罚，让冷门但置信度更高的共现边更容易浮上来。

## Rule 2: Cart/Order-to-Cart/Order

业务逻辑：加购和购买代表更强商业意图，适合挖掘互补品和高转化共现。

过滤：

- source: `type in {1, 2}`
- target: `type in {1, 2}`
- strict drop clicks: `type == 0` 不进入这路图
- direction: `ts_y > ts_x`
- time window: `0 < t <= 14 days`

时间衰减：

```text
w(A -> B) = target_value(type_y) * c / (t + c) * 1 / (N(A) * N(B)) ^ alpha * graph_weight
```

当前取值：

- target cart value: `3.0`
- target order value: `6.0`
- c: `7 days`
- alpha: `0.5`
- graph_weight: `1.5`

这路不使用剧烈衰减，因为补充购买可能跨越更长时间。

## Rule 3: Click-to-Cart/Order

业务逻辑：点击浏览过的商品指向最终加购/购买商品，可以挖掘替代品和转化目标。

过滤：

- source: `type == 0`
- target: `type in {1, 2}`
- direction: `ts_y > ts_x`
- time window: `0 < t <= 7 days`

时间衰减：

```text
w(A -> C) = target_value(type_y) * c / (t + c) * 1 / (N(A) * N(C)) ^ alpha * graph_weight
```

当前取值：

- target cart value: `3.0`
- target order value: `6.0`
- c: `2 days`
- alpha: `0.5`
- graph_weight: `2.0`

## Build Commands

PowerShell:

```powershell
conda activate kg_env
cd D:\Python\Kaggle\otto-recommender-system
python -m pip install -e .
python scripts\build_heuristic_covisitation.py --n-buckets 16 --degree-alpha 0.5
python scripts\run_heuristic_retrieval.py --candidate-topk 100 --eval-topk 20
python -m pytest -q
```

内存更稳的候选池生成方式：

```powershell
python scripts\run_heuristic_retrieval_bucketed.py --n-buckets 16 --candidate-topk 100 --min-candidates 50 --eval-topk 20
```

Kaggle 大数据时，如果内存紧张：

```powershell
python scripts\build_heuristic_covisitation.py --n-buckets 32
python scripts\build_heuristic_covisitation.py --n-buckets 64
```

## Smoke Result

使用 `--n-buckets 2` 的 smoke 版已经跑通：

```text
recommendations=28,447,120
weighted_recall@20=0.761238
```

## Stage-2 Closing Criteria

阶段二闭环产物：

- 三路图各自产出 Top-20 边表。
- 三路图合并后为每个 session 生成 `50-100` 个 candidate items。
- 评估仍然使用 Kaggle 风格 weighted Recall@20。
- 后续排序阶段使用 `heuristic_candidates_top100.parquet`，而不是只使用 Top-20 recommendation。
- 如果全量候选池单文件生成较慢，使用 `heuristic_candidates_top100_parts/` 作为正式候选数据集目录。
