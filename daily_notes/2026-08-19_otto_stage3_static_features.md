# 2026-08-19 OTTO Stage 3 Static Features

## Goal

第三阶段开始做静态特征双塔：

- `item_features.parquet`
- `user_features.parquet`

工程约束：

- Kaggle `/kaggle/working` 只有约 20GB，不生成超级宽表。
- 每张特征表独立存储，后续训练时按 batch 动态 join。
- 使用 Polars Lazy API：`pl.scan_parquet()`。
- 写出使用 `LazyFrame.sink_parquet()`。
- 写完显式 `del` 变量并 `gc.collect()`。

## Implemented Files

- `src/otto_recommender/feature_engineering.py`
- `scripts/build_item_features.py`
- `scripts/build_user_features.py`
- `tests/test_feature_engineering.py`

## Item Features

输出：

- `artifacts/features/item_features.parquet`

主键：

- `aid`

字段：

- `total_interactions`
- `click_count`
- `cart_count`
- `order_count`
- `recent_24h_interactions`
- `conversion_rate`

核心公式：

```text
conversion_rate = (cart_count + order_count) / (click_count + 1)
```

## User / Session Features

输出：

- `artifacts/features/user_features.parquet`

主键：

- `session`

字段：

- `session_length`
- `unique_items`
- `first_ts`
- `last_ts`
- `duration`
- `cart_count`
- `order_count`
- `is_window_shopping`

核心公式：

```text
duration = t_last - t_first
is_window_shopping = session_length >= 50 and cart_count == 0 and order_count == 0
```

## Local Smoke Result

本地输入：

- `data/test.parquet`

输出规模：

| table | rows | cols | size |
| --- | ---: | ---: | ---: |
| item_features | 783,486 | 7 | 4.34 MB |
| user_features | 1,671,803 | 9 | 21.44 MB |

测试：

```text
13 passed
```

## Commands

本地 smoke：

```powershell
conda activate kg_env
cd D:\Python\Kaggle\otto-recommender-system
python scripts\build_item_features.py --input data\test.parquet --output artifacts\features\item_features.parquet
python scripts\build_user_features.py --input data\test.parquet --output artifacts\features\user_features.parquet
python -m pytest -q
```

Kaggle full train:

```powershell
python scripts\build_item_features.py --input /kaggle/input/otto-full-optimized-memory-footprint/train.parquet --output /kaggle/working/item_features.parquet
python scripts\build_user_features.py --input /kaggle/input/otto-full-optimized-memory-footprint/train.parquet --output /kaggle/working/user_features.parquet
```

