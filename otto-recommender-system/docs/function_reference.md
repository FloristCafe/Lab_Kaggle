# Function Reference

## Data IO

- `flatten_sessions(records)`: 把 OTTO 原始 JSONL 的 session records 展开成事件表，列为 `session, aid, ts, type`。
- `iter_otto_jsonl(path, chunk_size=10000)`: 分块读取大型 `train.jsonl` / `test.jsonl`，适合以后做内存友好的特征统计。
- `read_otto_jsonl(path, n_sessions=None)`: 读取少量或中等规模 JSONL 到一个 DataFrame，本地微缩实验最方便。
- `read_events(path, **kwargs)`: 根据后缀读取 `.jsonl`、`.json`、`.parquet`、`.csv`。
- `write_events(df, path)`: 根据后缀写出 `.parquet` 或 `.csv`。

## Sampling

- `sample_jsonl_sessions(input_path, output_path, n_sessions, seed=42)`: 从完整 OTTO JSONL 中随机抽完整 session，生成本地小样本。

## Candidate Generation

- `popular_items(events, topk=100)`: 按事件类型加权统计全局热门商品。
- `recent_items_by_session(events, topk=20)`: 为每个 session 返回最近交互过的去重商品。
- `co_visitation_map(events, max_events_per_session=30, topk=40)`: 从 session 内共现构造 item-to-item 召回表。
- `candidates_from_recent_and_covisitation(events, session_ids=None, final_topk=20, covisitation_topk=40)`: 组合 recent items、co-visitation 扩展和 popular fallback，输出 `{session: [aid...]}`。

## Metrics And Submission

- `normalize_labels(value)`: 把 Kaggle 的空格分隔标签或 Python list 统一成整数 list。
- `recall_at_k(labels, predictions, k=20, weights=None)`: 计算 OTTO 风格加权 Recall@K，默认权重为 clicks 0.10、carts 0.30、orders 0.60。
- `candidates_to_submission(candidates, event_types=("clicks", "carts", "orders"))`: 输出 Kaggle 提交格式 `session_type, labels`。
- `predictions_for_metric(candidates, event_types=("clicks", "carts", "orders"))`: 输出本地验证需要的 `session, type, labels`。

## No-data Smoke Test

- `make_toy_events()`: 生成一个很小的 OTTO-like 事件表。
- `make_toy_labels()`: 生成 toy labels，用来验证 Recall@20 计算链路。

## Useful Commands

```powershell
cd D:\Python\Kaggle\otto-recommender-system
.venv\Scripts\python.exe sandbox\micro_runs\demo_local_pipeline.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\make_sample.py --input data\raw\train.jsonl --output data\sample\train_sample.jsonl --n-sessions 10000
.venv\Scripts\python.exe scripts\run_baseline.py --events data\sample\train_sample.jsonl --output artifacts\submissions\baseline_submission.csv
```
