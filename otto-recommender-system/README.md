# OTTO Recommender System

本项目用于配合 Kaggle Notebook 做 OTTO Recommender System 的大型实验，本地主要承担微缩沙盒、特征验证、候选召回逻辑调试和可复现实验记录。

## Recommended Workflow

1. 在 `data/sample/` 放很小的本地样本，用来快速验证 session parsing、候选生成、特征 join、ranking pipeline。
2. 在 `sandbox/micro_runs/` 做一次性探索，确认有效后再整理进 `src/otto_recommender/` 或 `notebooks/local/`。
3. 在 `notebooks/kaggle/` 存放准备上传到 Kaggle 的 notebook 版本，尽量只保留 Kaggle 环境可运行的代码。
4. 在 `artifacts/` 下保存本地生成的特征、候选、模型和提交文件；默认大文件不进入 Git。
5. 在 `docs/` 记录实验结论、召回策略、特征清单和 leaderboard 变化。

## Directory Map

```text
otto-recommender-system/
  configs/              # 本地和 Kaggle 的参数配置
  data/
    raw/                # 原始 Kaggle 数据，建议不提交
    sample/             # 微缩样本，可提交少量小文件
    interim/            # 清洗后的中间数据，建议不提交
    processed/          # 可直接训练或召回的数据，建议不提交
    external/           # 外部补充数据，建议不提交
  notebooks/
    kaggle/             # Kaggle Notebook 上传版
    local/              # 本地分析 notebook
  sandbox/
    micro_runs/         # 快速试错代码
  src/otto_recommender/ # 可复用 Python 模块
  scripts/              # 数据处理、训练、提交生成脚本
  artifacts/
    features/           # 特征缓存
    candidates/         # 候选召回缓存
    models/             # 模型文件
    submissions/        # 提交文件
    reports/            # 图表、指标、实验报告
  kaggle_working/       # 模拟 Kaggle /kaggle/working
  logs/                 # 本地运行日志
  tests/                # 小样本单元测试
```

## Space Policy

- `data/raw/` 放完整 OTTO 数据集或 Kaggle 下载包。
- `data/sample/` 只放微缩样本，目标是单次运行几秒到几分钟内完成。
- `artifacts/features/` 和 `artifacts/candidates/` 会快速变大，按实验编号或日期分子目录管理。
- `artifacts/submissions/` 保留值得对比的提交文件，临时提交可以放到 `kaggle_working/`。
- 大文件默认被 `.gitignore` 忽略，重要实验结论写进 `docs/`，不要只依赖缓存文件。

## Suggested Local Milestones

1. 构造 `data/sample/` 微缩数据，覆盖 clicks、carts、orders 三类目标。
2. 实现 covisitation / item-item / recency based candidates。
3. 在小样本上验证 Recall@20 的计算逻辑。
4. 把候选生成、特征构造、rerank 拆成可复用脚本。
5. 将稳定版本迁移到 `notebooks/kaggle/`，再用 Kaggle GPU/CPU 资源跑完整数据。

## First Code Modules

- `src/otto_recommender/io.py`: 读取 Kaggle 原始 `train.jsonl` / `test.jsonl`，并展开为事件表。
- `src/otto_recommender/sampling.py`: 从完整 JSONL 中抽取完整 session，生成本地微缩数据。
- `src/otto_recommender/candidates.py`: recent item、popular item、co-visitation 的 baseline 候选召回。
- `src/otto_recommender/metrics.py`: 本地验证用的 weighted Recall@K。
- `src/otto_recommender/submission.py`: 生成 Kaggle 需要的 `session_type,labels` 提交格式。
- `src/otto_recommender/toy_data.py`: 没有真实数据时用于 smoke test 的小样本。
- `scripts/run_baseline.py`: 从事件文件直接生成一个 baseline submission。
- `notebooks/kaggle/otto_baseline_template.py`: Kaggle Notebook 可复制/改造的起步模板。
- `docs/function_reference.md`: 当前函数用途和常用命令速查。
- `src/otto_recommender/polars_graph.py`: Polars 边表/COO 图召回、分块构建、早截断、Recall 评估。
- `scripts/split_local.py`: 把本地 parquet 切成训练事件和验证标签。
- `scripts/build_covisitation_edges.py`: 构建并合并高价值 Top-K 共现边。
- `scripts/run_graph_baseline.py`: 用紧凑共现边生成本地 Top-20 召回并评估。
- `docs/local_graph_sandbox.md`: 本地图召回沙盒说明和 PyCharm 命令。
- `src/otto_recommender/heuristic_covisitation.py`: 阶段二多规则共现图，包含 Click-to-Click、Cart/Order-to-Cart/Order、Click-to-Cart/Order。
- `scripts/build_heuristic_covisitation.py`: 分块构建三路 Top-20 heuristic co-visitation 图。
- `scripts/run_heuristic_retrieval.py`: 聚合三路图生成最终 Top-20 候选集并评估。
- `scripts/run_heuristic_retrieval_bucketed.py`: 分块生成每个 session 的 Top-100 候选池，更适合 Kaggle 大数据。

## Commands

```powershell
cd D:\Python\Kaggle\otto-recommender-system
python -m pip install -e .[dev]
python sandbox\micro_runs\demo_local_pipeline.py
pytest
```

真实数据在 Kaggle notebook 上时，可以先把同名模块复制进 notebook，或在 notebook 中用 `sys.path.append('/kaggle/input/your-code/src')` 这类方式导入。等本地有原始 JSONL 后，可用：

```powershell
python scripts\make_sample.py --input data\raw\train.jsonl --output data\sample\train_sample.jsonl --n-sessions 10000
```
