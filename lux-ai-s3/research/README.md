# 精选研究模块

这里补回两条适合科研汇报、可以小规模离线核验的主线。原始环境、执行器、BC/PPO框架仍在正式目录；这里不重新引入历史Launcher或失败修复脚本。

| 板块 | 核心问题 | 入口 |
|---|---|---|
| 一：表示与优化诊断 | 学不到教师Goal，是输入不足、函数类限制，还是优化未收敛？ | [representation_diagnostics](representation_diagnostics/README.md) |
| 二：同状态Goal信用与可复现性 | 改变一次合法Goal，是否出现可重复的官方收益差？ | [paired_goal_credit](paired_goal_credit/README.md) |

先看[导师汇报提纲](../docs/mentor_research_brief.md)。历史JSON原样保存在reports/evidence并登记SHA256；抽取代码的原文件哈希、函数范围和改动边界在[source_manifest.json](source_manifest.json)。

## 无环境复核

在项目根目录，使用已有WSL Python环境：

```bash
python -B -m research.representation_diagnostics.report
python -B -m research.paired_goal_credit.report
python -B -m pytest -q -p no:cacheprovider tests/test_research_representation.py tests/test_research_goal_credit.py
```

生成图需要额外安装requirements-research.txt。现有图和表已提交，无需安装绘图库即可看文档。以下目录须尚不存在，避免覆盖：

```bash
python -B -m research.representation_diagnostics.report --output-dir /tmp/lux-representation-review
python -B -m research.paired_goal_credit.report --output-dir /tmp/lux-credit-review
```

上述命令只复算已有证据；不会训练或创建环境。原始大数组、完整轨迹、模型和MLflow仍在外部资产/归档中，不能把这些轻量模块说成全量重训的一键复现。
