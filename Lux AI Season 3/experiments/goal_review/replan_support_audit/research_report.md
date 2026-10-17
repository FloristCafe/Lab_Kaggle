# R标签按任务模态的支持域审计

## 1. 实验假设与核心矛盾

检查Continue是否覆盖五类宏任务，避免总体R正类F1掩盖某些任务从无继续样本的问题。

## 2. 数据与防穿越边界

完整725场，580/145隔离；只统计冻结NPZ的scripted causal memory和review时刻。未来意图区域只用于既有R监督，不进入输入。污染memory后的支持域会变化，不能把本表直接套到corrupted或policy回流条件。

## 3. 特征构建与数学表达

free=review & replan_valid & alive & previous_valid；按专家任务mode统计R0/R1。Previous无效的强制R1单独计数，不计为可学习的二分类成功。

## 4. 离线评估与指标对比

| Split | Expert mode | Free Continue | Free Replan | Forced Replan |
|---|---|---:|---:|---:|
| train | SCORE | 27,578 | 43,570 | 3,318 |
| train | EXPLORE | 11 | 3,784 | 639 |
| train | DEFEND | 0 | 13,667 | 155 |
| train | INTERCEPT | 0 | 288,485 | 18,996 |
| train | RECHARGE | 1,705 | 24,006 | 82 |
| validation | SCORE | 7,098 | 9,421 | 749 |
| validation | EXPLORE | 2 | 961 | 119 |
| validation | DEFEND | 0 | 3,031 | 28 |
| validation | INTERCEPT | 0 | 72,737 | 4,831 |
| validation | RECHARGE | 412 | 5,878 | 0 |

Previous-mode输入分布见summary.json。原规则memory只发出SCORE/EXPLORE/RECHARGE，因此在该输入分布下，DEFEND/INTERCEPT无法获得“previous同任务且目标兼容”的R0标签。共享网络可能泛化规则，但不存在这些任务的直接Continue监督；不能由总体Continue recall推断已学会长期防守/追击保持。
需要结合冻结策略回流和真实短闭环衡量这项支持域缺口。任务标签本身仍是启发式，本统计不提供人工战略意图真值，也不自动修改样本或新增训练。

## 5. 性能瓶颈与物理开销

逐NPZ读取既有小数组，无模型forward、优化或比赛，训练更新数0。数据/源码哈希与MLflow run ID归档；CPU峰值/能耗未单独采集。
