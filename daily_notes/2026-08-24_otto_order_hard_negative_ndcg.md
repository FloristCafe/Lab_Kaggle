# 2026-08-24 OTTO Order Hard Negative NDCG Ablation

## 实验假设与核心矛盾

本轮实验检验 Order 模型的困难负样本配比是否影响真实购买排序质量；核心矛盾是 Order 正样本在候选集中极端稀缺，模型训练策略的边际收益会被召回覆盖上限强烈约束。

## 数据与防穿越边界

本轮沿用全局时间轴切分，特征窗口严格位于 label 窗口之前：

```text
t_feature < t_label
```

切分审计：

| item | value |
| --- | ---: |
| raw_rows | 6,928,123 |
| raw_sessions | 1,671,803 |
| feature_rows | 5,817,330 |
| label_event_rows | 1,110,793 |
| label_target_rows | 817,232 |
| sessions_with_both_sides | 33,703 |
| boundary_violations | 0 |

Order 标签分布：

| item | value |
| --- | ---: |
| order_label_events | 7,417 |
| order_label_sessions | 4,411 |
| candidate_rows_for_eval | 133,769,637 |
| order_positive_rows_in_candidates | 307 |

候选内 Order 正样本覆盖极低：

```text
candidate_order_event_coverage = 307 / 7,417 = 0.0414
```

解释：即使排序模型把候选内的 Order 正样本全部排入 Top-20，最终 NDCG 仍会被召回覆盖率压制。当前评估首先暴露的是 retrieval bottleneck，而不是纯 ranking bottleneck。

## 特征构建与数学表达

排序目标：

```text
P(order = 1 | session, aid, context)
```

训练样本采用困难负样本混合：

```text
positive : hard_click_negative : hard_graph_negative : random_negative
```

三组消融：

| group | ratio | hypothesis |
| --- | --- | --- |
| A | 1 : 8 : 10 : 2 | click hard 与 graph hard 相对平衡 |
| B | 1 : 10 : 8 : 2 | 更强调“高点击但未购买”的交易边界 |
| C | 1 : 6 : 12 : 2 | 更强调“高共现权重但未转化”的伪相关边界 |

困难负样本定义：

```text
hard_click = 1[local_click_count >= 2 and target_order = 0]
hard_graph = 1[graph_w_cart_order_to_cart_order >= q_0.80 and target_order = 0]
random = 1[target_order = 0 and not hard_click and not hard_graph]
```

NDCG@20 评估公式：

```text
DCG@20 = sum_i rel_i / log2(rank_i + 1)
IDCG@20 = best possible DCG@20 for the session
NDCG@20 = DCG@20 / IDCG@20
```

其中 `rel_i = 1` 表示预测 Top-20 中命中真实购买商品，否则为 0。该指标惩罚低位命中，优先奖励把购买商品排在前几位的模型。

## 离线评估与指标对比

Order LightGBM 使用二分类目标，三组模型均在未采样全量候选验证集上评估：

| group | train_rows | pos | neg | neg/pos | hit_sessions | top20_hits | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A 1:8:10:2 | 6,442 | 307 | 6,135 | 19.98 | 227 | 307 | 0.034552 |
| B 1:10:8:2 | 6,463 | 307 | 6,156 | 20.05 | 227 | 307 | 0.035151 |
| C 1:6:12:2 | 6,374 | 307 | 6,067 | 19.76 | 227 | 307 | 0.034143 |

排序结论：

```text
B > A > C
```

B 相对 A 的提升：

```text
relative_gain = (0.035151 - 0.034552) / 0.034552 = 1.73%
```

B 相对 C 的提升：

```text
relative_gain = (0.035151 - 0.034143) / 0.034143 = 2.95%
```

解释：在当前候选集下，更强调 `hard_click_negative` 的 B 组最优，说明 Order 模型确实需要学习“高点击但未购买”的交易决策边界。C 组加大 graph-hard 权重后反而最差，说明高 Cart/Order 共现边在当前召回图中可能包含更多伪相关或长尾噪声。

特征重要性在三组中基本稳定，Top 特征集中在：

```text
local_cart_count
candidate_score
graph_w_cart_order_to_cart_order
local_click_count
local_interaction_count
recent_24h_interactions
delta_t
```

这说明 Order ranker 并非只靠全局流行度或候选 rank 工作，局部购物意图和 cart/order 共现图均被模型利用。

## 工程变更与可复现性

本轮除训练 A/B/C 三组 Order 模型外，还完成了三项工程闭环。

首先，评估层从简单 Recall@20 升级为 Order 专用 NDCG@20。新评估逻辑按 session 聚合真实 Order ground truth，并对预测 Top-20 中不同排名的命中施加对数折扣：

```text
gain(rank) = 1 / log2(rank + 1)
```

该改动使评估目标从“是否命中”转为“是否把真实购买商品排在足够靠前的位置”，更符合订单转化场景的商业排序要求。

其次，新增 Order LightGBM 训练入口。训练侧读取困难负样本下采样数据，评估侧固定使用未采样全量候选验证集，避免下采样分布污染最终指标。

第三，困难负样本管道加入配额耗竭审计与回退机制。采样不再假设每个正样本附近都存在足够 hard negatives，而是按全局 bucket 统计可用量：

```text
requested_rows -> available_rows -> planned_rows -> unfilled_shortfall_rows
```

若某类困难负样本不足，缺口优先回补到随机负样本或其他仍有容量的困难负样本池；全局仍不足时只记录 shortfall，不重复采样、不伪造样本。

## 性能瓶颈与物理开销

采样耗竭审计：

| group | requested_neg | planned_neg | unfilled_shortfall | available_hard_click | available_hard_graph | available_random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 6,140 | 6,140 | 0 | 586,431 | 2,374,828 | 130,808,071 |
| B | 6,140 | 6,140 | 0 | 586,431 | 2,374,828 | 130,808,071 |
| C | 6,140 | 6,140 | 0 | 586,431 | 2,374,828 | 130,808,071 |

本轮没有发生全局 hard-negative 耗竭；回退机制仍然必要，因为它防止未来在更窄 bucket、更稀疏目标或 Kaggle 全量切片中因局部稀疏导致重复采样或中途失败。

物理开销：

| artifact | files | size |
| --- | ---: | ---: |
| full labeled validation candidates | 16 | 2.28 GB |
| A sampled train parts | 16 | 0.57 MB |
| B sampled train parts | 16 | 0.58 MB |
| C sampled train parts | 16 | 0.56 MB |
| Order LightGBM models | 3 | 4.64 MB |

代码验证：

```text
18 passed
```

未记录项：

| item | status |
| --- | --- |
| wall_clock_time | [待补充] |
| peak_memory | [待补充] |
| Kaggle full train runtime estimate | [待补充] |

## 结论

本轮 A/B/C 消融确认：Order 模型的第一版困难负采样应采用 B 组 `1 : 10 : 8 : 2`，即增加“仅 Click 未 Order”负样本权重，而不是继续加大 graph-hard 权重。

但当前最主要的商业瓶颈不是 LightGBM 参数，也不是 hard-negative 细微比例，而是候选召回覆盖不足：全量 Order label events 为 7,417，候选集中仅命中 307。下一阶段应优先提高 Order 召回覆盖，再继续讨论 ranker 特征与采样策略，否则 NDCG 的可提升空间会被候选集上限锁死。
