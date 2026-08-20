# 2026-08-20 OTTO Stage 3 Interaction Features

## 实验假设与核心矛盾

本轮增量假设：在三路共现召回候选集上补充 session-item 级局部行为、时间新鲜度和图来源权重，可以提升后续排序模型对 `click/cart/order` 目标的判别能力；核心矛盾是候选规模已达到 `158,392,531` 行，任何一次性宽表拼接都会放大内存与磁盘风险。

## 数据与防穿越边界

本地验证采用 session 尾部截断：每个长度不小于 2 的 session 留最后 1 个事件作为 label，其余事件作为 feature 侧历史。目标边界应强制满足：

```text
t_feature < t_label
```

当前审计结果：

| item | value |
| --- | ---: |
| label_sessions | 921,704 |
| strict_ok_sessions | 916,570 |
| timestamp_boundary_violations | 5,134 |

结论：当前切分在序列位置上成立，但不是严格时间戳无泄漏切分；`5,134` 个 session 出现 `feature_max_ts >= label_ts`，需要在进入正式 ranker 训练前修正。可选修正方式：

```text
1. 对同一 session 的 label timestamp 做严格过滤，只允许 ts < label_ts 的事件进入 feature。
2. 若最后时间戳存在多个并列事件，则整体作为 label group，避免同一时间戳事件互相泄漏。
```

验证 label 分布：

| type | total_labels |
| ---: | ---: |
| 0 click | 822,342 |
| 1 cart | 83,911 |
| 2 order | 15,451 |
| total | 921,704 |

Top-100 候选集规模：

| item | value |
| --- | ---: |
| candidate_rows | 158,392,531 |
| sessions | 1,671,803 |
| min_candidates_per_session | 50 |
| mean_candidates_per_session | 94.7435 |
| median_candidates_per_session | 100 |
| max_candidates_per_session | 100 |

候选集内可训练正负分布：

| item | value |
| --- | ---: |
| positive_rows_in_candidates | 449,518 |
| negative_rows_in_candidates | 157,943,013 |
| imbalance_ratio = negative / positive | 351.3608 |

按目标类型拆分的候选命中：

| type | positive_rows_in_candidates | candidate_recall@100 |
| ---: | ---: | ---: |
| 0 click | 359,517 | 0.4372 |
| 1 cart | 75,986 | 0.9056 |
| 2 order | 14,015 | 0.9071 |

未被候选集覆盖的 label 不应在 ranker 训练阶段被隐式忽略而不记录。当前 Top-100 候选整体 label 覆盖率：

```text
unweighted_label_coverage@100 = 449,518 / 921,704 = 0.4877
weighted_recall@100 = 0.8596
```

## 特征构建与数学表达

排序建模目标：

```text
P(y_type | session, aid, context)
```

局部交互统计：

```text
n_click(s, i)
n_cart(s, i)
n_order(s, i)
n_total(s, i) = n_click(s, i) + n_cart(s, i) + n_order(s, i)
```

重复曝光标记：

```text
is_repeated_item = 1[n_total(s, i) > 0]
```

时间新鲜度：

```text
delta_t(s, i) = t_last(s) - t_last(s, i)
```

缺失值策略保留两套列，供排序层消融：

```text
delta_t_nan: 保留缺失，由树模型学习 missing direction
delta_t_filled: 缺失填充为 9,999,999
```

图来源权重：

```text
w_total = w_click_to_click + w_cart_order_to_cart_order + w_click_to_cart_order
```

当前保留两类表达：一类是融合后的 `w_total`，另一类是三路独立图权重。前者降低维度，后者保留召回来源异质性；最终选择必须依赖离线消融，而不是在特征生成阶段提前压缩。

当前交互特征输出审计：

| item | value |
| --- | ---: |
| interaction_feature_rows | 158,392,531 |
| repeated_rows | 3,976,751 |
| delta_t_nan_nulls | 154,415,780 |
| unseen_candidate_ratio | 0.9749 |

解释：`delta_t_nan` 大规模缺失与候选来源一致，说明绝大多数候选来自图召回或热门补全，而非当前 session 内已交互 item。

## 离线评估与指标对比

召回阶段已有指标：

| metric | click | cart | order | weighted |
| --- | ---: | ---: | ---: | ---: |
| Recall@20 | 0.3552 | 0.8834 | 0.8953 | 0.8377 |
| Recall@100 / label coverage | 0.4372 | 0.9056 | 0.9071 | 0.8596 |

交互特征消融尚未完成：

| experiment | metric | result |
| --- | --- | --- |
| only `delta_t_nan` | Recall@20 / NDCG@20 / weighted metric | [待补充] |
| only `delta_t_filled` | Recall@20 / NDCG@20 / weighted metric | [待补充] |
| only `graph_weight_total` | Recall@20 / NDCG@20 / weighted metric | [待补充] |
| independent `graph_w_*` | Recall@20 / NDCG@20 / weighted metric | [待补充] |
| static + interaction combined baseline | Recall@20 / NDCG@20 / weighted metric | [待补充] |

必须补充的问题：

```text
1. 单凭 delta_t 系列特征是否能提升排序指标？
2. NaN 原生分裂是否优于 9,999,999 物理填充？
3. 三路图权重独立输入是否优于 total weight？
4. 加入 item/user 静态特征后，carts/orders 是否有稳定增益？
```

## 性能瓶颈与物理开销

当前采用分治管道：

```text
session bucket -> candidate chunk -> local join -> parquet part
```

内存复杂度目标：

```text
full_join_memory  = O(N_candidates)
chunk_join_memory = O(N_candidates / K)
```

落盘规模：

| artifact | value |
| --- | ---: |
| candidate_parts_size | 592.26 MB |
| interaction_feature_parts | 16 |
| interaction_feature_parts_size | 1.508 GB |
| strict_chunk_smoke_parts | 2 |
| strict_chunk_smoke_total_size | 113.15 MB |
| strict_chunk_smoke_rows_per_part | 4,953,399 |

尚未记录的物理开销：

| item | status |
| --- | --- |
| wall_clock_time | [待补充] |
| peak_memory | [待补充] |
| Kaggle full train runtime estimate | [待补充] |
| OOM failure threshold by chunk count | [待补充] |

结论：当前特征表已经具备进入 ranker baseline 的结构条件，但存在两个必须先处理的问题：一是本地时间戳防穿越边界有 `5,134` 个 session 未满足严格 `t_feature < t_label`；二是交互特征还没有任何单特征或特征组排序消融，不能声称其对最终模型有效。
