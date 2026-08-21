# 2026-08-21 OTTO Time-Split Ranker Baseline

## 实验假设与核心矛盾

本轮目标是把候选召回、交互特征、标签构造和点击排序基线串成一条严格可审计的链路，验证在真正的时间轴切分下，局部行为与图信号是否仍能提供稳定增益。核心矛盾不是“模型够不够复杂”，而是样本极度稀疏、正负比例极端失衡，且任何穿越边界都会让离线指标失真。

## 数据与防穿越边界

本地验证改成全局时间切分，而不是按 session 尾部切分。特征窗口只允许使用 cutoff 之前的数据，label 窗口来自 cutoff 之后的数据，强制边界为：

```text
t_feature < t_label
```

严格审计结果：

| item | value |
| --- | ---: |
| rows | 6,928,123 |
| sessions | 1,671,803 |
| feature_rows | 5,817,330 |
| label_event_rows | 1,110,793 |
| label_target_rows | 817,232 |
| sessions_with_both_sides | 33,703 |
| strict_ok_sessions | 33,703 |
| boundary_violations | 0 |

这说明时间切分本身是干净的，没有同一 session 的未来事件渗入特征侧。和上一版按 session 尾部切分相比，这次验证更严格，也更接近真实线上条件。

标签分布：

| type | total_labels |
| ---: | ---: |
| click | 803,430 |
| cart | 82,356 |
| order | 7,417 |

这组分布天然高度不平衡，后续 Click baseline 只是第一步，cart/order 需要单独目标或多目标训练。

## 特征构建与数学表达

候选样本仍然以 `(session, aid)` 为主键。对每个候选对，保留三类核心信号：

```text
n_click(s, i), n_cart(s, i), n_order(s, i)
is_repeated_item = 1[n_total(s, i) > 0]
delta_t(s, i) = t_last(s) - t_last(s, i)
```

图信号保留三路来源：

```text
w_total = w_click_to_click + w_cart_order_to_cart_order + w_click_to_cart_order
```

对 `delta_t` 采用两种缺失策略做消融：

```text
delta_t_nan: 保留缺失值，由树模型学习缺失分裂方向
delta_t_filled: 缺失填充为 9,999,999
```

严格时间切分后的候选集规模：

| item | value |
| --- | ---: |
| candidate_rows | 133,769,637 |
| sessions | 1,429,020 |
| min_candidates_per_session | 50 |
| mean_candidates_per_session | 93.609 |
| median_candidates_per_session | 100 |
| max_candidates_per_session | 100 |

候选集内的正样本命中：

| type | positive_rows |
| ---: | ---: |
| click | 19,900 |
| cart | 1,776 |
| order | 307 |

Click 目标的原始不平衡比：

```text
negative = 133,769,637 - 19,900 = 133,749,737
imbalance_ratio = negative / positive = 6,721.09
```

这里的数值只针对候选内可学习样本，不等于全量 label 分布。严格 time-cut 后，召回阶段的难度明显上升，说明之前依赖 session 尾部验证的指标偏乐观。

## 离线评估与指标对比

严格 time-cut 下的候选召回结果：

| metric | value |
| --- | ---: |
| weighted_recall@20 | 0.030002 |

这说明在真正的时间边界下，现有三路共现召回的覆盖仍然偏弱，后续不能直接把旧验证结果和这条结果混为一谈。

Click 基线采用 LightGBM，并在 labeled features 上做 session hash holdout 诊断。两种 `delta_t` 策略的对比结果如下：

| variant | eval_positive_rows | top20_hits | recall_at_20 |
| --- | ---: | ---: | ---: |
| delta_t_nan | 1,946 | 1,607 | 0.8258 |
| delta_t_filled | 1,946 | 1,616 | 0.8304 |

结论很明确：在这条 Click baseline 上，`delta_t_filled=9,999,999` 略优于保留 NaN，差距不大，但方向清晰。

进一步看特征重要性，模型主要依赖：

```text
local_interaction_count
is_repeated_item
graph_w_click_to_click
candidate_score
graph_weight_sum / graph_weight_max
```

这说明当前 baseline 的收益主要来自局部交互证据和图召回信号，`delta_t` 的作用更像边界修饰项，而不是主驱动因子。

## 性能瓶颈与物理开销

整条管道采用分块落盘，避免把候选、特征、标签拼成超级宽表：

```text
time split -> graph build -> candidate retrieval -> interaction features -> label join -> negative sampling -> click baseline
```

物理开销统计：

| artifact | size |
| --- | ---: |
| time_split | 66.49 MB |
| strict candidates parts | 554.29 MB |
| strict interaction features | 1.68 GB |
| labeled features | 2.28 GB |
| downsampled click training set | 9.97 MB |
| LightGBM models | 3.31 MB |

当前未记录项：

| item | status |
| --- | --- |
| wall_clock_time | [待补充] |
| peak_memory | [待补充] |
| full Kaggle runtime estimate | [待补充] |

## 结论

今天真正完成的是三件事：一是把验证边界从 session 局部尾部切分升级为全局时间切分，并消除了边界泄漏；二是把严格 time-cut 的候选、交互特征和 click 标签链路打通；三是跑出了第一条 click baseline，确认 `delta_t_filled` 在当前设定下略优于 `delta_t_nan`。

下一步不该继续堆特征，而应该先补 `cart/order` 目标、再做多目标排序消融，并补齐时间、内存和更稳定的离线评估记录。
