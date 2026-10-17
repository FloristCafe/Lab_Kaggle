# E4/E5：独立R与部署memory对照

## 1. 实验假设与核心矛盾

在宏review时刻加入独立Bernoulli R，检验Continue能否保持有效意图，并在因果规则、污染快照和策略历史memory间保持一致。当前状态=FINISHED，结论仍为NOT READY，不能仅凭Replan正类F1决定可部署性。

## 2. 数据与防穿越边界

725场，580/145隔离；matched E4两组scope=review_epochs、单位候选/每轮曝光一致。验证共105,267单位，与E1的64,414单位/stride4协议不同，E1→E4绝对差值不能全归因于R。
输入仅≤t合法观测，memory目标赋值来自<t的规则/策略输出；未来段与区域只生成监督。unknown filtered state保持mask，不默认Continue。
E5_corrupted的预设config同时用于训练和该arm的validation上下文，所以其原始summary的选模分数是在corrupted memory条件下。不能把它与scripted选模的E4 summary差异解释为纯训练corruption效应；下面用相同FP32冻结评估条件再比较。模型选择条件的差异仍作为局限保留。

## 3. 特征构建与数学表达

π(r,m,g|s,Iprev)：r=0保持(previous mode,goal,age)，r=1重新联合选择(m,g)；无有效previous时确定性强制Replan。Continue logprob仅logP(r=0)，Replan logprob为logP(r=1)+logP(m)+logP(g|m)。没有第577个KEEP。
E4两组关闭旧无界previous bias、context dropout=0；仅R head/条件执行语义不同。R=0样本只训练R，目标头在这些样本没有额外梯度；这是该策略分解的组成部分。
Hysteresis仅对一个(previous mode,goal)联合对加η≤0.5、age衰减、invalid时关闭；已测试其不改变训练目标或梯度。E5_hysteresis与combination采用相同已训练权重的η=0/.25推理对照，避免重复训练相同目标。

## 4. 离线评估与指标对比

### 训练各arm选中checkpoint（原始选模条件）

| Arm | 选模memory | Best epoch | Head Macro-F1 | 执行mode Macro-F1 | Joint | Free R1 F1 | Continue recall | False Replan |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| without_r | scripted | 3 | 0.4991 | 0.4803 | 18.31% | 0.9608 | 0.00% | 100.00% |
| with_r | scripted | 6 | 0.5065 | 0.4921 | 16.62% | 0.9694 | 79.22% | 20.78% |
| corrupted | corrupted | 5 | 0.5105 | 0.4596 | 17.55% | 0.9733 | 34.86% | 65.14% |

without_r free R confusion (row true, column prediction; Continue/Replan):

```text
[[    0  7512]
 [    0 92028]]
```

with_r free R confusion (row true, column prediction; Continue/Replan):

```text
[[ 5951  1561]
 [ 3991 88037]]
```

corrupted free R confusion (row true, column prediction; Continue/Replan):

```text
[[ 1670  3120]
 [ 1455 83494]]
```

Without_r是always-Replan对照，高R1 F1不代表学会R。应看自由决策单位的Continue召回、误重规划率与完整意图正确性，并把无效previous的强制分支从学习指标中分离。

| Arm | Mode | Precision | Recall | F1 | 条件Goal Top1 | Goal error | Goal entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| without_r | SCORE | 86.43% | 78.89% | 0.8249 | 72.31% | 1.058 | 1.258 |
| without_r | EXPLORE | 0.00% | 0.00% | 0.0000 | 6.93% | 6.226 | 5.084 |
| without_r | DEFEND | 50.65% | 13.93% | 0.2185 | 80.29% | 0.638 | 1.322 |
| without_r | INTERCEPT | 89.49% | 95.77% | 0.9253 | 5.09% | 6.204 | 4.766 |
| without_r | RECHARGE | 55.65% | 50.05% | 0.5270 | 66.42% | 0.976 | 1.554 |
| with_r | SCORE | 74.62% | 91.62% | 0.8225 | 72.23% | 1.000 | 1.663 |
| with_r | EXPLORE | 0.00% | 0.00% | 0.0000 | 8.96% | 5.750 | 5.300 |
| with_r | DEFEND | 58.30% | 17.00% | 0.2632 | 78.49% | 0.619 | 1.267 |
| with_r | INTERCEPT | 92.25% | 92.91% | 0.9258 | 5.40% | 5.760 | 4.754 |
| with_r | RECHARGE | 58.50% | 46.93% | 0.5208 | 67.87% | 0.948 | 1.875 |
| corrupted | SCORE | 80.58% | 86.69% | 0.8352 | 72.77% | 1.020 | 1.309 |
| corrupted | EXPLORE | 0.00% | 0.00% | 0.0000 | 8.13% | 6.023 | 5.134 |
| corrupted | DEFEND | 57.65% | 15.27% | 0.2414 | 80.91% | 0.549 | 1.209 |
| corrupted | INTERCEPT | 92.24% | 93.38% | 0.9281 | 5.30% | 6.044 | 4.706 |
| corrupted | RECHARGE | 50.80% | 59.41% | 0.5477 | 67.87% | 0.948 | 1.762 |

### 同权重、同145场验证的memory条件

| Arm | η | Memory | Joint | Free R1 F1 | Continue recall | False Replan | Reviews/frame | Previous age |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| without_r | 0.0 | scripted | 18.29% | 0.9608 | 0.00% | 100.00% | 0.390 | 7.66 |
| without_r | 0.0 | corrupted | 18.14% | 0.9722 | 0.00% | 100.00% | 0.390 | 17.86 |
| without_r | 0.0 | policy_replay | 17.31% | 0.9119 | 0.00% | 100.00% | 0.404 | 8.10 |
| with_r | 0.0 | scripted | 16.61% | 0.9694 | 79.49% | 20.51% | 0.390 | 7.66 |
| with_r | 0.0 | corrupted | 15.65% | 0.9657 | 80.25% | 19.75% | 0.390 | 17.86 |
| with_r | 0.0 | policy_replay | 13.23% | 0.9204 | 44.87% | 55.13% | 0.401 | 8.42 |
| with_r | 0.25 | scripted | 16.62% | 0.9694 | 79.49% | 20.51% | 0.390 | 7.66 |
| with_r | 0.25 | corrupted | 15.66% | 0.9657 | 80.25% | 19.75% | 0.390 | 17.86 |
| with_r | 0.25 | policy_replay | 12.77% | 0.9192 | 43.70% | 56.30% | 0.401 | 9.64 |
| corrupted | 0.0 | scripted | 18.13% | 0.9668 | 28.71% | 71.29% | 0.390 | 7.66 |
| corrupted | 0.0 | corrupted | 17.59% | 0.9733 | 35.70% | 64.30% | 0.390 | 17.86 |
| corrupted | 0.0 | policy_replay | 17.30% | 0.9167 | 14.94% | 85.06% | 0.403 | 7.92 |
| corrupted | 0.25 | scripted | 18.18% | 0.9668 | 28.71% | 71.29% | 0.390 | 7.66 |
| corrupted | 0.25 | corrupted | 17.61% | 0.9733 | 35.70% | 64.30% | 0.390 | 17.86 |
| corrupted | 0.25 | policy_replay | 17.19% | 0.9172 | 15.26% | 84.74% | 0.403 | 8.93 |

Scripted不是未来专家Goal，是真正因果规则memory；corrupted是标签盲的快照扰动；policy_replay把Actor过去输出回流memory，但专家物理轨迹保持固定，所以不是互动环境rollout。C的review时刻与可监督覆盖可能改变，比较时需同时读取coverage、previous-mode和age分布，不能仅看条件准确率。
离线规则previous-mode的DEFEND/INTERCEPT通道从未激活，实际策略会产生这些mode；这属于可测的支持域偏移，corruption引入随机mode为一种诊断/鲁棒性干预，不等同真实闭环错误轨迹。

### 固定物理帧分母：review时发出意图、非review时保持

| Arm | η | Memory | 有标签单位数 | Held-intent Joint | 已分配比例 | 当前坐标预测比例 |
|---|---:|---|---:|---:|---:|---:|
| without_r | 0.0 | scripted | 257,305 | 18.01% | 100.00% | 30.52% |
| without_r | 0.0 | corrupted | 257,305 | 17.86% | 100.00% | 31.45% |
| without_r | 0.0 | policy_replay | 257,305 | 16.97% | 100.00% | 39.63% |
| with_r | 0.0 | scripted | 257,305 | 16.22% | 100.00% | 25.57% |
| with_r | 0.0 | corrupted | 257,305 | 15.24% | 100.00% | 25.10% |
| with_r | 0.0 | policy_replay | 257,305 | 12.51% | 100.00% | 24.57% |
| with_r | 0.25 | scripted | 257,305 | 16.24% | 100.00% | 25.53% |
| with_r | 0.25 | corrupted | 257,305 | 15.25% | 100.00% | 25.08% |
| with_r | 0.25 | policy_replay | 257,305 | 12.08% | 100.00% | 23.66% |
| corrupted | 0.0 | scripted | 257,305 | 17.84% | 100.00% | 31.62% |
| corrupted | 0.0 | corrupted | 257,305 | 17.29% | 100.00% | 31.55% |
| corrupted | 0.0 | policy_replay | 257,305 | 16.77% | 100.00% | 35.81% |
| corrupted | 0.25 | scripted | 257,305 | 17.88% | 100.00% | 31.60% |
| corrupted | 0.25 | corrupted | 257,305 | 17.31% | 100.00% | 31.53% |
| corrupted | 0.25 | policy_replay | 257,305 | 16.69% | 100.00% | 35.21% |

该表在同一组全部clean有效帧评分，A/B/C分母一致，避免不同review覆盖率造成选择偏差。Scripted/corrupted的输入memory仍各自固定，输出意图仅为非review帧计分而保持；只有policy_replay将Actor输出回流输入。实际物理轨迹仍来自录像，不能将这个指标解释为环境得分。
同一(mode,goal)身份的Replan不再刷新age、blocked与到达标记，防止虚假续期。新runtime保留冻结脚本memory的原始特征与标签；兼容性及逐帧/分批一致性检查已通过。

### 实际环境短闭环：既有dev seed15001、双侧128步预算

| Arm | η | Side | Steps | Reviews/frame | Replan rate | Goal persistence | Nontrivial completion | Invalid persistence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| without_r | 0.0 | 0 | 129 | 0.287 | 100.00% | 37.03% | 0.92% | 0.00% |
| without_r | 0.0 | 1 | 129 | 0.326 | 100.00% | 67.48% | 0.45% | 0.00% |
| with_r | 0.0 | 0 | 129 | 0.318 | 100.00% | 34.34% | 1.62% | 0.00% |
| with_r | 0.0 | 1 | 129 | 0.287 | 100.00% | 15.62% | 0.92% | 0.00% |
| with_r | 0.25 | 0 | 129 | 0.310 | 100.00% | 50.26% | 1.02% | 0.00% |
| with_r | 0.25 | 1 | 129 | 0.295 | 100.00% | 43.02% | 2.57% | 0.00% |
| corrupted | 0.0 | 0 | 129 | 0.279 | 100.00% | 54.98% | 0.56% | 0.00% |
| corrupted | 0.0 | 1 | 129 | 0.341 | 100.00% | 65.07% | 1.68% | 0.00% |
| corrupted | 0.25 | 0 | 129 | 0.318 | 100.00% | 76.01% | 2.42% | 0.00% |
| corrupted | 0.25 | 1 | 129 | 0.318 | 100.00% | 72.68% | 1.60% | 0.00% |

Goal persistence比较有有效previous的目标身份是否保持，与R0比例分开；completion按每个非原地新意图首次抵达计数。Invalid persistence为0来自协议强制分支，不是学习成绩。在线无专家R真值，precision/recall/F1保持null，不制造标签。
这是人工截断的dev前缀，可能在当前option结束时略超128步，不报告完整比赛胜率。不能由单seed双侧前缀证明泛化或长期收益。
Q4/Q5仍须结合完整上述证据回答，不预设R或hysteresis已经解决mismatch。Q8当前NOT READY，不启动正式MAPPO。

## 5. 性能瓶颈与物理开销

四组后续训练在用户授权后并行执行；wall time反映共享设备竞争，不能与先前独占运行直接当成架构速度比较。GPU/host资源采样见autonomous_progress/resource_samples.csv。
冻结memory评估FP32、完整145场，每condition单独保存，支持中断恢复且检查checkpoint/数据/源代码哈希；额外训练更新数=0。CPU峰值/能耗归因[待补充：采样不等同独占归因]。
每组checkpoint、summary、config、dataset hash、源码hash和MLflow ID保持独立。原始训练产物与V2/官方环境未改写，最终holdout未执行。
