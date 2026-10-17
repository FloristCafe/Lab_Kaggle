# E3：空间容量受控对照阶段结果

## 1. 实验假设与核心矛盾

在clean标签、普通损失和自然采样不变时，仅替换空间主干，检验容量/下采样瓶颈能否解释EXPLORE零预测与Goal定位失败。
**Res6没有改善当前预算下的EXPLORE选择或整体Joint；不能由此排除全部容量/优化问题。NOT READY。** 当前已完成4组空间结构，下一项=容量组完成，进入E4配对。

## 2. 数据与防穿越边界

同一725场，580训练/145验证，259,507训练候选单位行、64,414相同clean验证行；match和可用source seed隔离。输入≤t合法观测，历史意图赋值<t；未来轨迹仅用于标签。
同一seed42、normal loss、25%previous-context dropout、batch16、每轮1000更新、最多6轮；参数差异仅model.architecture。数据、源码、候选标签、训练/验证文件与每轮实际采样类别直方图完全一致。
每组96,000单位曝光，EXPLORE共1,091次；有限预算不是完整遍历六次。选模为mode-head Macro-F1，Joint打破平局。

## 3. 特征构建与数学表达

共用每帧18通道、三帧ConvGRU(hidden32)、unit22→query32、五类mode和五张576格条件goal；π(m,g)=π(m)π(g|m)，推理联合argmax。无R head、无attention/SE/FiLM/Transformer。
Current空间路径为32→48→64→48→32的两级down/up，6×6瓶颈带全局均值；Res6替换为6个64ch残差块、每块两层3×3、stride1、24×24保持，末端投影回32。
后续Res8为8个128ch残差块；dilated6为64ch、d=[1,2,3,5,8,4]。编码器之外头部与时序模块保持不变。该treatment同时改变深度/宽度/下采样/瓶颈全局路径，不能把差值解释为仅参数数量的作用。
各组GroupNorm具有全图空间耦合，Current也有显式全局均值，不能以局部卷积感受野断言其看不到远处。E0单点梯度图属于当时权重的空间敏感性诊断，不能代替新模型训练后的ERF/细定位结论。

## 4. 离线评估与指标对比

| Architecture | Trainable params | Best epoch | Mode Macro-F1 | Balanced acc. | 执行mode Macro-F1 | Joint | EXPLORE recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 185,809 | 6 | 0.4737 | 45.78% | 0.4971 | 19.14% | 0.00% |
| res6 | 528,977 | 6 | 0.4739 | 45.69% | 0.5097 | 18.72% | 0.00% |
| res8 | 2,471,569 | 2 | 0.4659 | 46.46% | 0.4594 | 17.89% | 0.00% |
| dilated6 | 528,977 | 6 | 0.4678 | 45.54% | 0.5014 | 19.15% | 0.00% |

Res6相对Current：Macro-F1 +0.000170，Joint -0.421pp，净正确单位数-271。联合执行mode Macro-F1略升，但条件goal与Joint没有同步收益。

| Architecture | 专家mode条件Goal Top1 | Top5 | Top10 | Manhattan error | Entropy/nat |
|---|---:|---:|---:|---:|---:|
| current | 22.87% | 36.52% | 45.55% | 4.699 | 3.884 |
| res6 | 22.37% | 35.41% | 44.22% | 4.897 | 3.797 |
| res8 | 21.81% | 33.98% | 42.34% | 5.452 | 4.088 |
| dilated6 | 22.80% | 36.66% | 45.82% | 4.698 | 3.802 |

| Architecture | Mode | Precision | Recall | F1 | 条件Goal Top1 | Goal error | Goal entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| current | SCORE | 77.95% | 90.72% | 0.8385 | 73.33% | 0.987 | 1.293 |
| current | EXPLORE | 0.00% | 0.00% | 0.0000 | 9.23% | 5.927 | 5.161 |
| current | DEFEND | 55.17% | 8.99% | 0.1547 | 79.76% | 0.655 | 1.318 |
| current | INTERCEPT | 90.58% | 94.79% | 0.9264 | 5.21% | 6.030 | 4.745 |
| current | RECHARGE | 64.68% | 34.37% | 0.4489 | 66.87% | 0.974 | 1.910 |
| res6 | SCORE | 79.94% | 89.37% | 0.8439 | 72.18% | 1.049 | 1.411 |
| res6 | EXPLORE | 0.00% | 0.00% | 0.0000 | 7.71% | 6.255 | 5.178 |
| res6 | DEFEND | 60.50% | 8.09% | 0.1428 | 78.86% | 0.619 | 1.386 |
| res6 | INTERCEPT | 89.99% | 94.98% | 0.9242 | 4.87% | 6.282 | 4.589 |
| res6 | RECHARGE | 63.14% | 35.99% | 0.4584 | 66.67% | 0.980 | 1.978 |
| res8 | SCORE | 80.83% | 86.61% | 0.8362 | 71.32% | 1.095 | 1.438 |
| res8 | EXPLORE | 0.00% | 0.00% | 0.0000 | 10.33% | 6.212 | 4.790 |
| res8 | DEFEND | 43.70% | 2.92% | 0.0548 | 77.74% | 0.948 | 1.571 |
| res8 | INTERCEPT | 90.09% | 94.03% | 0.9201 | 4.27% | 7.025 | 4.996 |
| res8 | RECHARGE | 55.43% | 48.73% | 0.5186 | 67.16% | 0.963 | 1.799 |
| dilated6 | SCORE | 80.85% | 90.25% | 0.8529 | 72.96% | 1.007 | 1.249 |
| dilated6 | EXPLORE | 0.00% | 0.00% | 0.0000 | 7.44% | 6.326 | 4.998 |
| dilated6 | DEFEND | 63.89% | 5.17% | 0.0957 | 79.76% | 0.567 | 1.251 |
| dilated6 | INTERCEPT | 90.31% | 95.33% | 0.9275 | 5.22% | 6.021 | 4.649 |
| dilated6 | RECHARGE | 61.89% | 36.95% | 0.4627 | 66.98% | 0.973 | 1.899 |

Res6的mode-head混淆矩阵，行真值/列预测，SCORE/EXPLORE/DEFEND/INTERCEPT/RECHARGE：

```text
[[10035     0    17  1120    57]
 [  140     0     0   560    26]
 [  268     0   144  1181   186]
 [ 1750     0    60 44485   539]
 [  360     0    17  2085  1384]]
```

六轮Res6训练mixed-policy和每轮验证的原始mode head都没有预测EXPLORE；最佳checkpoint的联合执行mode同样零预测。EXPLORE条件Goal Top1=7.71%，INTERCEPT条件Goal Top1=4.87%，均未显示目标定位改善。
结合E2：原空间主干在balanced曝光下EXPLORE recall43.25%，因此架构加深不是出现非零探索预测的必要条件；不平衡抑制模式选择的证据比“仅因当前主干过浅”更强。这不证明容量永远无用，也不能把balanced的低precision/Goal归因给单一因素。
所有比较仍是单seed/同一fold，预设预算末轮仍可成为最佳。未做充分优化、容量×采样交互或跨seed对照，不能因Res6没有收益而宣判容量假设已被排除。
下一项按预注册E4_without_r推进；E3各项都相对Current使用normal loss和自然采样，以隔离空间结构。组合最优标签/采样/结构留待E6实际结果后决定。
Replan未启用；原始日志中的always-Replan F1不构成已学会R的证据。离线持久性是相邻标注采样行上的预测一致性，未测新闭环完成率或胜率；geodesic/online completion无可靠证据时保持null。

## 5. 性能瓶颈与物理开销

| Architecture | Wall/min | Peak allocated MiB |
|---|---:|---:|
| current | 16.01 | 118.64 |
| res6 | 18.32 | 188.58 |
| res8 | 18.65 | 332.11 |
| dilated6 | 40.35 | 188.58 |

Res6峰值reserved=214.00MiB，显存未构成限制；停止在第6轮预算上限，并非patience早停或OOM。CPU峰值/能耗[待补充：未采集]。
Res6 MLflow=d4583d63922d429fb3255edd958a0918，数据manifest=8a8fbfb269daaa33226d95cfb7c94bcaf692b0445b5aa0c8a18fdc63fde24f13；原始checkpoint和训练指标保持不变。报告程序仅读取指标并统计模型参数，不做forward、训练或比赛。
[待补充：剩余架构summary.json，当前remaining=[]]。当前NOT READY，不进入正式MAPPO。
