# E1_clean_seed42：正式结果复核

## 1. 实验假设与核心矛盾

在当前空间主干不变、普通 CE+目标区域 NLL 的条件下，检验严格合法 SAP 与删失清理后的有限预算 Goal IL baseline。
**运行已正常完成，但 EXPLORE 仍从未被选择，NOT READY。** 最佳第 6 轮；停止第 6 轮，达到配置上限，不是早停收敛。
本项不是 Replan/Continue 训练：model.replan=False。原始日志中约95.9%的 Replan正类F1来自 always-Replan，对 R 能力没有证明力。

## 2. 数据与防穿越边界

725 场，580/145 match/source-seed 隔离；训练 eligible 单位行 259,507，验证 64,414，固定每四帧可监督单位行。网络特征仅≤t合法观测，历史 memory 赋值来自<t；未来轨迹只参与监督标签。
数据 manifest、模型/标签/训练代码哈希、检查点 epoch/分数与预注册配置均匹配，MLflow状态FINISHED。复核只读取现有产物，没有额外训练或赛程。
每轮16,000单位曝光，六轮合计 96,000 次；其中EXPLORE 1,091 次（1.14%），不是对所有训练标签完整遍历六次。

## 3. 特征构建与数学表达

18通道×3帧ConvGRU，unit22→query32，π(m,g)=π(m)π(g|m)，推理联合argmax。
无R head；Previous Goal为因果规则memory，训练按帧25%清零该上下文。损失为confidence加权mode CE和专家mode条件目标区域NLL。
标签清理同时覆盖SAP合法性和删失尾段，因此即使后续matched dirty对照有收益，也不能在没有进一步消融时全归因于SAP一项。

## 4. 离线评估与指标对比

| 指标 | 最佳检查点 |
|---|---:|
| Mode-head accuracy | 87.04% |
| Mode-head Macro-F1 | 0.4737 |
| Mode-head balanced accuracy | 45.78% |
| 联合argmax执行mode accuracy | 81.43% |
| 执行mode Macro-F1 / balanced accuracy | 0.4971 / 54.33% |
| Joint Mode+Goal correctness | 19.14% |
| Goal Top1 / Top5 / Top10（专家mode条件） | 22.87% / 36.52% / 45.55% |
| Goal最短Manhattan误差 / entropy | 4.699格 / 3.884 nat |
| 预测目标距离 / 专家region最短距离 | 8.342 / 9.596格 |
| 段内相邻样本预测意图一致率 | 43.41%（54,412对） |

| Mode | support | Precision | Recall | F1 | 专家mode条件Goal Top1 | Joint |
|---|---:|---:|---:|---:|---:|---:|
| SCORE | 11,229 | 77.95% | 90.72% | 83.85% | 73.33% | 71.70% |
| EXPLORE | 726 | 0.00% | 0.00% | 0.00% | 9.23% | 0.00% |
| DEFEND | 1,779 | 55.17% | 8.99% | 15.47% | 79.76% | 31.25% |
| INTERCEPT | 46,834 | 90.58% | 94.79% | 92.64% | 5.21% | 4.48% |
| RECHARGE | 3,846 | 64.68% | 34.37% | 44.89% | 66.87% | 42.17% |

混淆矩阵，行真值、列预测，顺序 SCORE/EXPLORE/DEFEND/INTERCEPT/RECHARGE：

```text
[[10187     0     8  1009    25]
 [  168     0     7   533    18]
 [  358     0   160  1073   188]
 [ 1880     0    67 44396   491]
 [  475     0    48  2001  1322]]
```

| Epoch | Val Macro-F1 | Balanced accuracy | Joint | 训练EXPLORE曝光 | 验证EXPLORE预测数 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.3872 | 39.91% | 15.75% | 192 | 0 |
| 2 | 0.4573 | 46.21% | 18.11% | 176 | 0 |
| 3 | 0.4550 | 48.07% | 17.41% | 163 | 0 |
| 4 | 0.4246 | 40.06% | 17.77% | 183 | 0 |
| 5 | 0.4570 | 45.28% | 18.80% | 183 | 0 |
| 6 | 0.4737 | 45.78% | 19.14% | 194 | 0 |

主要解释：
- 六轮训练中的mixed-policy统计与每轮验证均无EXPLORE预测；这不是仅在陌生验证地图上失败，也不是单纯goal head把EXPLORE位置猜错。
- 普通损失下少数类梯度分配不足仍是有力假设，但尚未做weighted/balanced对照，不能宣判主因，也不能排除标签语义、容量和优化问题。
- INTERCEPT mode recall=94.79%，但其专家mode条件Goal Top1仅5.21%，对应joint仅4.48%。长期攻击目标定位是独立的主要瓶颈。
- mode-head与联合执行mode指标不同；DEFEND条件Goal较好，但mode-head recall低，不能混成一个accuracy。
- 旧400场mode accuracy86.20%、joint18.81%只作历史参考；新580/145隔离、clean标签、memory和采样改变，19.14%不能当成可归因的提升。
- 六轮末仍选出最佳macro-F1，且训练/验证并无旧实验那种明显整体过拟合分离；有限96k曝光不能证明已充分优化。
- 离线goal persistence/segment consistency是在已标注采样行的相邻预测比较，不是闭环option完成。completion、invalid persistence、geodesic无可靠证据时保持null。

下一项为E1_dirty_control：相同扩展数据、划分、memory、网络与预算，仅训练标签视图改为旧噪声标签，验证仍严格clean。该对照完成后进入E2_weighted和E2_balanced，分离清理与梯度分配的影响。[待补充：这三项正式summary.json]。
E3容量、E4独立R、E5部署memory仍待执行；本项结果不足以修改架构或选E6，也不进入正式MAPPO。

## 5. 性能瓶颈与物理开销

完整训练+验证 960.37s（16.01分钟）；GPU峰值 allocated=118.64MiB、reserved=132.00MiB，实际B16/BF16已测量。
训练预算6,000次优化、96,000单位曝光；每轮完整验证64,414单位。CPU峰值/能耗[待补充：未采集]。显存远未达到8GiB限制，本次终止原因是轮数预算，不是OOM。
记录：MLflow df87831230cf4396b578a9fe74b907e4；数据哈希 8a8fbfb269daaa33226d95cfb7c94bcaf692b0445b5aa0c8a18fdc63fde24f13。检查点和原始指标保持不变，复核证据见verification.json。
