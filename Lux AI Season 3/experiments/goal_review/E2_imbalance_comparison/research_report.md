# E2：类别不平衡对照阶段结果

## 1. 实验假设与核心矛盾

固定clean数据、网络和训练预算，仅改变Mode梯度分配，检验EXPLORE零预测是否主要受类别不平衡影响。
**平衡采样使EXPLORE Recall达到43.25%，但Precision仅5.94%，整体Joint降至17.85%；不平衡贡献得到支持，可靠Goal能力尚未建立，NOT READY。** E2三项已完成；下一步按既定E3计划独立改变空间主干，不自动叠加胜出的loss/sampler。
这是一条seed42/fold0的有限预算结果，支持梯度分配有影响，不能据此排除标签/容量/优化/部署偏移。

## 2. 数据与防穿越边界

725场，580/145 match与可用source-seed隔离；259,507训练eligible单位行、64,414相同clean验证行。标签候选计数[47522, 2888, 8119, 185307, 15671]，EXPLORE占1.11%。
当前特征仅≤t合法观测，previous意图赋值来自<t，未来轨迹只进入监督。数据/源码哈希、候选行、划分与验证标签一致。
normal与weighted的每轮实际采样类别直方图完全相同，六轮均96,000单位曝光，EXPLORE均1,091次；改变的是损失权重，没有额外喂更多稀有样本。balanced干预改变曝光直方图，预算不变。

## 3. 特征构建与数学表达

18通道×3帧ConvGRU、unit22、当前空间主干、无R head；π(m,g)=π(m)π(g|m)，推理(m,g)联合argmax。
Mode loss为Σconfidence_i·w_y·CE_i / Σconfidence_i·w_y，w_c=min(N/(5N_c),5)。实际权重（SCORE/EXPLORE/DEFEND/INTERCEPT/RECHARGE）=[1.0921552181243896, 5.0, 5.0, 0.2800833284854889, 3.311939239501953]。
目标区域NLL、Previous Goal dropout=25%、lr=1e-3、weight_decay=1e-4、batch16、每轮1000更新和最多6轮均不变；loss数值因加权改变，不直接按train_loss高低判断优化优劣。

## 4. 离线评估与指标对比

同一预注册checkpoint选择：mode-head Macro-F1，Joint打破平局。下列是选中checkpoint，不以EXPLORE峰值事后换模型。

| Arm | Best epoch | Mode accuracy | Mode Macro-F1 | Balanced acc. | 联合执行mode Macro-F1 | Joint | EXPLORE recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | 6 | 87.04% | 0.4737 | 45.78% | 0.4971 | 19.1371% | 0.00% |
| weighted | 6 | 83.35% | 0.5281 | 57.27% | 0.4604 | 19.1402% | 0.55% |
| balanced | 6 | 75.28% | 0.5223 | 65.88% | 0.4265 | 17.8471% | 43.25% |

加权相对normal：Macro-F1 +0.0544，balanced accuracy +11.49pp；联合执行mode Macro-F1 -0.0367。原始mode head收益没有自动传到联合策略。
Joint只净增加2个正确单位样本，不能描述为有意义的整体Goal收益。Goal条件Top1为normal=22.87%、weighted=22.46%，Manhattan误差4.699/4.740格。

| Mode | normal precision / recall / F1 | weighted precision / recall / F1 | balanced precision / recall / F1 |
|---|---:|---:|---:|
| SCORE | 77.95% / 90.72% / 0.8385 | 69.42% / 93.88% / 0.7982 | 79.58% / 87.55% / 0.8338 |
| EXPLORE | 0.00% / 0.00% / 0.0000 | 17.39% / 0.55% / 0.0107 | 5.94% / 43.25% / 0.1044 |
| DEFEND | 55.17% / 8.99% / 0.1547 | 36.33% / 35.92% / 0.3612 | 18.16% / 57.62% / 0.2762 |
| INTERCEPT | 90.58% / 94.79% / 0.9264 | 95.28% / 84.92% / 0.8980 | 97.29% / 74.21% / 0.8419 |
| RECHARGE | 64.68% / 34.37% / 0.4489 | 47.90% / 71.06% / 0.5722 | 47.50% / 66.77% / 0.5551 |

加权模型原始mode-head混淆矩阵，行真值/列预测，SCORE/EXPLORE/DEFEND/INTERCEPT/RECHARGE：

```text
[[10542     1    94   449   143]
 [  298     4    18   357    49]
 [  261     0   639   530   349]
 [ 3757    17   857 39771  2432]
 [  327     1   151   634  2733]]
```

最佳模型EXPLORE原始head：预测23次、正确4次，precision17.39%、recall0.55%、F1=0.0107；联合执行mode：预测38次、mode正确3次，但这3次的goal均未命中，EXPLORE Joint仍0。
EXPLORE专家mode条件Goal Top1=7.71%，误差6.340格、entropy=5.225nat；INTERCEPT条件Goal Top1=4.90%。目标空间推断没有被Mode类别权重解决。

| Weighted epoch | Macro-F1 | EXPLORE recall | head预测数 | head正确数 | 联合mode正确数 | EXPLORE Joint |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.4741 | 0.00% | 0 | 0 | 0 | 0.00% |
| 2 | 0.5208 | 4.68% | 269 | 34 | 16 | 0.00% |
| 3 | 0.4532 | 0.00% | 0 | 0 | 0 | 0.00% |
| 4 | 0.5233 | 0.00% | 0 | 0 | 0 | 0.00% |
| 5 | 0.5269 | 1.24% | 29 | 9 | 3 | 0.00% |
| 6 | 0.5281 | 0.55% | 23 | 4 | 3 | 0.00% |

第2轮EXPLORE recall短暂4.68%，最终0.55%，并非稳定学会；按EXPLORE峰值挑epoch2会改变预注册选择标准。模式头/执行mode/完整Joint必须分开解释。
无R head，本轮always-Replan的高正类F1不能作为学会R的证据。没有新增闭环赛程，completion与invalid persistence无可靠证据时保持null。
单seed且样本按录像相关，未做跨seed或逐match配对置信区间，微小Joint差异仅作描述。Balanced实际曝光直方图=[19272, 19087, 19092, 19222, 19327]；性能与取舍见三组表，仍需要容量与部署条件对照。


### Balanced sampler：E2完整结果

每类约20%曝光，EXPLORE合计19,087次，是normal的17.49倍；总预算仍96,000。模型、候选集、验证标签、seed和源码不变。
最佳epoch6：EXPLORE mode-head正确314/726，recall43.25%，precision5.94%，F1=0.1044；预测EXPLORE 5,289次，其中4,975次误报，4,055次来自真实INTERCEPT。
联合(m,g)argmax后的EXPLORE mode正确181/726、recall24.93%、precision3.82%；完整EXPLORE intent命中18/726、2.48%。Mode recall与最终Goal成功不能混为一谈。
总体Joint=17.85%，比normal下降1.29pp；联合执行mode Macro-F1=0.4265，模式头accuracy=75.28%。在自然类别分布下，少数类召回提升伴随明显误报和多数类代价。

| Balanced mode | 专家mode条件Goal Top1 | Manhattan误差 | Entropy/nat | Joint |
|---|---:|---:|---:|---:|
| SCORE | 72.74% | 1.027 | 1.139 | 69.63% |
| EXPLORE | 9.37% | 5.905 | 4.640 | 2.48% |
| DEFEND | 78.53% | 0.678 | 0.937 | 55.14% |
| INTERCEPT | 3.65% | 6.710 | 5.335 | 1.88% |
| RECHARGE | 67.52% | 0.967 | 1.583 | 46.72% |

Q1更新：**不平衡/梯度曝光是原先EXPLORE零预测的重要贡献因素**。在原架构不变时，平衡曝光已使EXPLORE召回持续非零，因此“原架构完全不能表示EXPLORE”缺少支持；同时precision和Goal仍差，不能宣称主要问题已解决，也不能排除容量/标签/优化与部署memory。
平衡采样同时改变mode和goal损失的类别曝光，INTERCEPT训练曝光也从normal的68,640次降到19,222次。因此整体Goal下降不能独立归因于空间编码器或某一种loss。
训练类别先验约均匀，验证保留自然先验；这是误报上升的一个合理解释，尚未做校准/先验修正对照，不认定它为唯一原因。本轮不新增此类调参。
原始标签为启发式真值，所以这些指标只度量对标签的一致性，不能直接证明真实战略探索能力或线上收益。第4轮EXPLORE recall峰值67.49%不是预注册最佳模型，不据此换checkpoint。
E2结论为指标取舍而非统一胜出：weighted的mode Macro-F1最高，balanced的balanced accuracy/EXPLORE recall最高，normal与weighted的整体Joint几乎相同且高于balanced。
下一项严格相对E1_clean：**E3_res6，普通损失、自然采样，仅换6×64ch、stride1空间主干**。容量实验不叠加E2采样变化，最终组合留待后续E6。仍为NOT READY。
Balanced耗时16.75分钟，peak allocated=118.64MiB、reserved=132.00MiB；达到第6轮上限，非patience早停。MLflow=a30d0ced75194b9ab158b8d83888678f。


## 5. 性能瓶颈与物理开销

normal耗时16.01分钟，weighted耗时16.83分钟；weighted峰值allocated=118.64MiB、reserved=132.00MiB。
两组都达到epoch6预算上限，非patience早停，也没有OOM。CPU峰值/能耗[待补充：未采集]。
weighted MLflow=a0bc7250a10d4265924625418bba2b21；数据manifest=8a8fbfb269daaa33226d95cfb7c94bcaf692b0445b5aa0c8a18fdc63fde24f13。本报告只读取现有产物，未训练、推理或执行比赛。
