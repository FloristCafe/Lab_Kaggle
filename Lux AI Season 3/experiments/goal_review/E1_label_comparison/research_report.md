# E1：Clean 与 Dirty 训练标签对照

## 1. 实验假设与核心矛盾

在同一数据、网络、memory、seed和固定优化预算下，检验标签清理能否缓解EXPLORE零召回与Goal定位失败。
**结果没有统一优势：clean的选中检查点Joint更高，dirty的Macro-F1更高；两组EXPLORE均始终零预测。** 单seed结果不足以宣称清理无效、噪声有益或网络容量不足。

## 2. 数据与防穿越边界

两组均725场，580训练/145验证；同一组64,414个clean验证单位样本、相同类别支持。网络输入来自≤t合法观测，Previous Goal赋值仅来自<t；未来轨迹只参与监督。
配置仅study.label_view不同；源码哈希、分组文件集合、验证行数和标签计数匹配。训练候选行clean=259,507、dirty=308,208，候选模式计数分别为[47522, 2888, 8119, 185307, 15671]、[53881, 2916, 12477, 221767, 17167]。
清理改变标签、分段和样本保留，故此对照估计的是**整个标签清理流程**的效应；不能把差异全归因于SAP过滤，也不是固定相同单位行仅换标签值。
两组每轮16,000单位曝光、共六轮96,000次。EXPLORE曝光clean=1,091、dirty=875；均未完整遍历所有候选六次。

## 3. 特征构建与数学表达

共用18通道×3帧ConvGRU、22维unit输入、当前空间主干与五类mode/576格goal。π(m,g)=π(m)π(g|m)，推理联合argmax。
两组普通confidence加权mode CE+专家mode条件目标区域NLL；25% previous-context dropout；无R head。clean先过滤非法SAP证据并处理删失；dirty仅在隔离训练视图保留旧标签，验证始终clean。

## 4. 离线评估与指标对比

按预注册mode-head Macro-F1选模型，Joint打破平局：clean最佳epoch=6，dirty最佳epoch=5；两组均在epoch6达到预算上限，均未发生patience早停。

| 指标 | Clean | Dirty control | Clean−Dirty |
|---|---:|---:|---:|
| Mode accuracy | 87.04% | 86.96% | +0.08 pp |
| Macro-F1 | 47.37% | 51.41% | -4.04 pp |
| Balanced accuracy | 45.78% | 48.03% | -2.25 pp |
| Joint correctness | 19.14% | 18.18% | +0.96 pp |
| Goal Top1 | 22.87% | 22.13% | +0.74 pp |
| Goal Top5 | 36.52% | 35.22% | +1.30 pp |
| Goal Top10 | 45.55% | 44.34% | +1.21 pp |
| EXPLORE recall | 0.00% | 0.00% | +0.00 pp |

Macro-F1作为百分数展示，上表clean=0.4737、dirty=0.5141；不能将其与accuracy混用。

| Mode | Clean recall | Dirty recall | Clean条件Goal Top1 | Dirty条件Goal Top1 |
|---|---:|---:|---:|---:|
| SCORE | 90.72% | 76.12% | 73.33% | 72.68% |
| EXPLORE | 0.00% | 0.00% | 9.23% | 8.95% |
| DEFEND | 8.99% | 24.96% | 79.76% | 78.08% |
| INTERCEPT | 94.79% | 96.94% | 5.21% | 4.50% |
| RECHARGE | 34.37% | 42.12% | 66.87% | 65.83% |

选择规则影响解读：dirty的epoch6 Joint=19.01%，但因Macro-F1低于epoch5，没有被选中。固定第6轮比较，clean Joint=19.14%，差值仅+0.13pp；Macro-F1分别0.4737/0.4778。不可为某一结论事后改成按Joint挑模型。
六轮训练mixed-policy统计与每轮验证的原始mode head中，两组EXPLORE预测数均0；各自最佳检查点的联合执行mode也未选择EXPLORE。说明当前配置/预算下，整个清理流程尚未解决少数类任务选择问题。INTERCEPT条件Goal Top1两组均约5%，也是独立瓶颈。
目前仅同一seed/fold各一条优化轨迹；未保存逐单位预测用于逐match配对重采样，不给虚构的置信区间。帧/单位相关，不能把64,414个标签当独立重复实验。
R head未启用，因此约95.9%的原始Replan正类F1属于always-Replan对照，不纳入已学会R的证据。尚无新增闭环完成率或胜率结果。

下一项为**E2_weighted**：使用clean标签，保持模型、预算、seed和划分不变，仅启用cap=5的class-weighted mode CE；再执行E2_balanced。观察EXPLORE是否开始被预测以及其它mode的precision/recall代价。
该顺序检验梯度分配假设；当前证据不足以跳到容量重构或E6，继续保持NOT READY，不启动正式MAPPO。[待补充：E2两组summary.json]。

## 5. 性能瓶颈与物理开销

Clean耗时16.01分钟；Dirty耗时16.64分钟。两组peak allocated=118.64/118.64MiB，peak reserved均132MiB。停止由轮数预算决定，未发生OOM。
两组总训练曝光192,000；验证每轮均64,414单位，最后另重载最佳检查点完整验证。CPU峰值/能耗[待补充：未采集]。
MLflow clean=df87831230cf4396b578a9fe74b907e4；dirty=6b62738c59cc48fb8b455e1ca5acefd4；数据manifest=8a8fbfb269daaa33226d95cfb7c94bcaf692b0445b5aa0c8a18fdc63fde24f13。原始训练产物保持不变，比较程序只读取现有指标，没有训练、推理或赛程。
