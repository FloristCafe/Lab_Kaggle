# Goal-Level IL：标签、容量、Replan 与部署记忆受控研究

## 1. 实验假设与核心矛盾

本轮分别检验严格标签、类别梯度、空间主干和意图记忆是否改善 Goal IL，再按事前登记规则组合候选。**结论：NOT READY；E6 组合未解决 EXPLORE 与持续意图缺失，暂不进入正式 MAPPO。**
已完成 11 个正式 IL run（同 seed42），4 个完整145场冻结记忆研究、14个双侧/η组合的实际短前缀；没有增加下载、最终留出集或正式RL训练。后续实验由用户启动。

## 2. 数据与防穿越边界

725场胜方合法视角，580训练/145验证；按Match ID划分，已知source seeds均唯一且跨集合不相交，旧80验证ID保持验证。4,202,800存活单位状态，1,293,473有效Goal，过滤69.22%；有效类计数={'SCORE': 234467, 'EXPLORE': 14407, 'DEFEND': 39451, 'INTERCEPT': 927745, 'RECHARGE': 77403}。EXPLORE仅1.11%，INTERCEPT71.73%。过滤状态没有被默认标成Continue，实际恢复filtered Continue=0。
输入为≤T的公开观测，previous intent只来自<T的规则/策略输出；标签可使用>T的专家轨迹推导语义段、区域和纠错证据，不能回流输入。事件分段前剔除8,021次非法SAP，合法148,151次；非法SAP派生宏事件审计为0。死亡、截尾、证据不足不伪造Goal。
数据、标签规则、memory生产器的8项哈希一致，manifest=8a8fbfb269daaa33226d95cfb7c94bcaf692b0445b5aa0c8a18fdc63fde24f13。标签仍是启发式推断，没有人工意图真值；source seed隔离也不保证地图结构/目标分布完全新颖。
E1/E2/E3验证64,414单位（每4帧采样）；E4/E5/E6验证105,267单位（review epochs），两协议的绝对差值不能归因于单项改动。E5/E6 corrupted配置同时改变训练与选模validation的记忆上下文，不能将原始summary之间的差异解释为纯训练corruption因果效应。以下共用FP32冻结评估缓解评估条件差异，但不消除选模条件差异。
同一145场验证被反复用于选epoch、选组件、选η，没有独立测试、种子复现或显著性推断。E6是验证选择的探索性组合。

## 3. 特征构建与数学表达

每帧18个因果物理通道，三帧分组编码与ConvGRU(hidden32)融合；单位22维状态形成query32。输出 mode[B,16,5]、goal[B,16,5,24,24]及独立Bernoulli r。π(r,m,g|s,Iprev)：r=0保持原(mode,goal)，r=1联合选择新(mode,goal)；无有效previous强制r1。Continue logπ仅包含logP(r0)，Replan额外包含mode和goal概率。没有577类KEEP。
训练为confidence-weighted mode CE + expert-mode区域NLL +自由previous上的R BCE；r0样本不为新mode/goal贡献梯度。三项E2仅改变mode梯度分配/单位采样，loss权重上限5。E3保持普通loss，只替换空间主干；移除下采样也改变全局汇聚路径，不能等同纯参数量实验。

| Architecture | Parameters | Conv modules | Widths | Downsample | R80 / R95 |
| --- | --- | --- | --- | --- | --- |
| current | 185809 | 12 | [5, 8, 32, 48, 64] | 2 | 19.00 / 33.00 |
| res6 | 528977 | 21 | [5, 8, 32, 64] | 0 | 13.00 / 29.00 |
| res8 | 2471569 | 25 | [5, 8, 32, 64, 128] | 0 | 11.00 / 24.00 |
| dilated6 | 528977 | 21 | [5, 8, 32, 64] | 0 | 26.00 / 34.00 |
| current_trained | 185809 | 12 | [5, 8, 32, 48, 64] | 2 | 28.00 / 38.00 |

current为stem+两次下采样/上采样；res6/res8各6×64/8×128残差块；dilated6为6×64、d=[1,2,3,5,8,4]。GroupNorm和current全局均值使理论信息支持可全图；不能由局部卷积公式断言模型看不到对角。梯度半径为单状态局部敏感性，除current_trained外为初始化权重诊断，不能当成已训练容量胜负。
bounded hysteresis仅对上一个(mode,goal)联合对加η≤0.5，80步衰减为0、100步使memory失效。η不改变训练loss/梯度，因此E5_hysteresis与combination按同checkpoint的η=0/.25推理对照完成，没有重复训练相同目标。
执行器修正：同目标r1不再重置age、blocked、reached；新目标/失效目标仍正常初始化。冻结scripted规则只在目标改变/失效时r1，故该修正不改变已生成特征和标签。133项单元测试、真实前缀特征一致性、240帧逐流/分批一致性通过。

## 4. 离线评估与指标对比

### 正式受控训练

| Arm | scope / memory | Best | Mode Acc | Macro-F1 | Balanced Acc | Joint | EXP recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1_clean | goal_frames / scripted | 6 | 87.04% | 0.4737 | 45.78% | 19.14% | 0.00% |
| E1_dirty_control | goal_frames / scripted | 5 | 86.96% | 0.5141 | 48.03% | 18.18% | 0.00% |
| E2_weighted | goal_frames / scripted | 6 | 83.35% | 0.5281 | 57.27% | 19.14% | 0.55% |
| E2_balanced | goal_frames / scripted | 6 | 75.28% | 0.5223 | 65.88% | 17.85% | 43.25% |
| E3_res6 | goal_frames / scripted | 6 | 87.01% | 0.4739 | 45.69% | 18.72% | 0.00% |
| E3_res8 | goal_frames / scripted | 2 | 86.45% | 0.4659 | 46.46% | 17.89% | 0.00% |
| E3_dilated | goal_frames / scripted | 6 | 87.39% | 0.4678 | 45.54% | 19.15% | 0.00% |
| E4_without_r | review_epochs / scripted | 3 | 86.91% | 0.4991 | 47.73% | 18.31% | 0.00% |
| E4_with_r | review_epochs / scripted | 6 | 86.79% | 0.5065 | 49.69% | 16.62% | 0.00% |
| E5_corrupted | review_epochs / corrupted | 5 | 87.02% | 0.5105 | 50.95% | 17.55% | 0.00% |
| E6_candidate | review_epochs / corrupted | 4 | 82.88% | 0.5104 | 55.10% | 17.92% | 0.00% |

选模=mode-head Macro-F1，joint只破平局；每轮1,000更新×16单位，最多6轮、patience5。所有run跑到6轮预算上限，这不是充分收敛证据。每个run96,000单位曝光，远少于全库129万有效状态。mode-head argmax与执行joint argmax的mode不同，分别记录。
严格clean未获得对dirty的Macro-F1优势（0.4737 vs0.5141），joint略高（19.14% vs18.18%）；清理同时改变样本和分段，且无人工语义真值，不能宣称清理已验证正确。dirty仅隔离对照，不进入候选。
E2 weighted提高Macro-F1到0.5281，joint仅净增2/64,414；balanced把EXPLORE召回从0推到43.25%，但precision5.94%、joint降到17.85%。不平衡对模式塌缩有直接干预证据，重采样仍未解决目标质量。
E3 dilated与current的joint差仅7/64,414（19.148% vs19.137%）；res6/res8更差，全部EXPLORE召回为0。事前规则机械选dilated为E6组件，不是统计上可靠的架构胜出。

### Goal细定位（给定专家mode）

| Arm | Goal@1 | @5 | @10 | L1 | Entropy | Pred distance | Expert distance | Segment consistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_clean | 22.87% | 36.52% | 45.55% | 4.699 | 3.884 | 8.34 | 9.60 | 43.41% |
| E1_dirty_control | 22.13% | 35.22% | 44.34% | 4.790 | 3.665 | 8.81 | 9.60 | 44.77% |
| E2_weighted | 22.46% | 35.76% | 44.87% | 4.740 | 3.936 | 7.49 | 9.60 | 45.05% |
| E2_balanced | 21.64% | 33.57% | 42.16% | 5.201 | 4.250 | 5.72 | 9.60 | 43.74% |
| E3_res6 | 22.37% | 35.41% | 44.22% | 4.897 | 3.797 | 8.59 | 9.60 | 42.37% |
| E3_res8 | 21.81% | 33.98% | 42.34% | 5.452 | 4.088 | 6.41 | 9.60 | 42.25% |
| E3_dilated | 22.80% | 36.66% | 45.82% | 4.698 | 3.802 | 8.82 | 9.60 | 43.16% |
| E4_without_r | 21.98% | 35.47% | 44.54% | 4.886 | 3.902 | 7.30 | 9.67 | 48.70% |
| E4_with_r | 22.25% | 36.29% | 45.73% | 4.542 | 3.979 | 8.11 | 9.67 | 49.62% |
| E5_corrupted | 22.33% | 36.23% | 45.47% | 4.756 | 3.875 | 6.64 | 9.67 | 49.33% |
| E6_candidate | 21.74% | 35.82% | 45.08% | 4.672 | 3.962 | 6.97 | 9.67 | 50.78% |

Goal TopK按真实模式的条件目标头评分，Joint同时要求执行模式与目标区域正确，不能混为同一Accuracy。动态未知地图geodesic未定义，保持null。段内一致性/离线persistence来自采样时点，不能冒充真实闭环保持。
同协议因果基线：给定专家mode时当前坐标命中19.80%，E1 Goal@1=22.87%；previous目标8.46%，因果previous完整意图joint7.90%。当前坐标是有效SCORE/DEFEND/RECHARGE常见目标，不应全部当垃圾；该oracle-mode基线不可直接和模型joint比较。平均目标区域仅1.013格，除SCORE外四类均单格，INTERCEPT专家目标距当前均值12.48格，其Goal@1仍仅约5%，精确单格伪标签可识别性是待解决瓶颈。

### 冻结同权重的A/B/C记忆研究

所有行共有257,305个有效单位状态。review时发出意图，其他帧保持，在全部clean物理帧计分；分母不随策略review改变。A=因果scripted，B=标签盲corrupted快照，C=Actor历史输出回流memory。A/B输出保持仅用于计分，输入memory各自固定。C仍沿用专家物理轨迹，因此不是环境闭环。

| Arm | η | Memory | Held Joint | review/frame | Continue recall | False Replan |
| --- | --- | --- | --- | --- | --- | --- |
| without_r | 0.0 | scripted | 18.01% | 0.390 | — | — |
| without_r | 0.0 | corrupted | 17.86% | 0.390 | — | — |
| without_r | 0.0 | policy_replay | 16.97% | 0.404 | — | — |
| with_r | 0.0 | scripted | 16.22% | 0.390 | 79.49% | 20.51% |
| with_r | 0.0 | corrupted | 15.24% | 0.390 | 80.25% | 19.75% |
| with_r | 0.0 | policy_replay | 12.51% | 0.401 | 44.87% | 55.13% |
| with_r | 0.25 | scripted | 16.24% | 0.390 | 79.49% | 20.51% |
| with_r | 0.25 | corrupted | 15.25% | 0.390 | 80.25% | 19.75% |
| with_r | 0.25 | policy_replay | 12.08% | 0.401 | 43.70% | 56.30% |
| corrupted | 0.0 | scripted | 17.84% | 0.390 | 28.71% | 71.29% |
| corrupted | 0.0 | corrupted | 17.29% | 0.390 | 35.70% | 64.30% |
| corrupted | 0.0 | policy_replay | 16.77% | 0.403 | 14.94% | 85.06% |
| corrupted | 0.25 | scripted | 17.88% | 0.390 | 28.71% | 71.29% |
| corrupted | 0.25 | corrupted | 17.31% | 0.390 | 35.70% | 64.30% |
| corrupted | 0.25 | policy_replay | 16.69% | 0.403 | 15.26% | 84.74% |
| candidate | 0.0 | scripted | 17.80% | 0.390 | 0.00% | 100.00% |
| candidate | 0.0 | corrupted | 17.75% | 0.390 | 0.00% | 100.00% |
| candidate | 0.0 | policy_replay | 17.16% | 0.398 | 0.00% | 100.00% |
| candidate | 0.25 | scripted | 17.91% | 0.390 | 0.00% | 100.00% |
| candidate | 0.25 | corrupted | 17.81% | 0.390 | 0.00% | 100.00% |
| candidate | 0.25 | policy_replay | 17.07% | 0.397 | 0.00% | 100.00% |

E4配对加R后，规则条件Continue召回79.49%、False Replan20.51%，但固定帧joint从18.01%降到16.22%；C进一步从without-R16.97%降到with-R12.51%。R分类学会了一部分规则兼容性，未解决部署错配。
E5 corruption使C joint回升到16.77%，仍未超过without-R16.97%，且C Continue召回仅14.94%。A→C退化仍存在，不能宣称解决exposure bias。η=.25对A/B有微小帮助，对C普遍恶化；E6事前选择η=0。E6本身η=.25的C也更差（17.07% vs17.16%）。
E6选择=dilated6、class_weighted、E5_corrupted、R=True、η=0.0。其C fixed-frame joint17.16%略高without-R16.97%，同时**最佳checkpoint自由R完全始终Replan、Continue召回0**，不能作为最终成功候选；组件优点未必可叠加。

### E6逐类结果与误差

| Mode | Support | Predicted | Precision | Recall | F1 | Goal@1 | Joint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCORE | 17268 | 23445 | 68.20% | 92.60% | 0.7855 | 70.31% | 67.63% |
| EXPLORE | 1082 | 0 | 0.00% | 0.00% | 0.0000 | 9.24% | 0.00% |
| DEFEND | 3059 | 2926 | 34.83% | 33.31% | 0.3405 | 79.34% | 39.03% |
| INTERCEPT | 77568 | 69870 | 94.74% | 85.34% | 0.8980 | 5.10% | 3.51% |
| RECHARGE | 6290 | 9026 | 44.78% | 64.26% | 0.5278 | 67.85% | 52.00% |

原始选模105,267单位下mode82.88%、Macro-F1 .5104、balanced55.10%、joint17.92%；EXPLORE head predicted=0，执行joint模式会少量选EXPLORE但其正确目标joint仍0。完整11组逐类、混淆矩阵、R矩阵见per_mode.csv和confusion_matrices.md；冻结条件全量JSON保留。

### 实际dev短闭环（seed15001）

| Arm | η | Side | Steps | review/frame | Free Continue/N | Goal persist | Completion | score/opponent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| without_r | 0.0 | 0 | 129 | 0.287 | 0/316 | 37.03% | 0.92% | 0/31 |
| without_r | 0.0 | 1 | 129 | 0.326 | 0/412 | 67.48% | 0.45% | 0/213 |
| with_r | 0.0 | 0 | 129 | 0.318 | 0/396 | 34.34% | 1.62% | 0/269 |
| with_r | 0.0 | 1 | 129 | 0.287 | 0/320 | 15.62% | 0.92% | 0/0 |
| with_r | 0.25 | 0 | 129 | 0.310 | 0/380 | 50.26% | 1.02% | 0/206 |
| with_r | 0.25 | 1 | 129 | 0.295 | 0/344 | 43.02% | 2.57% | 0/0 |
| corrupted | 0.0 | 0 | 129 | 0.279 | 0/311 | 54.98% | 0.56% | 0/0 |
| corrupted | 0.0 | 1 | 129 | 0.341 | 0/438 | 65.07% | 1.68% | 0/202 |
| corrupted | 0.25 | 0 | 129 | 0.318 | 0/396 | 76.01% | 2.42% | 0/274 |
| corrupted | 0.25 | 1 | 129 | 0.318 | 0/388 | 72.68% | 1.60% | 0/235 |
| candidate | 0.0 | 0 | 129 | 0.310 | 0/380 | 64.47% | 0.58% | 0/207 |
| candidate | 0.0 | 1 | 129 | 0.256 | 0/268 | 43.28% | 2.17% | 0/0 |
| candidate | 0.25 | 0 | 129 | 0.256 | 0/266 | 63.53% | 0.00% | 0/0 |
| candidate | 0.25 | 1 | 129 | 0.264 | 0/272 | 60.29% | 0.55% | 0/0 |

每行预算128 primitive步，当前option结束后实际129步，均非终局；score只为前缀计数，不是整局胜率。所有14前缀的自由previous决策也没有Continue，不能归咎于全部强制r1。目标身份持久性与r0不同，r1重新选择相同坐标同样会保持目标。非原地首次抵达率通常<3%，E6η0双侧1/172和4/184；前缀短、单dev种子且已有使用，不能外推整局质量。在线没有专家R真值，precision/recall/F1=null。Invalid persistence=0由协议保证，不是学习成绩。

### Q1–Q8研究结论

| 问题 | 结论及边界 |
|---|---|
| Q1 EXPLORE=0主因 | **类别不平衡是已实证的主要贡献因素**：同网络balanced把recall推到43.25%，排除完全无法表达EXPLORE。precision仅5.94%、E6仍0，标签归因/状态混淆/记忆支持缺口仍可能重要，无人工意图真值不能确定唯一主因。 |
| Q2 spatial encoder过浅 | **未证明是主要瓶颈**：当前已有全图信息支持，放大res6/res8没有改善，dilated仅7个正确样本优势；有限预算不能证明更大架构长期无效。固定32维temporal/query、目标单格噪音仍需分离验证。 |
| Q3 filtered造成部署错配 | **错配实测存在，不能全部归因过滤率**。新过滤69.22%，恢复filtered Continue=0；规则训练previous从不DEFEND/INTERCEPT，真实Actor会产生这些mode。训练自由R0样本SCORE27578、EXPLORE11、DEFEND0、INTERCEPT0、RECHARGE1705，支持域明显不足。offline A约.390 review/frame、C约.398–.404，实际prefix约.256–.341且状态不同；差值不是过滤的单变量因果效应。 |
| Q4 R真正解决了吗 | **没有**。A下R分类改善，但完整意图joint降低，C降幅更大；真实prefix有有效previous也全r1，E6又退化为常量r1。R1高F1主要受正类占比支配。 |
| Q5 Previous Goal exposure bias | **有直接分布差异与冻结干预证据**：with-R A16.22%→C12.51%，Continue79.49%→44.87%；C含previous DEFEND/INTERCEPT而A均0。corruption缓解joint退化但未带来可靠Continue；C并非互动rollout，尚不能归因全部在线损失。 |
| Q6 accuracy86.20%误导吗 | **若理解为五类均已学会则误导**。E1 mode87.04%但Macro-F1 .4737、balanced45.78%、EXPLORE0；balanced提高少数类召回却降低joint。须同时报告逐类、两种mode定义及joint。 |
| Q7 学会长期战略吗 | **证据不足**。E6预测距离6.97vs专家9.67格、段内一致性50.78%；目标精确单格占主导，INTERCEPT Goal@1约5%，真实非平凡完成低，且全r1。不能仅凭长语义段或预测离当前较远就宣称长期规划，也不能仅凭anchor率断言恒等映射。精确相同query冲突仅3/64,414不可避免joint错误，不能解释总体失败；这不证明表示充分。 |
| Q8 正式MAPPO | **NOT READY**。阻塞项：EXPLORE精确意图仍失效、规则R0支持不足、memory反馈鲁棒性差、真实prefix无Continue且目标完成低、标签语义无人审。R-aware行为概率/SMDP接口sanity通过，但正式R-aware rollout buffer与PPO optimizer尚未接完；旧MAPPO入口不能直接接新checkpoint。 |

优先修复方向是合法、标签盲且具DEFEND/INTERCEPT持续状态覆盖的memory行为策略与兼容监督，另做分层意图盲审来评估单格标签噪音；当前结果不支持继续盲目加宽网络或直接用RL掩盖标签问题。**本报告没有启动这些后续实验；实现完成后将给用户可执行指令。**

## 5. 性能瓶颈与物理开销

| Arm | Wall min | Peak allocated MiB | Peak reserved MiB | MLflow |
| --- | --- | --- | --- | --- |
| E1_clean | 16.01 | 118.6 | 132.0 | df87831230cf4396b578a9fe74b907e4 |
| E1_dirty_control | 16.64 | 118.6 | 132.0 | 6b62738c59cc48fb8b455e1ca5acefd4 |
| E2_weighted | 16.83 | 118.6 | 132.0 | a0bc7250a10d4265924625418bba2b21 |
| E2_balanced | 16.75 | 118.6 | 132.0 | a30d0ced75194b9ab158b8d83888678f |
| E3_res6 | 18.32 | 188.6 | 214.0 | d4583d63922d429fb3255edd958a0918 |
| E3_res8 | 18.65 | 332.1 | 382.0 | 56d80564c5954190b1fa46e5e01599b6 |
| E3_dilated | 40.35 | 188.6 | 214.0 | 7e861bfe0c764ddb864a404f4cfe4a0e |
| E4_without_r | 52.54 | 113.6 | 132.0 | d96a24e3c5da4bcfb860233a86427a7b |
| E4_with_r | 53.42 | 113.6 | 132.0 | 1909834a8fea472d832bd1604a02ca07 |
| E5_corrupted | 54.38 | 113.6 | 132.0 | ab5a285dedb64149aaf6f1d72f07e001 |
| E6_candidate | 27.12 | 183.8 | 194.0 | c75c058572a74f3e8441491b80562351 |

后四组受控训练并行共享GPU，其wall不能与前六组独占运行直接做吞吐比较；E6随后独立运行。30秒设备采样仅覆盖四组并行区间：2026-10-17T05:51:58.793964+00:00至2026-10-17T06:41:35.238729+00:00，共100点，GPU总显存峰值3180MiB，温度最高75°C，host available最低5386MiB。单进程peak allocated与整卡进程外开销不同；后续评估/E6无连续资源采样，不能把该采样冒充全流程峰值，能耗未独占归因。
数据NPZ277.65MiB、JSON106.97MiB；读取LRU1 shard、按shard组织样本，无全库展开。725原始录像与旧数据保留，无删除与下载。冻结研究单条件按游戏16流分批且各流时间顺序独立，输入FP32/TF32关闭，所有完整验证145场，额外训练更新数0；对应wall和MLflow见summary.json，原始输出保留。
Git=5199df6bffd017f44c899919bb4ce766186ae11f；官方环境=c6d1e665d12a467b299fb586b3f876e47847746b；V2与vendor未修改。训练/评估checkpoint、源码、配置、数据hash均独立保存；源码未提交，不能只凭HEAD复现，必须使用交付快照。133项单元测试、静态检查与diff检查已通过，软件一致性不等同泛化质量。
此前授权自动批次已结束，当前执行归属已恢复为用户；自动接续入口检查execution_owner，当前拒绝agent继续启动。旧授权plan单独归档，完成产物不删除、不覆写、不重新标成pending。
