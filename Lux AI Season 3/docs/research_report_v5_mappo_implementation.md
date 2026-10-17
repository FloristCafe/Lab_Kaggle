# IL 初始化的联合 Semi-MDP MAPPO：v5 实施报告

## 1. 实验假设与核心矛盾

**工程结论：READY FOR USER PILOT；1000局正式训练尚未批准。** 本轮检验的是训练系统正确性，没有检验MAPPO是否改善战略。此前IL报告的NOT READY属于行为/科学证据缺失，新任务授权实现RL pilot，并不使旧失败指标自动变好。
新增独立模块、入口与配置；旧IL、旧MAPPO入口、V2、官方环境及数据生产器均保留。没有执行100局pilot、最终50种子赛程或新IL搜索。

### 实际审计与最小实现计划

| 审计对象 | 发现 | 实施 |
|---|---|---|
| 交付源码、checkpoint、报告 | 活跃goal_review/strategic源码与交付快照逐文件hash一致，E6 checkpoint与快照一致；HEAD不足以代表未提交源码 | 基于实际快照，不迁移旧StrategicActor权重形状 |
| 当前Actor | E6为dilated6、18通道×3帧ConvGRU32、unit22→query32，R/Mode/Goal输出与报告一致 | 精确加载原Actor；原始R与中性R各自独立配置 |
| 原采样器 | Continue仍抽取随后丢弃的MG，缺少实际分支/条件支持记录 | 新conditional sampler，仅在真实Replan时抽MG |
| 旧MAPPO入口 | 只支持二输出Mode/Goal，未接R、逐单位review与分支概率；已有duration GAE可复用 | 复用GAE数学核心，新增R-aware Buffer/PPO/入口 |
| 旧环境桥接 | 合并terminated/truncated、包含隔离的隐藏状态诊断与counterfactual | 新worker仅传合法EnvObs、官方reward和原始终止标记，无隐藏状态诊断 |
| GoalManager | 旧实现团队review所有单位，energy跳变可能误判重生；没有当前目标冲突支持掩码 | 新TeamMemory复用identity-age语义，逐单位复审、公开图保守可达性检查、顺序目标预留 |
| Critic / 恢复 | 旧Critic状态较少，旧入口未完整恢复进度/RNG | 独立初始化Critic、完整大局边界恢复Actor/Critic/Adam/RNG/进度 |

文件划分：agents/mappo_v5/policy.py负责概率与AR掩码，environment.py负责公开SMDP与事件，learning.py负责Critic/Buffer/PPO；scripts/train_mappo_v5.py为独立入口，check_mappo_v5.py为限时软件检查。配置与只读分析入口独立保存。

## 2. 数据与防穿越边界

复用IL：`experiments/goal_review/E6_candidate_seed42/best_model.pth`，SHA256=`4a356ca463c5beee4acb4949cf93e8b6b453634262167025cd54043d88b2013b`。
原始Actor参数hash=`1c895422bfbf8e6265f24b79d666486df94c1d0b8c579e4ee0443510c00bf250`；中性化后=`adfce434c08fc949e1dd532fb707302dad2520b21209665d17d4a395f518797b`。仅replan_head.weight/bias被置零，Encoder/Query/Mode/Goal逐张量保持。原始checkpoint没有覆写。
IL数据725场、580/145隔离的事实不变，但RL不读取专家label、未来位置或过滤掩码；在线特征截止≤T，previous intent来自<T的真实历史选择。R诊断只读取一个既有录像24帧的合法输入，不是验证性能。IL生产器8项hash全部一致，无需重新生成NPZ。
训练seed42的环境seeds为40042000–40042099，双方交替；与IL source seeds和dev不相交。E1/E2使用同一序列。dev seeds17001–17004、双方交换，每checkpoint8局；17001曾用于历史dev，不能称独立最终测试。未调用最终holdout赛程，其他训练seed复现留待pilot通过后。
固定对手为现有V2；也支持现有v0，主配置不混入额外对手。Actor/对手均仅用各自合法观测，worker不输出隐藏地图或随机物理参数。对手所有V2模块的hash写入manifest。
执行环境：Windows仅文件编辑/VSCode；计算、热数据、checkpoint、SQLite均在WSL ext4。训练解释器为ml-workspace venv，官方JAX worker用项目venv，固定官方commit `c6d1e665d12a467b299fb586b3f876e47847746b`。

## 3. 特征构建与数学表达

### Actor、实际分支与目标冲突

π(a_i)=π_R(r_i)[π_M(m_i)π_G(g_i|m_i,先前目标预留)]^r_i。训练随机采样R→Mode→Goal；greedy评估使用R阈值与masked联合(m,g) argmax，评估密度不进入PPO。
按固定UID顺序处理需要复审的单位；未复审单位完整保持原intent，其目标继续预留。已有有效目标也预留，轮到本单位时才释放自己的预留，避免抢占尚未处理单位的旧目标。目标支持排除当前sensor可确认的小行星及已预留目标，未观测地形保持可能通行；过期地形belief不能单独充当不可达证明，不使用未来图或按瞬时能量硬截断长期目标。
保存decision_mask、forced_action_mask、branch_mask[R,M,G]、R/Mode/Goal action masks、顺序、动作、old_logprob及三个old logprob因子。重算直接使用保存的条件支持，不能在更新后重新生成old policy。
Continue只采样R，保留原mode/goal/age/blocked/reached；没有MG随机数或MG策略项。没有有效previous、公开证伪/已知不可达、生命周期重置或继承冲突时，首版采用**整个动作由确定性masked joint argmax强制分配**：所有branch mask为0、logprob=0、无Actor PPO。这比“只强制R、仍随机抽MG”更保守，后者不是本版策略。
确定性Gate：无效时强制，其他review单位在已抵达、持续阻塞或目标年龄≥16时Replan；Gate不训练R，只有真正自由采样的MG计入PPO。普通路由更新不触发宏目标变化。每单位独立review时钟上限8，其他单位出生/复审不会让它无限延后；通常soft review最小间隔3、严重威胁需持续3步。
公开图可达性把未知当可通行，仅已知封闭才判不可达，动态地形可能随后改变。与旧IL执行器相比，这些事件/掩码变化会改变行为，必须重测匹配E0，不能拿旧前缀数字充当新baseline。

### R初始化审计

原始自由样本N=208，R logit分位数(min/p10/p50/p90/max)=[2.2883074283599854, 4.999128818511963, 6.921884536743164, 7.698383808135986, 8.941957473754883]；平均Replan概率=99.7172%，greedy Replan=100.00%。
分previous mode的支持和输出见smoke summary；本样本未覆盖SCORE/DEFEND/INTERCEPT previous mode，不将缺失条件填成结果。强制R与自由R分开统计，不能把硬规则产生的r1当网络偏好。
中性化使自由R logit=0、随机概率=0.5；greedy的0.5平局仍按Replan处理，所以不能由greedy不变断言中性化无效。R LR=1e-4仅用于中性初始化；原始R对照用3e-5，因此原始/中性对照同时变化初始化与LR，不能纯归因初始化。

### Centralized Critic

独立seed+1000初始化HistoryEncoder、12张全队context maps、全体单位30维set聚合(mean/max)、12维team context，无大型Transformer。输入包括合法地图/可见敌军/V2 belief、所有己方位置与能量、previous mode/goal/age/valid、目标占位/冲突、五种战术状态、官方比分与获胜小局数、剩余时间/小局索引、信息熵/完成计数。缺失真实冷却、隐藏敌军/全图真实状态不会伪造。
Actor与Critic不共享参数；Adam参数按Encoder/Query、Mode/Goal、R、Critic分组，ID检查防重复优化。Continue-only minibatch将MG grad设为None，连Adam已有动量也跳过，防止没有采样MG仍改变其输出层。

### Semi-MDP Buffer / PPO

CPU transition保存state与next_state（含原始previous intent七字段）、实际action、所有mask/顺序、old概率、团队value/next value、宏reward、τ、episode ID、任务terminal/外部truncated和原始引擎标记/primitive rewards。
R_k=Σ(j=0..τ-1)γ^j r_j；δ=R_k+γ^τ(1-terminal)V(next)-V；A_k=δ+(γλ)^τ(1-terminal)trace_link A_(k+1)。episode切换、外部截断和buffer尾部切断trace；外部截断保留next-value bootstrap，完整大局末不bootstrap。
共享团队优势，逐单位exp(newLP-savedOldLP)与clip=.1，仅真实Actor分支参与；先在任一更新前核验全buffer误差≤1e-4，然后用保存oldLP进行所有epoch。每团队边界内部按有效Actor单位平均，再对有有效动作的边界平均。记录KL、clip、实际分支熵、raw/normalized优势、各分支梯度与Critic固定target更新前后MSE/EV。
熵为实际采样分支的条件熵正则，不宣称完整联合分布熵；它不会在Continue样本给MG添加梯度。
γ=.99，λ=.95，PPO 4epochs，buffer32，minibatch4；LR为Encoder/Query1e-5、MG3e-5、neutral R1e-4、Critic1e-4；全局梯度clip .5。FP32 rollout/update、TF32关闭，以保持跨batch形状概率一致。

### 官方奖励与跨小局

官方step返回累计team_wins，直接逐步求和会重复奖励已有领先。本版明确使用其**终局差值**r=own official wins-opponent official wins，其他步为0；这是终局官方目标的零和表示，不是逐步原样累加累计值。没有点数、击杀、探索、能量或完成塑形。若pilot回报过稀疏，先分析，不自动加塑形。
官方is_terminal始终False，在505步返回engine_truncated=True；505步是这个有限五小局任务的完整终点，映射task terminal=True并停止bootstrap。人工短截断是truncated=True、terminal=False，保留bootstrap。
每小局边界清单位intent、战术状态与三帧临时history；CausalObservationBuilder/Reconstructor、terrain/energy记忆、漂移估计与遗迹/方程belief对象保留。小局获胜计数保留，直到505步统一产生终局奖励；不会误当五个独立episode。

## 4. 离线评估与指标对比

本轮所有数字仅为软件证据，不能当RL胜率、Critic游戏价值学习或MAPPO优于IL的结论。

| 配置短检查 | 实际primitive步 | PPO optimizer steps | 更新前LP最大误差 | checkpoint往返 |
|---|---:|---:|---:|---|
| E0_frozen | 18 | 0 | — | True |
| E1_GM_pilot | 18 | 1 | 4.76837158203125e-07 | True |
| E2_RMG_pilot | 18 | 1 | 9.5367431640625e-07 | True |

联合梯度fixture明确使用手设测试reward，验证同一次E2更新能激活各分支：R梯度范数=1.854533、Mode=0.659365、Goal=0.656188；这是软件fixture，不能当真实回报收益。真实短前缀奖励均0，Critic MSE有限不代表学到游戏价值。
本次完整测试=154项（含新增21项MAPPO单元测试）；另用项目venv通过8项parser测试和4个官方物理fixture。静态检查与diff检查通过。限时GPU/环境check耗时8.60s，MLflow=`9e51d08ca3634c02869a3f63f9a32df8`；checkpoint恢复同时核对Actor、Critic、Adam tensor/参数组与CPU/CUDA/NumPy RNG，真实compressed journal的old概率往返一致。
真实环境检查跨过一次小局边界（103个noop primitive steps），不是整场大局；同一reconstructor、history清零、intent清零且task未终止。该noop前缀没有发现遗迹，因此另有非空合法观测fixture验证已见遗迹/能量与solver/drift对象跨局保留，避免空集合兼容性假阳性。
初次smoke最终JSON写入因NumPy战术计数不可序列化失败，已显式转换并保留失败日志；后续检查通过，没有隐藏失败或覆盖旧IL报告。

### E0/E1/E2独立配置

| Config | 完整训练局 | Gate | R初始化 | R LR |
|---|---:|---|---|---:|
| E0_frozen | 0 | learned | original | 3e-05 |
| E0_gate | 0 | deterministic | original | 3e-05 |
| E0_neutral | 0 | learned | neutral | 0.0001 |
| E1_GM_pilot | 100 | deterministic | original | 3e-05 |
| E2_RMG_pilot | 100 | learned | neutral | 0.0001 |
| E2_original_pilot | 100 | learned | original | 3e-05 |

所有配置从不可变common载入为独立对象，resolved快照每run保存。common继承IL网络/战术配置；继承的旧il/rl/study字段只作lineage，新入口不使用旧训练超参数。E1/E2环境、奖励、Critic、PPO设置一致，仅策略Gate/初始化的预设treatment不同；R虽然存在于E1 Actor，优化器不包含它。
可复制命令、顺序、日志、输出和恢复示例见[执行手册](runbook.md)。先E0原始与匹配Gate，再E1、E2，中性/原始R对照已单独给出。训练结束用report_mappo_v5_pilot.py只读比对，工程PASS不自动升级科研READY。

### 仍需pilot回答的问题

100个完整大局尚未执行，实际终局回报分布、Critic学习、Mode/Goal长期收益、自由R是否再坍缩、完整比赛吞吐均未测。forced整动作排除了这部分Actor梯度，必须检查其占比；如果占绝大多数，需要单独评估替代分解，不能默改主实验。公开观测不提供生命实例ID，同帧同坐标重生可能无法辨识；采用可见死亡/位置跳变/小局重置的保守生命周期证据，不读取隐藏单位状态。
pilot须无LP/时间/终止错误，loss与梯度有限、无目标冲突；同时检查R/Mode坍缩、停滞、非平凡完成和Critic target误差。E1须与匹配Gate Frozen比较，E2须与E1比较；E2未超过E1不能声称学习R有效。4个dev seed、单训练seed仅供pilot，未做统计强度证明。
通过上述科研复核后才配置约1000局及多个训练seed；当前入口硬拒绝games>100，本次没有提供或启动正式1000局配置。

## 5. 性能瓶颈与物理开销

模型+Adam约十余MiB数量级，CPU buffer32含双状态、goal mask和概率约12MiB；FP32 minibatch4激活预计保守低于1GiB。实测smoke peak allocated=140.26MiB、reserved=150MiB；显存为8150.56MiB。分段memory guard会重置峰值，应结合各段日志，不能把最后探针当系统整卡绝对峰值。实际100局时长/主机峰值/能耗尚未采集。
新worker无counterfactual，CPU只跑官方环境与固定合法对手；频繁物理步和时空路由可能主导耗时。原始录像/NPZ不重复写入；每个update的全部真实transition用np.savez_compressed无损保存，goal mask按位打包，Actor输入保留原float32，old概率不被新量化扰动。每run约2GiB rollout预算、4GiB空闲保留线（按chunk检查），中断旧journal也计入；达到线停止而不填满磁盘。buffer持久化后释放，另留首末局小份torch样本；评估记录逐决策greedy trace。每完整游戏边界原子更新last.pth，MLflow只存紧凑lineage和指标，避免重复上传整套权重。
恢复仅保证完整游戏边界。中途崩溃会从上一完整游戏重来；manifest拒绝变更配置、源码、IL hash或seed，保留各attempt日志。Actor、Critic、Adam、RNG、progress都保存；不序列化官方隐藏环境状态，不承诺mid-option恢复。
用户拥有后续执行权：本轮Agent只运行短软件检查，没有启动长训练、任何pilot或自动接续。原始IL、V2、官方vendor和既有未提交改动全部保留。
