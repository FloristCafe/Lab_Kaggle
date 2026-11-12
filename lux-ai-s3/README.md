# Lux AI Season 3：分层强化学习科研复盘与复现框架

研究问题：在部分可观测、跨五小局保留地图知识的团队博弈中，宏观Goal策略能否通过模仿学习和MAPPO获得稳定的官方收益？

本项目保留完整的科研主线，而不是仅保留网络结构：官方作战环境、V2基线、因果特征、Goal执行链、专家数据审计、BC、MAPPO、恢复、配对评估以及有来源的结果。大量重复实验Runner、失败修复脚本、旧Waypoint策略和中间产物不放回主项目。

**真实状态：已有框架与受控实验，不是已战胜V2的高竞争力成品。A/B/C仍是历史轻量原型，不等于先前设想的完整高上限架构。**

## 先看这四处

1. [研究问题、实验迭代与结论](docs/research_story.md)：为什么做、如何控制变量、哪些结果支持/不支持假设。
2. [环境、执行器与学习数据流](docs/architecture_and_environment.md)：从公开观察到官方收益的真实调用关系。
3. [数据与教师监督边界](docs/data_and_teachers.md)：合法历史信息、长期Goal标签与primitive动作标签的区别。
4. [复现与资产说明](docs/reproduction.md)：无环境核验、原始资产位置、历史CLI和严格的复现边界。

## 代码结构

```text
lux-ai-s3/
├── vendor/lux-design-s3/       # 固定官方环境源码、revision、license
├── agents/v2/                # 原生启发式基线
├── agents/mappo/             # 因果物理、Actor/Critic、Manager、Controller、Router桥接
├── training/                 # 数据构建、BC、PPO、checkpoint与统一评估
├── scripts/                  # 官方CPU环境及外部专家IPC worker
├── baselines/frozen_v5_2/    # 冻结B0/FULL核心依赖快照，不恢复失败实验入口
├── tests/                    # 无环境单测
├── tools/                    # 预检、数据/结果核验、本地路径登记
├── docs/                     # 科研说明与复现说明
└── reports/                  # 精选冻结证据、逐局结果摘录、模型索引
```

## 无需训练即可检查

在项目根目录的WSL Bash中：

```bash
python -B tools/recompute_results.py
python -B -m pytest -q -p no:cacheprovider tests
```

第一条仅用标准库核对证据哈希、地图配对点估计、BC选模和1200局官方胜数记录；第二条执行张量、物理、合法支持与提交恢复测试。均不启动官方环境或训练。

## 已保留的主要证据

- G05：FULL Goal替换相对旧B0，固定三模型新确认开发地图等权改善+1.792，地图描述区间[+1.375,+2.194]。这是相对B0，不是已经超过完整V2。
- G06：四臂消融有策略级正向方向，但checkpoint异质性明显，不能把Mode效果作可加的因果贡献。
- A01/A02、D03–D12、PGC：保留原结论和关键证据；未支持或证据不足的路线不包装成成功，也不恢复其大量尝试代码。
- 新BC：三架构各8轮，按Calibration选择A/B/C的第7/7/6轮。
- 新RL：仅A、种子1313969260完成1200局606000步；训练混合对手平均分差-2.41。B/C正式RL未执行，A最终开发集评估也未执行。训练平均分差不是最终checkpoint的泛化指标。

## 资产与GitHub边界

只上传源码、环境、协议、紧凑证据与复算脚本。不上传venv、完整第三方仓库、大模型权重、NPZ、原始轨迹或MLflow数据库。

完整历史项目仍在仓库外归档；必要BC数据、权重和已完成A训练结果已恢复到原来的外部artifact位置，仍不进入Git。对没有公开发布的数据/权重，本仓库提供索引和SHA256，不能承诺仅clone便重现全部历史成绩。

恢复过程中不修改冻结模型、原始日志和实验协议；没有启动训练、对战或访问保护集。
