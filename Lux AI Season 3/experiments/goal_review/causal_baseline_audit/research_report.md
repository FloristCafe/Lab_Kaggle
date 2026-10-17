# 合法观测基线与目标区域难度审计

## 1. 实验假设与核心矛盾

用无需拟合的当前坐标与真实因果memory基线，检查Goal Top1是否只是近距离/驻守标签带来的表面收益。此审计不改变标签、模型或实验预算。

## 2. 数据与防穿越边界

完整145场既有验证集，两种row协议分别与E1/E3、E4/E5对齐。网络可用输入≤t，previous赋值来自<t；future labels仅用于评分。
Oracle-mode坐标诊断假设已知真mode，**只可与专家mode条件Goal指标比较**，不能当作可部署完整策略。因果previous-intent joint基线完全使用真实离线memory；previous无效时采用固定INTERCEPT/current-coordinate回退。

## 3. 特征构建与数学表达

anchor-hit=1[position_t∈expert_goal_region]；previous-hit=1[previous_goal∈region]；joint=previous-hit·1[previous_mode=expert_mode]。
uniform-hit=|region|/576为均匀随机空间目标的期望命中率，不模拟移动可达性，也不替代物理控制基线。Region area与目标距离按有效单位帧加权，不按segment等权。

## 4. 离线评估与指标对比

| Scope | Mode | Count | Oracle-mode current anchor | Oracle-mode previous goal | Causal previous intent Joint | Mean region cells | Anchor distance |
|---|---|---:|---:|---:|---:|---:|---:|
| goal_frames | SCORE | 11,229 | 66.75% | 43.08% | 43.00% | 1.08 | 2.10 |
| goal_frames | EXPLORE | 726 | 13.22% | 0.55% | 0.55% | 1.00 | 6.24 |
| goal_frames | DEFEND | 1,779 | 79.54% | 6.58% | 0.00% | 1.00 | 1.22 |
| goal_frames | INTERCEPT | 46,834 | 2.41% | 0.34% | 0.00% | 1.00 | 12.48 |
| goal_frames | RECHARGE | 3,846 | 68.10% | 8.63% | 6.71% | 1.00 | 0.95 |
| review_epochs | SCORE | 17,268 | 66.15% | 41.39% | 41.30% | 1.07 | 2.22 |
| review_epochs | EXPLORE | 1,082 | 12.29% | 0.28% | 0.18% | 1.00 | 6.29 |
| review_epochs | DEFEND | 3,059 | 81.14% | 6.70% | 0.00% | 1.00 | 1.10 |
| review_epochs | INTERCEPT | 77,568 | 2.49% | 0.32% | 0.00% | 1.00 | 12.42 |
| review_epochs | RECHARGE | 6,290 | 68.47% | 8.76% | 6.85% | 1.00 | 0.94 |

聚合与混淆矩阵见summary.json。当前位置命中不一定是垃圾标签：SCORE/DEFEND/RECHARGE的长期持守可合法要求留在原地；不能把这些语义与EXPLORE/INTERCEPT混为同一个identity-collapse判断。
本审计没有人工意图真值，不能凭低模型Top1判定伪标签正确或错误，也不使用这些验证结果改变本轮标签。正式阶段仍需结合每类定位、模式选择、段一致性和短闭环。

## 5. 性能瓶颈与物理开销

仅流式读取现有压缩标签/单位状态，不读取隐藏引擎状态，不展开整库特征到磁盘；训练与比赛更新数均为0。数据和程序哈希、MLflow run ID保存在summary.json。
CPU峰值/能耗[待补充：本次未单独测量]。此结果属于诊断，不构成新增训练或新的holdout评估。
