# 数据、教师与监督边界

## 可用资料

| 资料 | 数量 | 用法 |
|---|---:|---|
| 原始历史Replay | 725完整游戏，725地图 | 公开观察审计，不能直接把私有replay全状态当输入 |
| G06 S2 FULL | 48游戏、21042团队复审状态 | 14165/3388/3489，按16/4/4地图分组 |
| 固定Flat提交43368177 | 31游戏，3875团队帧 | 3250/375/250；六类primitive动作辅助 |

旧400场IL来源种子缺失未解决；旧12步Waypoint标签与旧权重不用于新长期Goal Actor。

## 监督语义

宏观教师明确命名为G06_FULL_S2_B0_RMode_V2_adapted_Goal：R/Mode来自冻结B0，Goal是V2提议经过原Adapter后真实被Manager接收的目标。它不是Flat或Frog的长期Goal标签，也不是原生V2完整宏观策略。

primitive专家通常不暴露R/Mode/长期Goal。逆向Controller小诊断出现不兼容或多解，没有据此制造大规模Goal监督。当前只用原始动作分类辅助Encoder，SAP落点不在该辅助损失内。

L_BC = L_R + L_Mode + L_Goal + 0.1 L_primitive。只有实际活跃分支计算对应NLL；强制R、Continue与Goal合法掩码与部署一致。

输入只使用决策时已经合法可知的观察/历史；未来轨迹可用于离线审计标签，不能回流到特征。Teacher独有衰减记忆不输入学生，因此部分教师目标未必由学生特征唯一确定。

## 教师与对手分开

已运行的对手：V2、Flat Neurons、aDg4b FN/FP、Gregor v73b。每个外部版本有双方505步准入、commit、License、权重SHA和独立CPU依赖。

Frog Parade、YumeNeko、Team K源码可审查，但本轮未取得可验收的最终权重，未加入对手池。Frog的Rust物理/记忆代码有参考价值，不等于已有可运行第二名Agent。

外部仓库不复制进GitHub项目。tools/prepare_local_assets.py只为新代码位置生成派生路径登记，不修改历史注册表、不改变权重或参考哈希。

PFSP采用Beta(1,1)平滑胜率、失败概率平方、10%探索，再以35%质量锚定V2。名单、采样RNG和统计都写入checkpoint。不同架构训练后实际对手分布会变化，不能声称始终面对完全相同的序列。

## 数据偏差

历史下载继承了胜局筛选；本轮没有按结果再次筛选。宏观标签中DEFEND占比很高、RECHARGE无样本；探索地图已被多次查看。零标签、不利地图及失败记录保留在外部原始资产里，不为追求效果删除。

数据Manifest、来源索引和标签审计见reports/evidence/expert_data_audit.json、expert_provenance.json及reports/model_index.json。完整JSON/NPZ与权重仍在Git之外。
