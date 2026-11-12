# 研究证据入口

- evidence/source_index.json：从外部归档复制的原始小型结果、协议与Manifest，保持字节不变并记录SHA256。
- evidence/G05.json、G06.json：完整地图级汇总与机制分母，不只保留有利地图。
- evidence/A01.json、A02.json、D12.json：成对策略对照与描述区间。
- evidence/D03.json、D07.json、D10.json及D07/D10/D11_notes.md：离线拟合和接口诊断的真实结论。
- evidence/PGC01.json、review_trigger.json：未晋级结论；对应大量失败Runner不恢复。
- results/bc_training.json：三模型全部24个epoch和原Calibration选择。
- results/training_a_games.jsonl：1200局官方终局胜数、地图、阵营、对手及原文件SHA。它是审阅摘录，不是完整动作轨迹。
- results/training_a_curve.json：300批训练损失、KL、梯度、对手统计和耗时。
- results/training_a_summary.json：训练期表现，禁止当成最终模型评估。
- model_index.json：外部BC与RL checkpoint身份；权重不提交Git。

python tools/recompute_results.py可在不导入模型、不启动环境的情况下复算关键表格。其复算范围与原始证据保留范围明确区分；历史区间保留原协议输出，不在整理时重选统计口径。
