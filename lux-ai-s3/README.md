# Lux AI S3 · 精简架构存档

本目录由外层Lab_Kaggle仓库统一管理，不包含嵌套Git仓库。已停止训练项目维护，只保留少量可阅读的架构原型与物理代码。

现有A/B/C确实是轻量原型，不是先前提出的完整高上限架构：A为共享空间Actor（57220参数），B增加有限五小局空间GRU（61956参数），C增加单位Attention和条件前缀分配（63824参数）。保留代码用于结构参考，不宣称已实现完整Physics-first、长期Memory Transformer或反事实舰队学习系统。

## 保留内容

- agents/mappo/model_a.py、model_b.py、model_c.py：三个历史Actor原型。
- agents/mappo/shared/model.py：共享空间编码器、Actor和Critic定义。
- agents/mappo/shared/policy.py：条件R/Mode/Goal解码与合法支持。
- agents/mappo/shared/physics.py、features.py、observations.py：确定性物理工具及因果观察组件。
- agents/v2/：原生V2参考源码，保持不变。
- tests/test_architecture.py：CPU结构与基本因果边界测试。

训练、BC、评估Runner、对手池、实验配置、历史研究工具、vendor环境和繁杂文档已移出主仓库。这里没有正式训练/评估入口，也不承诺直接端到端运行比赛。测试中的张量schema供阅读与进一步重写时参考。

输入约定：grid[batch,19,24,24]、units[batch,16,22]、team[batch,6]、reach[batch,16,5,576]；B另需memory[batch,5,4,24,24]和memory_count。不要将这些轻量原型误当成已经验证的高竞争力Agent。

## 结构测试

在WSL Bash中：

```bash
cd /home/issue/Kaggle/Lab_Kaggle/lux-ai-s3
/home/issue/ml-workspace/.venv/bin/python -B -m pytest -q -p no:cacheprovider tests
```

## 仓库外归档

旧工程、数据、原始实验结果和第三方仓库集中移至：

/home/issue/ml-workspace/archives/lux-s3-retired-20261112_C01

归档保存清理前Git bundle、原始目录、逐文件SHA256、符号链接文本、特殊测试节点元数据及移动日志。RESTORE.md说明原路径映射。原工程可恢复，但不保留散落的训练指令作为当前入口。

Windows临时工作区另归档至C:\Users\lenovo\Documents\Codex-Archives\lux-s3-retired-20261112_C01。

没有训练、运行环境对战或访问保护集。
