# Frozen v5.2 B0 / FULL reference

保留原B0采样、Semi-MDP学习、执行环境、GoalManager/Controller和FULL Goal Adapter的依赖闭包，逐字节从历史归档恢复。该目录不是新的训练策略，不加入当前对手池。

所有历史策略类使用旧agents命名空间；查看或导入时必须在独立进程中将本目录放在sys.path首位，不能与主项目agents混用。

这里不恢复失败实验Runner、不恢复旧12步Waypoint训练流程，也不自动重启任何关闭路线。历史provenance.py中的绝对路径和核验契约原样保留；完整历史Runner与原始数据仍在仓库外归档，不能把这份源码快照宣称为所有旧实验的一键重跑入口。

当前可操作的官方环境、数据构建、BC、RL及配对评估位于项目根的vendor、agents/mappo、training与scripts。source_manifest.json标识所有冻结参考文件。
