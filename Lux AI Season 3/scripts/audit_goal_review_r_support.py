"""Count task-specific Continue support under the frozen causal context policy."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.strategic.experiment import digest,save_json,setup
from agents.strategic.features import MODES


def main():
    setup();folder=ROOT/'data/dataset_v3_goal_review';out=ROOT/'experiments/goal_review/replan_support_audit'
    out.mkdir(parents=True,exist_ok=True);manifest=json.loads((folder/'manifest.json').read_text())
    accum={split:{'free':np.zeros((5,2),int),'forced':np.zeros(5,int),'previous_modes':np.zeros(5,int)} for split in ('train','validation')}
    from ml_workspace.tracking import tracked_run
    import mlflow
    with tracked_run('goal_review_R_task_support',experiment_name='lux-s3-goal-review',params={
        'operation':'read_only_replan_label_support','seed':42,'training_updates':0}) as run:
        for name,row in manifest['files'].items():
            with np.load(folder/name,allow_pickle=False) as z:
                valid=z['replan_valid']&z['review_frames'][:,None]&z['active'];uv=z['unit_vectors']
                free=valid&(uv[:,:,11]>.5);forced=valid&~free;mode=z['mode'];r=z['replan']
                assert (r[forced]==1).all()
                stats=accum[row['split']]
                np.add.at(stats['free'],(mode[free],r[free].astype(int)),1)
                stats['forced']+=np.bincount(mode[forced],minlength=5)
                stats['previous_modes']+=np.bincount(uv[free,12:17].argmax(1),minlength=5)
        result={split:{key:value.tolist() for key,value in values.items()} for split,values in accum.items()}
        result.update(mode_order=list(MODES),R_order=['Continue','Replan'],
            dataset_manifest_sha256=digest(folder/'manifest.json'),source_sha256=digest(Path(__file__)),
            mlflow_run_id=run.info.run_id,scope='scripted causal input memory; free decisions only, forced invalid-previous Replan counted separately',training_updates=0)
        save_json(out/'summary.json',result)
        rows='\n'.join(f"| {split} | {name} | {accum[split]['free'][i,0]:,} | {accum[split]['free'][i,1]:,} | {accum[split]['forced'][i]:,} |" for split in accum for i,name in enumerate(MODES))
        (out/'research_report.md').write_text(f"""# R标签按任务模态的支持域审计

## 1. 实验假设与核心矛盾

检查Continue是否覆盖五类宏任务，避免总体R正类F1掩盖某些任务从无继续样本的问题。

## 2. 数据与防穿越边界

完整725场，580/145隔离；只统计冻结NPZ的scripted causal memory和review时刻。未来意图区域只用于既有R监督，不进入输入。污染memory后的支持域会变化，不能把本表直接套到corrupted或policy回流条件。

## 3. 特征构建与数学表达

free=review & replan_valid & alive & previous_valid；按专家任务mode统计R0/R1。Previous无效的强制R1单独计数，不计为可学习的二分类成功。

## 4. 离线评估与指标对比

| Split | Expert mode | Free Continue | Free Replan | Forced Replan |
|---|---|---:|---:|---:|
{rows}

Previous-mode输入分布见summary.json。原规则memory只发出SCORE/EXPLORE/RECHARGE，因此在该输入分布下，DEFEND/INTERCEPT无法获得“previous同任务且目标兼容”的R0标签。共享网络可能泛化规则，但不存在这些任务的直接Continue监督；不能由总体Continue recall推断已学会长期防守/追击保持。
需要结合冻结策略回流和真实短闭环衡量这项支持域缺口。任务标签本身仍是启发式，本统计不提供人工战略意图真值，也不自动修改样本或新增训练。

## 5. 性能瓶颈与物理开销

逐NPZ读取既有小数组，无模型forward、优化或比赛，训练更新数0。数据/源码哈希与MLflow run ID归档；CPU峰值/能耗未单独采集。
""")
        mlflow.log_artifacts(str(out));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
