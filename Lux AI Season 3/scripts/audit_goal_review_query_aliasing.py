"""Bound label conflicts among units with exactly identical current actor inputs."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.strategic.experiment import digest,save_json,setup


def main():
    setup();folder=ROOT/'data/dataset_v3_goal_review';out=ROOT/'experiments/goal_review/query_aliasing_audit'
    out.mkdir(parents=True,exist_ok=True);manifest=json.loads((folder/'manifest.json').read_text())
    paths=[folder/name for name,row in manifest['files'].items() if row['split']=='validation']
    results={scope:{'units':0,'duplicate_groups':0,'duplicate_units':0,'conflicting_joint_groups':0,
        'unavoidable_mode_errors':0,'unavoidable_joint_errors':0,'unavoidable_oracle_mode_goal_errors':0}
        for scope in ('goal_frames','review_epochs')};examples=[]
    from ml_workspace.tracking import tracked_run
    import mlflow
    with tracked_run('goal_review_query_aliasing',experiment_name='lux-s3-goal-review',params={
        'operation':'read_only_identical_input_bound','seed':42,'validation_games':len(paths),'training_updates':0}) as run:
        for path in paths:
            with np.load(path,allow_pickle=False) as z:
                valid=z['label_valid']&z['active'];positions=z['positions'];units=z['unit_vectors']
                modes=z['mode'];ids=z['segment_id'];regions=z['goal_regions'];reviews=z['review_frames']
                for scope,totals in results.items():
                    ticks=np.flatnonzero(valid.any(1)&(reviews if scope=='review_epochs' else np.arange(len(valid))%4==0))
                    for t in ticks:
                        eligible=np.flatnonzero(valid[t]);totals['units']+=len(eligible);groups={}
                        for u in eligible:
                            key=positions[t,u].tobytes()+units[t,u].tobytes()
                            groups.setdefault(key,[]).append(int(u))
                        for group in groups.values():
                            n=len(group)
                            if n<2:continue
                            totals['duplicate_groups']+=1;totals['duplicate_units']+=n
                            target_modes=modes[t,group];counts=np.bincount(target_modes,minlength=5)
                            totals['unavoidable_mode_errors']+=n-int(counts.max())
                            best_joint=best_oracle_goal=0
                            for mode in range(5):
                                selected=np.asarray(group)[target_modes==mode]
                                if not len(selected):continue
                                votes=regions[ids[t,selected]].sum(0);best=int(votes.max())
                                best_joint=max(best_joint,best);best_oracle_goal+=best
                            totals['unavoidable_joint_errors']+=n-best_joint
                            totals['unavoidable_oracle_mode_goal_errors']+=n-best_oracle_goal
                            if best_joint<n:
                                totals['conflicting_joint_groups']+=1
                                if len(examples)<20:examples.append({'file':path.name,'scope':scope,'step':int(t),
                                    'units':group,'position':positions[t,group[0]].tolist(),
                                    'modes':target_modes.tolist(),'goals':z['goal'][t,group].tolist(),
                                    'interpretation':'exactly identical positions and stored 22 features within the same shared-history frame'})
        for r in results.values():
            for kind in ('mode','joint','oracle_mode_goal'):
                r[f'perfect_function_upper_bound_{kind}']=1-r[f'unavoidable_{kind}_errors']/max(1,r['units'])
        result={'scopes':results,'examples':examples,'dataset_manifest_sha256':digest(folder/'manifest.json'),
                'source_sha256':digest(Path(__file__)),'mlflow_run_id':run.info.run_id,'training_updates':0,
                'boundary':'same physical position + exact 22 scripted causal context features + same shared map/history implies identical per-unit queries; no UID input. Bound applies to deterministic top1 in this fixed input condition, not to sampled diversification or changed policy memory.'}
        save_json(out/'summary.json',result)
        (out/'research_report.md').write_text('''# Unit query 输入混叠审计

## 1. 实验假设与核心矛盾

检查同帧完全相同单位输入是否对应互斥伪标签，避免把不可辨识冲突误判成网络深度不足。

## 2. 数据与防穿越边界

完整145场既有验证录像；仅使用合法公开重建的position和22维因果memory，未来目标区域只用于标签冲突评分。scope分别与E1/E3和E4/E5对齐。

## 3. 特征构建与数学表达

同帧单位位置与22维输入逐字节相同时，共享空间编码和局部query输入完全相同。对n个这类单位，任意确定性联合预测最多命中max_(m,g)Σ1[label_mode=m and g∈region]个；oracle-mode目标上限按每个mode分别取最优格后求和。

## 4. 离线评估与指标对比

具体计数、上限和最多20个反例见summary.json。上限仅对当前scripted输入条件下的确定性Top1成立；随机采样或不同memory可改变单位输入/输出。接近100%的上限不能证明特征足够或可泛化，它只说明这一类可检验的冲突不足以解释全部失败。

## 5. 性能瓶颈与物理开销

仅读取压缩标签与单位状态，无新拟合、推理或比赛，无隐藏状态访问。CPU峰值/能耗未单独测量；源码、数据哈希与MLflow run ID归档于summary.json。
''')
        mlflow.log_artifacts(str(out));print(json.dumps(results),flush=True)


if __name__=='__main__':main()
