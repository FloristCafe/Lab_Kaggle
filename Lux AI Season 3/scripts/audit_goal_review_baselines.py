"""Cheap causal baselines on exactly the supervised validation rows; no fitting."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.strategic.experiment import digest,save_json,setup
from agents.strategic.features import MODES


def classification(c):
    precision=c.diagonal()/np.maximum(c.sum(0),1)
    recall=c.diagonal()/np.maximum(c.sum(1),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return {'accuracy':float(c.trace()/max(1,c.sum())),'macro_f1':float(f1.mean()),
            'balanced_accuracy':float(recall.mean()),'precision':precision.tolist(),
            'recall':recall.tolist(),'f1':f1.tolist(),'confusion':c.tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/dataset_v3_goal_review')
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/goal_review/causal_baseline_audit')
    args=p.parse_args();setup();args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data_dir/'manifest.json').read_text())
    files=[name for name,row in manifest['files'].items() if row['split']=='validation']
    accum={scope:np.zeros((5,8),float) for scope in ('goal_frames','review_epochs')}
    confusion={scope:np.zeros((5,5),int) for scope in accum}
    from ml_workspace.tracking import tracked_run
    import mlflow
    with tracked_run('goal_review_causal_baselines',experiment_name='lux-s3-goal-review',params={
        'operation':'read_only_baseline_audit','validation_games':len(files),'seed':42,'training_updates':0}) as run:
        for name in files:
            path=args.data_dir/name
            if digest(path)!=manifest['files'][name]['npz_sha256']:raise ValueError('NPZ changed')
            with np.load(path,allow_pickle=False) as z:
                valid=z['label_valid'] & z['active'];ids=z['segment_id'];regions=z['goal_regions']
                area=regions.reshape(len(regions),-1).sum(1)
                positions=z['positions'];goals=z['goal'];units=z['unit_vectors'].astype(np.float32)
                mode=z['mode'];review=z['review_frames'];T=len(valid)
                for scope in accum:
                    chosen=valid.copy()
                    if scope=='goal_frames':chosen[np.arange(T)%4!=0]=False
                    else:chosen&=review[:,None]
                    t,u=np.where(chosen);sid=ids[t,u];y=mode[t,u]
                    xy=positions[t,u];target=goals[t,u]
                    assert (xy>=0).all() and (xy<24).all() and (area[sid]>0).all()
                    anchor_hit=regions[sid,xy[:,0],xy[:,1]].astype(bool)
                    pv=units[t,u,11]>.5
                    previous=np.rint(units[t,u,8:10]*23).astype(int).clip(0,23)
                    previous=np.where(pv[:,None],previous,xy)
                    pm=np.where(pv,units[t,u,12:17].argmax(1),3)
                    previous_hit=regions[sid,previous[:,0],previous[:,1]].astype(bool)
                    joint=previous_hit & (pm==y)
                    data=np.column_stack((np.ones(len(t)),anchor_hit,previous_hit,joint,area[sid]/576,
                                          np.abs(target-xy).sum(1),area[sid],pv))
                    np.add.at(accum[scope],y,data);np.add.at(confusion[scope],(y,pm),1)
        result={}
        for scope,a in accum.items():
            total=a.sum(0);n=max(1,total[0]);per={}
            for i,name in enumerate(MODES):
                v=a[i];denom=max(1,v[0]);per[name]={'count':int(v[0]),'oracle_mode_anchor_hit':v[1]/denom,
                    'oracle_mode_previous_goal_hit':v[2]/denom,'causal_previous_intent_joint':v[3]/denom,
                    'uniform_goal_expected_hit':v[4]/denom,'target_anchor_distance':v[5]/denom,
                    'target_region_area':v[6]/denom,'valid_previous_fraction':v[7]/denom}
            result[scope]={'units':int(n),'oracle_mode_anchor_hit':total[1]/n,
                'oracle_mode_previous_goal_hit':total[2]/n,'causal_previous_intent_joint':total[3]/n,
                'uniform_goal_expected_hit':total[4]/n,'mean_target_anchor_distance':total[5]/n,
                'mean_target_region_area':total[6]/n,'valid_previous_fraction':total[7]/n,
                'previous_mode_classification':classification(confusion[scope]),'per_mode':per}
        result.update(dataset_manifest_sha256=digest(args.data_dir/'manifest.json'),validation_files=files,
            source_sha256=digest(Path(__file__)),mlflow_run_id=run.info.run_id,training_updates=0,
            boundary='oracle-mode anchor/previous-location diagnostics use label mode only for scoring; NOT deployable joint policies. Causal previous-intent baseline uses only actual stored memory; invalid memory falls back to INTERCEPT/current coordinate.')
        save_json(args.output_dir/'summary.json',result)
        lines=[]
        for scope in accum:
            for mode,stats in result[scope]['per_mode'].items():
                lines.append(f"| {scope} | {mode} | {stats['count']:,} | {stats['oracle_mode_anchor_hit']:.2%} | {stats['oracle_mode_previous_goal_hit']:.2%} | {stats['causal_previous_intent_joint']:.2%} | {stats['target_region_area']:.2f} | {stats['target_anchor_distance']:.2f} |")
        report=f"""# 合法观测基线与目标区域难度审计

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
{chr(10).join(lines)}

聚合与混淆矩阵见summary.json。当前位置命中不一定是垃圾标签：SCORE/DEFEND/RECHARGE的长期持守可合法要求留在原地；不能把这些语义与EXPLORE/INTERCEPT混为同一个identity-collapse判断。
本审计没有人工意图真值，不能凭低模型Top1判定伪标签正确或错误，也不使用这些验证结果改变本轮标签。正式阶段仍需结合每类定位、模式选择、段一致性和短闭环。

## 5. 性能瓶颈与物理开销

仅流式读取现有压缩标签/单位状态，不读取隐藏引擎状态，不展开整库特征到磁盘；训练与比赛更新数均为0。数据和程序哈希、MLflow run ID保存在summary.json。
CPU峰值/能耗[待补充：本次未单独测量]。此结果属于诊断，不构成新增训练或新的holdout评估。
"""
        (args.output_dir/'research_report.md').write_text(report);mlflow.log_artifacts(str(args.output_dir))
        print(json.dumps({scope:{k:v for k,v in result[scope].items() if k not in ('per_mode','previous_mode_classification')} for scope in accum}),flush=True)


if __name__=='__main__':main()
