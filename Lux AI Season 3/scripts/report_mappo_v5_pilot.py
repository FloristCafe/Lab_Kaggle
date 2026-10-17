"""Read completed pilots; no rollout, model inference or automatic next stage."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.strategic.experiment import digest,save_json


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dirs',nargs='+',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    records=[];references=[]
    for path in args.run_dirs:
        s=read(path/'summary.json');m=read(path/'manifest.json');seeds=read(path/'seed_plan.json');cfg=s['config']
        if s['status']!='FINISHED':raise ValueError('only complete pilots are comparable')
        updates=read(path/'update_metrics.json') if (path/'update_metrics.json').exists() else []
        trained=s['training'];evaluations=s['evaluations'];counters=[x['metrics'] for x in trained]
        checks={'complete_games':len(trained)==cfg['games'] and all(x['complete'] and not x['truncated'] and x['steps']==505 for x in trained),
            'finite_metrics':all(np.isfinite(v) for u in updates for v in u.values() if isinstance(v,(int,float))),
            'behavior_logprob':all(u['first_logprob_error']<=1e-4 for u in updates),
            'no_goal_conflicts':all(x.get('goal_conflicts',0)==0 for x in trained+evaluations),
            'evaluation_complete':all(x['complete'] and not x['truncated'] for x in evaluations),
            'IL_hash_same_as_manifest':m['initialization']['il_checkpoint_sha256']==s['initialization']['il_checkpoint_sha256']}
        total=lambda name:sum(x.get(name,0) for x in counters)
        per_mode=np.asarray([x['mode_counts'] for x in trained]).sum(0) if trained else np.zeros(5,int)
        row={'run':path.name,'arm':s['arm'],'seed':s['seed'],'training_games':len(trained),'checks':checks,
            'free_R_decisions':total('free_R'),'free_Continue':total('free_continue'),
            'Continue_rate':total('free_continue')/max(1,total('free_R')),'forced_actions':total('forced_actions'),
            'mode_counts':per_mode.tolist(),'largest_mode_share':float(per_mode.max()/max(1,per_mode.sum())),
            'nonzero_reward_options':sum(u['reward_nonzero'] for u in updates),
            'critic_mse_improved_updates':sum(u['critic_mse_after']<u['critic_mse_before'] for u in updates),
            'critic_updates':len(updates),'last_critic_mse':updates[-1]['critic_mse_after'] if updates else None,
            'win_rate':s['win_rate'],'mean_margin':s['evaluation_mean_margin'],'wall_seconds_current_attempt':s['wall_seconds_current_attempt'],
            'mlflow_run_id':s['mlflow_run_id'],'summary_sha256':digest(path/'summary.json')}
        row['behavior_review_required']=(len(trained)>0 and (row['largest_mode_share']>.95 or
            cfg['policy']['gate']=='learned' and (row['Continue_rate']<.01 or row['Continue_rate']>.99)))
        records.append(row);references.append((path,s,m,seeds))
    paired=[]
    for i,(pa,sa,ma,sea) in enumerate(references):
        for pb,sb,mb,seb in references[i+1:]:
            keys=('review','tactical','reward','opponents','evaluation','model')
            if any(sa['config'][key]!=sb['config'][key] for key in keys):raise ValueError('unmatched environment/evaluation')
            if ma['initialization']['il_checkpoint_sha256']!=mb['initialization']['il_checkpoint_sha256']:raise ValueError('different IL checkpoint')
            if ma['source_sha256']!=mb['source_sha256']:raise ValueError('different training/execution source snapshots')
            if sea['train'] and seb['train'] and sea['train']!=seb['train']:raise ValueError('unmatched training map seeds')
            a={(r['seed'],r['side'],r['opponent']):r for r in sa['evaluations']}
            b={(r['seed'],r['side'],r['opponent']):r for r in sb['evaluations']}
            if a.keys()!=b.keys():raise ValueError('unpaired evaluation cases')
            paired.append({'A':pa.name,'B':pb.name,'paired_cases':len(a),'unique_env_seeds':len({k[0] for k in a}),
                'B_minus_A_win_rate':float(np.mean([int(b[k]['win'])-int(a[k]['win']) for k in a])),
                'B_minus_A_mean_margin':float(np.mean([b[k]['game_margin']-a[k]['game_margin'] for k in a])),
                'scope':'paired dev estimate only; sides of one seed are dependent; no strength/significance claim'})
    result={'runs':records,'paired':paired,'stage':'PILOT_REVIEW_ONLY','formal_candidate':'NOT APPROVED',
        'next_action':'inspect collapse, critic targets, sparse returns and paired E1/E2; human research review before 1000 games'}
    save_json(args.output_dir/'summary.json',result)
    rows='\n'.join(f"| {r['run']} | {r['training_games']} | {r['win_rate']:.2%} | {r['mean_margin']:.3f} | {r['Continue_rate']:.2%} | {r['largest_mode_share']:.2%} | {all(r['checks'].values())} | {r['behavior_review_required']} |" for r in records)
    text=f"""# MAPPO v5 pilot 审核

## 1. 实验假设与核心矛盾

检验真实回报能否改善Mode/Goal，并用配对E1/E2判断学习R是否有额外收益。正式1000局候选尚未批准。

## 2. 数据与防穿越边界

仅公开观测与过去实际意图；所有对照共用checkpoint、环境与dev协议。训练map seed须匹配，dev双方交换阵营；dev已重复使用，不是最终留出集。每seed两侧不独立。

## 3. 特征构建与数学表达

共享Actor、独立Critic、真实采样分支old logprob与mask；R_k=Σγ^j r_j，bootstrap=γ^τ V，GAE使用(γλ)^τ。完整505步任务终止不bootstrap，外部截断bootstrap且切断trace。

## 4. 离线评估与指标对比

| Run | 完整训练局 | Dev win | Margin | 训练Continue | 最大mode占比 | 工程检查 | 行为需复核 |
|---|---:|---:|---:|---:|---:|---|---|
{rows}

配对差值、critic固定target前后MSE、非零奖励及源码/数据hash见summary.json及原始run。Frozen组无训练，训练Continue/mode占比的零值不是行为成绩。
R-neutral与原始R配置还改变R LR(1e-4 vs3e-5)，不能把两组差异纯归因为初始化。E2只有超过E1，且行为/时间/价值检查通过，才能讨论R的额外收益。100局与4个dev seed的初步结果不支持最终强度结论。

## 5. 性能瓶颈与物理开销

wall仅当前attempt，续跑各attempt MLflow需合并；磁盘checkpoint/source/rollout samples在原run，数据与模型均留WSL ext4。无自动后续训练。先读工程检查与行为警报，再审阅回报和Critic曲线；程序不会把工程PASS自动判成科研READY。
"""
    (args.output_dir/'research_report.md').write_text(text)
    print(json.dumps(result))


if __name__=='__main__':main()
