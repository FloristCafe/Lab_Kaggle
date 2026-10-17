"""Audit matched review/R training arms and frozen memory conditions."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.strategic.experiment import digest,save_json
from agents.strategic.features import MODES


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/goal_review/E45_replan_memory_comparison')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    names={'without_r':'E4_without_r_seed42','with_r':'E4_with_r_seed42','corrupted':'E5_corrupted_seed42'}
    folders={key:ROOT/'experiments/goal_review'/name for key,name in names.items()}
    results={key:read(path/'summary.json') for key,path in folders.items() if (path/'summary.json').exists()}
    if not {'without_r','with_r'}<=results.keys():
        save_json(out/'summary.json',{'status':'PENDING_MATCHED_R_PAIR','completed':list(results)});return
    left,right=results['without_r'],results['with_r']
    ca,cb=copy.deepcopy(left['config']),copy.deepcopy(right['config'])
    assert not ca['model'].pop('replan') and cb['model'].pop('replan')
    ma,mb=[read(folders[key]/'manifest.json') for key in ('without_r','with_r')]
    epochs={key:[read(path/f'validation_epoch_{i}.json') for i in range(1,results[key]['stopped_epoch']+1)] for key,path in folders.items() if key in results}
    checks={'only_R_changes_in_matched_pair':ca==cb,
        'same_dataset':left['dataset_manifest_sha256']==right['dataset_manifest_sha256'],
        'same_matches':ma['train_files']==mb['train_files'] and ma['validation_files']==mb['validation_files'],
        'same_candidates':ma['training_mode_counts']==mb['training_mode_counts'] and ma['validation_mode_counts']==mb['validation_mode_counts'],
        'same_source':ma['source_sha256']==mb['source_sha256'] and ma['training_script_sha256']==mb['training_script_sha256'],
        'same_sampled_histograms':all(a['training_sampled_mode_counts']==b['training_sampled_mode_counts'] for a,b in zip(epochs['without_r'],epochs['with_r'],strict=True))}
    assert all(checks.values()),checks
    training_rows=[]
    for key,s in results.items():
        m=s['metrics'];r=m['free_replan']
        training_rows.append({'arm':key,'selection_context':s['config']['study']['context'],'best_epoch':s['best_epoch'],
            'head_macro_f1':m['mode']['macro_f1'],'executed_macro_f1':m['executed_mode']['macro_f1'],
            'joint':m['joint_mode_goal_accuracy'],'free_replan_precision':r['precision'][1],
            'free_replan_recall':r['recall'][1],'free_replan_f1':r['f1'][1],
            'free_continue_precision':r['precision'][0],'free_continue_recall':r['recall'][0],
            'false_replan':m['false_replan_rate'],'free_confusion':r['confusion_matrix']})
    studies={}
    for key in names:
        path=ROOT/'experiments/goal_review'/f'E5_memory_{key}'/'summary.json'
        if path.exists():
            s=read(path);assert not s.get('smoke_only') and s['training_updates']==0
            assert len(s['plan']['validation_files'])==145,'formal study requires full validation replay coverage'
            assert s['plan']['checkpoint_sha256']==digest(folders[key]/'best_model.pth')
            studies[key]=s
    protocol_checks={}
    if studies:
        plans=[s['plan'] for s in studies.values()]
        for field in ('validation_files','dataset_manifest_sha256','conditions','dev_seed','sides',
                      'prefix_steps','batch_games','tf32','source_sha256','evaluation_sources','runtime_memory_sha256'):
            protocol_checks[field]=all(plan[field]==plans[0][field] for plan in plans)
        protocol_checks['FP32']=all(s['inference_precision']=='FP32' for s in studies.values())
        assert all(protocol_checks.values()),protocol_checks
    eval_rows=[];online=[]
    for key,s in studies.items():
        for eta,values in s['conditions'].items():
            for condition in ('scripted','corrupted','policy_replay'):
                m=values[condition];r=m['free_replan']
                eval_rows.append({'arm':key,'eta':float(eta),'memory_condition':condition,
                    'macro_f1':m['mode']['macro_f1'],'executed_macro_f1':m['executed_mode']['macro_f1'],
                    'joint':m['joint_mode_goal_accuracy'],'explore_recall':m['per_mode']['EXPLORE']['recall'],
                    'free_replan_f1':r['f1'][1],'continue_recall':r['recall'][0],
                    'false_replan':m['false_replan_rate'],'review_frequency':m['macro_review_frequency'],
                    'coverage':m['goal_label_coverage'],'previous_valid':m['previous_goal_valid_fraction'],
                    'previous_modes':m['previous_mode_counts'],'previous_age':m['previous_age_mean'],
                    'all_frame_held_intent':m['all_frame_held_intent']})
            for prefix in values['closed_loop']:
                online.append({'arm':key,'eta':float(eta),**prefix})
    summary={'status':'FINISHED' if len(studies)==3 else 'PENDING_FROZEN_STUDIES','checks':checks,
        'training_rows':training_rows,'memory_rows':eval_rows,'online_prefixes':online,
        'frozen_protocol_checks':protocol_checks,
        'summary_sha256':{key:digest(folders[key]/'summary.json') for key in results},'readiness':'NOT READY'}
    save_json(out/'summary.json',summary)
    table='\n'.join(f"| {r['arm']} | {r['selection_context']} | {r['best_epoch']} | {r['head_macro_f1']:.4f} | {r['executed_macro_f1']:.4f} | {r['joint']:.2%} | {r['free_replan_f1']:.4f} | {r['free_continue_recall']:.2%} | {r['false_replan']:.2%} |" for r in training_rows)
    confusion='\n\n'.join(f"{r['arm']} free R confusion (row true, column prediction; Continue/Replan):\n\n```text\n{np.asarray(r['free_confusion'])}\n```" for r in training_rows)
    perclass='\n'.join(f"| {key} | {mode} | {s['metrics']['mode']['precision'][i]:.2%} | {s['metrics']['mode']['recall'][i]:.2%} | {s['metrics']['mode']['f1'][i]:.4f} | {s['metrics']['per_mode'][mode]['goal_top1']:.2%} | {s['metrics']['per_mode'][mode]['goal_error']:.3f} | {s['metrics']['per_mode'][mode]['goal_entropy']:.3f} |" for key,s in results.items() for i,mode in enumerate(MODES))
    memory='\n'.join(f"| {r['arm']} | {r['eta']} | {r['memory_condition']} | {r['joint']:.2%} | {r['free_replan_f1']:.4f} | {r['continue_recall']:.2%} | {r['false_replan']:.2%} | {r['review_frequency']:.3f} | {r['previous_age']:.2f} |" for r in eval_rows) or '| [待补充：完整冻结权重评估] | | | | | | | | |'
    held='\n'.join(f"| {r['arm']} | {r['eta']} | {r['memory_condition']} | {r['all_frame_held_intent']['units']:,} | {r['all_frame_held_intent']['joint']:.2%} | {r['all_frame_held_intent']['assigned_intent_fraction']:.2%} | {r['all_frame_held_intent']['anchor_prediction_rate']:.2%} |" for r in eval_rows) or '| [待补充：固定分母评估] | | | | | | |'
    prefixes='\n'.join(f"| {r['arm']} | {r['eta']} | {r['side']} | {r['primitive_steps']} | {r['review_frequency']:.3f} | {r['replan_rate']:.2%} | {r['goal_persistence']:.2%} | {r['goal_completion_rate']:.2%} | {r['invalid_goal_persistence']:.2%} |" for r in online) or '| [待补充：实际dev前缀] | | | | | | | | |'
    report=f"""# E4/E5：独立R与部署memory对照

## 1. 实验假设与核心矛盾

在宏review时刻加入独立Bernoulli R，检验Continue能否保持有效意图，并在因果规则、污染快照和策略历史memory间保持一致。当前状态={summary['status']}，结论仍为NOT READY，不能仅凭Replan正类F1决定可部署性。

## 2. 数据与防穿越边界

725场，580/145隔离；matched E4两组scope=review_epochs、单位候选/每轮曝光一致。验证共{ma['validation_unit_rows']:,}单位，与E1的64,414单位/stride4协议不同，E1→E4绝对差值不能全归因于R。
输入仅≤t合法观测，memory目标赋值来自<t的规则/策略输出；未来段与区域只生成监督。unknown filtered state保持mask，不默认Continue。
E5_corrupted的预设config同时用于训练和该arm的validation上下文，所以其原始summary的选模分数是在corrupted memory条件下。不能把它与scripted选模的E4 summary差异解释为纯训练corruption效应；下面用相同FP32冻结评估条件再比较。模型选择条件的差异仍作为局限保留。

## 3. 特征构建与数学表达

π(r,m,g|s,Iprev)：r=0保持(previous mode,goal,age)，r=1重新联合选择(m,g)；无有效previous时确定性强制Replan。Continue logprob仅logP(r=0)，Replan logprob为logP(r=1)+logP(m)+logP(g|m)。没有第577个KEEP。
E4两组关闭旧无界previous bias、context dropout=0；仅R head/条件执行语义不同。R=0样本只训练R，目标头在这些样本没有额外梯度；这是该策略分解的组成部分。
Hysteresis仅对一个(previous mode,goal)联合对加η≤0.5、age衰减、invalid时关闭；已测试其不改变训练目标或梯度。E5_hysteresis与combination采用相同已训练权重的η=0/.25推理对照，避免重复训练相同目标。

## 4. 离线评估与指标对比

### 训练各arm选中checkpoint（原始选模条件）

| Arm | 选模memory | Best epoch | Head Macro-F1 | 执行mode Macro-F1 | Joint | Free R1 F1 | Continue recall | False Replan |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{table}

{confusion}

Without_r是always-Replan对照，高R1 F1不代表学会R。应看自由决策单位的Continue召回、误重规划率与完整意图正确性，并把无效previous的强制分支从学习指标中分离。

| Arm | Mode | Precision | Recall | F1 | 条件Goal Top1 | Goal error | Goal entropy |
|---|---|---:|---:|---:|---:|---:|---:|
{perclass}

### 同权重、同145场验证的memory条件

| Arm | η | Memory | Joint | Free R1 F1 | Continue recall | False Replan | Reviews/frame | Previous age |
|---|---:|---|---:|---:|---:|---:|---:|---:|
{memory}

Scripted不是未来专家Goal，是真正因果规则memory；corrupted是标签盲的快照扰动；policy_replay把Actor过去输出回流memory，但专家物理轨迹保持固定，所以不是互动环境rollout。C的review时刻与可监督覆盖可能改变，比较时需同时读取coverage、previous-mode和age分布，不能仅看条件准确率。
离线规则previous-mode的DEFEND/INTERCEPT通道从未激活，实际策略会产生这些mode；这属于可测的支持域偏移，corruption引入随机mode为一种诊断/鲁棒性干预，不等同真实闭环错误轨迹。

### 固定物理帧分母：review时发出意图、非review时保持

| Arm | η | Memory | 有标签单位数 | Held-intent Joint | 已分配比例 | 当前坐标预测比例 |
|---|---:|---|---:|---:|---:|---:|
{held}

该表在同一组全部clean有效帧评分，A/B/C分母一致，避免不同review覆盖率造成选择偏差。Scripted/corrupted的输入memory仍各自固定，输出意图仅为非review帧计分而保持；只有policy_replay将Actor输出回流输入。实际物理轨迹仍来自录像，不能将这个指标解释为环境得分。
同一(mode,goal)身份的Replan不再刷新age、blocked与到达标记，防止虚假续期。新runtime保留冻结脚本memory的原始特征与标签；兼容性及逐帧/分批一致性检查已通过。

### 实际环境短闭环：既有dev seed15001、双侧128步预算

| Arm | η | Side | Steps | Reviews/frame | Replan rate | Goal persistence | Nontrivial completion | Invalid persistence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{prefixes}

Goal persistence比较有有效previous的目标身份是否保持，与R0比例分开；completion按每个非原地新意图首次抵达计数。Invalid persistence为0来自协议强制分支，不是学习成绩。在线无专家R真值，precision/recall/F1保持null，不制造标签。
这是人工截断的dev前缀，可能在当前option结束时略超128步，不报告完整比赛胜率。不能由单seed双侧前缀证明泛化或长期收益。
Q4/Q5仍须结合完整上述证据回答，不预设R或hysteresis已经解决mismatch。Q8当前NOT READY，不启动正式MAPPO。

## 5. 性能瓶颈与物理开销

四组后续训练在用户授权后并行执行；wall time反映共享设备竞争，不能与先前独占运行直接当成架构速度比较。GPU/host资源采样见autonomous_progress/resource_samples.csv。
冻结memory评估FP32、完整145场，每condition单独保存，支持中断恢复且检查checkpoint/数据/源代码哈希；额外训练更新数=0。CPU峰值/能耗归因[待补充：采样不等同独占归因]。
每组checkpoint、summary、config、dataset hash、源码hash和MLflow ID保持独立。原始训练产物与V2/官方环境未改写，最终holdout未执行。
"""
    (out/'research_report.md').write_text(report)
    print(json.dumps({'completed_training':list(results),'completed_memory_studies':list(studies),'readiness':'NOT READY'}))


if __name__=='__main__':main()
