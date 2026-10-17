"""Read controlled spatial-backbone results; no training, inference or games."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.model import ReviewActor
from agents.strategic.experiment import digest,save_json
from agents.strategic.features import MODES


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/goal_review/E3_capacity_comparison')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    names={'current':'E1_clean_seed42','res6':'E3_res6_seed42','res8':'E3_res8_seed42','dilated6':'E3_dilated_seed42'}
    folders={key:ROOT/'experiments/goal_review'/name for key,name in names.items()}
    results={key:read(folder/'summary.json') for key,folder in folders.items() if (folder/'summary.json').exists()}
    assert 'current' in results and 'res6' in results
    manifests={key:read(folders[key]/'manifest.json') for key in results}
    epochs={key:[read(folders[key]/f'validation_epoch_{i}.json') for i in range(1,s['stopped_epoch']+1)] for key,s in results.items()}
    base=results['current'];bm=manifests['current'];checks={};rows=[]
    for key,s in results.items():
        config=copy.deepcopy(s['config']);config['model'].pop('architecture')
        baseline=copy.deepcopy(base['config']);baseline['model'].pop('architecture')
        m=manifests[key]
        checks[key]={'finished':s['status']=='FINISHED' and not s.get('smoke_only'),
            'only_architecture_changes':config==baseline,
            'same_dataset':s['dataset_manifest_sha256']==base['dataset_manifest_sha256'],
            'same_matches':m['train_files']==bm['train_files'] and m['validation_files']==bm['validation_files'],
            'same_eligible_labels':m['training_mode_counts']==bm['training_mode_counts'] and m['validation_mode_counts']==bm['validation_mode_counts'],
            'same_sources':m['source_sha256']==bm['source_sha256'] and m['training_script_sha256']==bm['training_script_sha256'],
            'same_sampled_histograms':all(a['training_sampled_mode_counts']==b['training_sampled_mode_counts'] for a,b in zip(epochs['current'],epochs[key],strict=True))}
        assert all(checks[key].values()),checks[key]
        actor=ReviewActor(s['config']['model']);count=sum(p.numel() for p in actor.parameters() if p.requires_grad);del actor
        v=s['metrics']
        rows.append({'architecture':key,'parameters':count,'best_epoch':s['best_epoch'],
            'mode_accuracy':v['mode']['accuracy'],'macro_f1':v['mode']['macro_f1'],
            'balanced_accuracy':v['mode']['balanced_accuracy'],'executed_macro_f1':v['executed_mode']['macro_f1'],
            'joint':v['joint_mode_goal_accuracy'],'goal_top1':v['goal_top1'],'goal_top5':v['goal_top5'],
            'goal_top10':v['goal_top10'],'goal_manhattan':v['goal_manhattan'],'goal_entropy':v['goal_entropy'],
            'explore_recall':v['mode']['recall'][1],'explore_precision':v['mode']['precision'][1],
            'explore_f1':v['mode']['f1'][1],'wall_minutes':s['wall_seconds']/60,
            'peak_allocated_mib':s['cuda_memory']['peak_allocated_mib']})
    a,b=base['metrics'],results['res6']['metrics']
    evidence={'checks':checks,'rows':rows,'summary_sha256':{key:digest(folders[key]/'summary.json') for key in results},
        'res6_minus_current_net_joint_correct':round((b['joint_mode_goal_accuracy']-a['joint_mode_goal_accuracy'])*a['valid_goal_units']),
        'remaining':[key for key in names if key not in results],'readiness':'NOT READY'}
    save_json(out/'comparison_evidence.json',evidence)
    table='\n'.join(f"| {r['architecture']} | {r['parameters']:,} | {r['best_epoch']} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.2%} | {r['executed_macro_f1']:.4f} | {r['joint']:.2%} | {r['explore_recall']:.2%} |" for r in rows)
    goals='\n'.join(f"| {r['architecture']} | {r['goal_top1']:.2%} | {r['goal_top5']:.2%} | {r['goal_top10']:.2%} | {r['goal_manhattan']:.3f} | {r['goal_entropy']:.3f} |" for r in rows)
    classes='\n'.join(f"| {key} | {mode} | {s['metrics']['mode']['precision'][i]:.2%} | {s['metrics']['mode']['recall'][i]:.2%} | {s['metrics']['mode']['f1'][i]:.4f} | {s['metrics']['per_mode'][mode]['goal_top1']:.2%} | {s['metrics']['per_mode'][mode]['goal_error']:.3f} | {s['metrics']['per_mode'][mode]['goal_entropy']:.3f} |" for key,s in results.items() for i,mode in enumerate(MODES))
    costs='\n'.join(f"| {r['architecture']} | {r['wall_minutes']:.2f} | {r['peak_allocated_mib']:.2f} |" for r in rows)
    next_arm=next((names[key].removesuffix('_seed42') for key in names if key not in results),None)
    report=f"""# E3：空间容量受控对照阶段结果

## 1. 实验假设与核心矛盾

在clean标签、普通损失和自然采样不变时，仅替换空间主干，检验容量/下采样瓶颈能否解释EXPLORE零预测与Goal定位失败。
**Res6没有改善当前预算下的EXPLORE选择或整体Joint；不能由此排除全部容量/优化问题。NOT READY。** 当前已完成{len(results)}组空间结构，下一项={next_arm or '容量组完成，进入E4配对'}。

## 2. 数据与防穿越边界

同一725场，580训练/145验证，259,507训练候选单位行、64,414相同clean验证行；match和可用source seed隔离。输入≤t合法观测，历史意图赋值<t；未来轨迹仅用于标签。
同一seed42、normal loss、25%previous-context dropout、batch16、每轮1000更新、最多6轮；参数差异仅model.architecture。数据、源码、候选标签、训练/验证文件与每轮实际采样类别直方图完全一致。
每组96,000单位曝光，EXPLORE共1,091次；有限预算不是完整遍历六次。选模为mode-head Macro-F1，Joint打破平局。

## 3. 特征构建与数学表达

共用每帧18通道、三帧ConvGRU(hidden32)、unit22→query32、五类mode和五张576格条件goal；π(m,g)=π(m)π(g|m)，推理联合argmax。无R head、无attention/SE/FiLM/Transformer。
Current空间路径为32→48→64→48→32的两级down/up，6×6瓶颈带全局均值；Res6替换为6个64ch残差块、每块两层3×3、stride1、24×24保持，末端投影回32。
后续Res8为8个128ch残差块；dilated6为64ch、d=[1,2,3,5,8,4]。编码器之外头部与时序模块保持不变。该treatment同时改变深度/宽度/下采样/瓶颈全局路径，不能把差值解释为仅参数数量的作用。
各组GroupNorm具有全图空间耦合，Current也有显式全局均值，不能以局部卷积感受野断言其看不到远处。E0单点梯度图属于当时权重的空间敏感性诊断，不能代替新模型训练后的ERF/细定位结论。

## 4. 离线评估与指标对比

| Architecture | Trainable params | Best epoch | Mode Macro-F1 | Balanced acc. | 执行mode Macro-F1 | Joint | EXPLORE recall |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

Res6相对Current：Macro-F1 {b['mode']['macro_f1']-a['mode']['macro_f1']:+.6f}，Joint {100*(b['joint_mode_goal_accuracy']-a['joint_mode_goal_accuracy']):+.3f}pp，净正确单位数{evidence['res6_minus_current_net_joint_correct']:+d}。联合执行mode Macro-F1略升，但条件goal与Joint没有同步收益。

| Architecture | 专家mode条件Goal Top1 | Top5 | Top10 | Manhattan error | Entropy/nat |
|---|---:|---:|---:|---:|---:|
{goals}

| Architecture | Mode | Precision | Recall | F1 | 条件Goal Top1 | Goal error | Goal entropy |
|---|---|---:|---:|---:|---:|---:|---:|
{classes}

Res6的mode-head混淆矩阵，行真值/列预测，SCORE/EXPLORE/DEFEND/INTERCEPT/RECHARGE：

```text
{np.asarray(b['mode']['confusion_matrix'])}
```

六轮Res6训练mixed-policy和每轮验证的原始mode head都没有预测EXPLORE；最佳checkpoint的联合执行mode同样零预测。EXPLORE条件Goal Top1={b['per_mode']['EXPLORE']['goal_top1']:.2%}，INTERCEPT条件Goal Top1={b['per_mode']['INTERCEPT']['goal_top1']:.2%}，均未显示目标定位改善。
结合E2：原空间主干在balanced曝光下EXPLORE recall43.25%，因此架构加深不是出现非零探索预测的必要条件；不平衡抑制模式选择的证据比“仅因当前主干过浅”更强。这不证明容量永远无用，也不能把balanced的低precision/Goal归因给单一因素。
所有比较仍是单seed/同一fold，预设预算末轮仍可成为最佳。未做充分优化、容量×采样交互或跨seed对照，不能因Res6没有收益而宣判容量假设已被排除。
下一项按预注册{next_arm or 'E4_without_r'}推进；E3各项都相对Current使用normal loss和自然采样，以隔离空间结构。组合最优标签/采样/结构留待E6实际结果后决定。
Replan未启用；原始日志中的always-Replan F1不构成已学会R的证据。离线持久性是相邻标注采样行上的预测一致性，未测新闭环完成率或胜率；geodesic/online completion无可靠证据时保持null。

## 5. 性能瓶颈与物理开销

| Architecture | Wall/min | Peak allocated MiB |
|---|---:|---:|
{costs}

Res6峰值reserved={results['res6']['cuda_memory']['peak_reserved_mib']:.2f}MiB，显存未构成限制；停止在第6轮预算上限，并非patience早停或OOM。CPU峰值/能耗[待补充：未采集]。
Res6 MLflow={results['res6']['mlflow_run_id']}，数据manifest={base['dataset_manifest_sha256']}；原始checkpoint和训练指标保持不变。报告程序仅读取指标并统计模型参数，不做forward、训练或比赛。
[待补充：剩余架构summary.json，当前remaining={evidence['remaining']}]。当前NOT READY，不进入正式MAPPO。
"""
    (out/'research_report.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for key,es in epochs.items():
        x=np.arange(1,len(es)+1)
        axes[0].plot(x,[e['mode']['macro_f1'] for e in es],label=key)
        axes[1].plot(x,[e['joint_mode_goal_accuracy'] for e in es],label=key)
        axes[2].plot(x,[e['goal_manhattan'] for e in es],label=key)
    for ax,title in zip(axes,('Mode Macro-F1','Joint correctness','Oracle-mode goal Manhattan'),strict=True):
        ax.set_title(title);ax.set_xlabel('Budget epoch (16k unit exposures)');ax.legend()
    axes[0].set_ylim(0,1);axes[1].set_ylim(0,1)
    fig.tight_layout();fig.savefig(out/'comparison_overview.png',dpi=160);plt.close(fig)
    print(json.dumps({'architectures':list(results),'checks_pass':True,'next':next_arm,'readiness':'NOT READY'}))


if __name__=='__main__':main()
