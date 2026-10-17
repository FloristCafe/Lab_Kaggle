"""Read completed normal/weighted/balanced controls without new training/inference."""
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
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/goal_review/E2_imbalance_comparison')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    names={'normal':'E1_clean_seed42','weighted':'E2_weighted_seed42','balanced':'E2_balanced_seed42'}
    folders={key:ROOT/'experiments/goal_review'/name for key,name in names.items()}
    results={key:read(folder/'summary.json') for key,folder in folders.items() if (folder/'summary.json').exists()}
    assert 'normal' in results and 'weighted' in results
    manifests={key:read(folders[key]/'manifest.json') for key in results}
    epochs={key:[read(folders[key]/f'validation_epoch_{i}.json') for i in range(1,s['stopped_epoch']+1)] for key,s in results.items()}
    baseline=results['normal'];meta=manifests['normal'];checks={}
    for key,s in results.items():
        config=copy.deepcopy(s['config']);config['study'].pop('loss')
        base_config=copy.deepcopy(baseline['config']);base_config['study'].pop('loss')
        m=manifests[key]
        checks[key]={
            'finished':s['status']=='FINISHED' and not s.get('smoke_only'),
            'only_loss_sampling_changes':config==base_config,
            'same_dataset':s['dataset_manifest_sha256']==baseline['dataset_manifest_sha256'],
            'same_train_matches':m['train_files']==meta['train_files'],
            'same_validation_matches':m['validation_files']==meta['validation_files'],
            'same_training_rows':m['training_unit_rows']==meta['training_unit_rows'],
            'same_training_class_counts':m['training_mode_counts']==meta['training_mode_counts'],
            'same_validation_labels':m['validation_mode_counts']==meta['validation_mode_counts'],
            'same_sources':m['source_sha256']==meta['source_sha256'] and m['training_script_sha256']==meta['training_script_sha256']}
    checks['weighted']['same_sampled_mode_histograms_each_epoch']=all(
        a['training_sampled_mode_counts']==b['training_sampled_mode_counts']
        for a,b in zip(epochs['normal'],epochs['weighted'],strict=True))
    assert all(all(c.values()) for c in checks.values()),checks
    rows=[]
    for key,s in results.items():
        m=s['metrics'];explore=m['per_mode']['EXPLORE']
        rows.append({'arm':key,'best_epoch':s['best_epoch'],'mode_accuracy':m['mode']['accuracy'],
            'macro_f1':m['mode']['macro_f1'],'balanced_accuracy':m['mode']['balanced_accuracy'],
            'executed_macro_f1':m['executed_mode']['macro_f1'],'joint':m['joint_mode_goal_accuracy'],
            'goal_top1':m['goal_top1'],'explore_recall':explore['recall'],'explore_precision':explore['precision'],
            'explore_f1':explore['f1'],'explore_joint':explore['joint_mode_goal_accuracy'],
            'training_exposures':np.asarray([e['training_sampled_mode_counts'] for e in epochs[key]]).sum(0).tolist()})
    a,b=baseline['metrics'],results['weighted']['metrics'];n=a['valid_goal_units']
    evidence={'checks':checks,'rows':rows,'weighted_class_weights':manifests['weighted']['class_weights'],
        'weighted_minus_normal_net_joint_correct':round((b['joint_mode_goal_accuracy']-a['joint_mode_goal_accuracy'])*n),
        'validation_units':n,'summary_sha256':{key:digest(folders[key]/'summary.json') for key in results},
        'balanced_status':'FINISHED' if 'balanced' in results else 'PENDING_USER_RUN','ready':'NOT READY'}
    save_json(out/'comparison_evidence.json',evidence)
    table='\n'.join(f"| {r['arm']} | {r['best_epoch']} | {r['mode_accuracy']:.2%} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.2%} | {r['executed_macro_f1']:.4f} | {r['joint']:.4%} | {r['explore_recall']:.2%} |" for r in rows)
    class_header='| Mode | '+' | '.join(f'{key} precision / recall / F1' for key in results)+' |'
    class_separator='|---|'+'---:|'*len(results)
    classes='\n'.join('| '+mode+' | '+' | '.join(
        f"{s['metrics']['mode']['precision'][i]:.2%} / {s['metrics']['mode']['recall'][i]:.2%} / {s['metrics']['mode']['f1'][i]:.4f}"
        for s in results.values())+' |' for i,mode in enumerate(MODES))
    es=epochs['weighted']
    epoch_table='\n'.join(f"| {i} | {e['mode']['macro_f1']:.4f} | {e['mode']['recall'][1]:.2%} | {e['mode']['predicted_count'][1]} | {e['mode']['confusion_matrix'][1][1]} | {e['executed_mode']['confusion_matrix'][1][1]} | {e['per_mode']['EXPLORE']['joint_mode_goal_accuracy']:.2%} |" for i,e in enumerate(es,1))
    next_step=('E2_balanced尚待用户执行；保持normal CE，仅按单位类别平衡采样。'
               if 'balanced' not in results else 'E2三项已完成；下一步按既定E3计划独立改变空间主干，不自动叠加胜出的loss/sampler。')
    balanced_evidence=('[待补充：balanced sampler正式summary.json及对应曝光/召回取舍]。'
                       if 'balanced' not in results else f"Balanced实际曝光直方图={rows[-1]['training_exposures']}；性能与取舍见三组表，仍需要容量与部署条件对照。")
    conclusion=('加权改善模式头召回，但EXPLORE仍极弱，联合Goal表现基本不变；NOT READY。'
                if 'balanced' not in results else '平衡采样使EXPLORE Recall达到43.25%，但Precision仅5.94%，整体Joint降至17.85%；不平衡贡献得到支持，可靠Goal能力尚未建立，NOT READY。')
    balanced_section=''
    if 'balanced' in results:
        s=results['balanced'];c=s['metrics'];r=next(row for row in rows if row['arm']=='balanced')
        checks['balanced']['fixed_total_exposure']=sum(r['training_exposures'])==96000
        assert checks['balanced']['fixed_total_exposure']
        fp=c['mode']['predicted_count'][1]-c['mode']['confusion_matrix'][1][1]
        goal_rows='\n'.join(f"| {mode} | {c['per_mode'][mode]['goal_top1']:.2%} | {c['per_mode'][mode]['goal_error']:.3f} | {c['per_mode'][mode]['goal_entropy']:.3f} | {c['per_mode'][mode]['joint_mode_goal_accuracy']:.2%} |" for mode in MODES)
        balanced_section=f"""
### Balanced sampler：E2完整结果

每类约20%曝光，EXPLORE合计{r['training_exposures'][1]:,}次，是normal的{r['training_exposures'][1]/rows[0]['training_exposures'][1]:.2f}倍；总预算仍96,000。模型、候选集、验证标签、seed和源码不变。
最佳epoch{s['best_epoch']}：EXPLORE mode-head正确314/726，recall43.25%，precision5.94%，F1=0.1044；预测EXPLORE 5,289次，其中{fp:,}次误报，4,055次来自真实INTERCEPT。
联合(m,g)argmax后的EXPLORE mode正确181/726、recall24.93%、precision3.82%；完整EXPLORE intent命中18/726、2.48%。Mode recall与最终Goal成功不能混为一谈。
总体Joint={c['joint_mode_goal_accuracy']:.2%}，比normal下降{100*(a['joint_mode_goal_accuracy']-c['joint_mode_goal_accuracy']):.2f}pp；联合执行mode Macro-F1={c['executed_mode']['macro_f1']:.4f}，模式头accuracy={c['mode']['accuracy']:.2%}。在自然类别分布下，少数类召回提升伴随明显误报和多数类代价。

| Balanced mode | 专家mode条件Goal Top1 | Manhattan误差 | Entropy/nat | Joint |
|---|---:|---:|---:|---:|
{goal_rows}

Q1更新：**不平衡/梯度曝光是原先EXPLORE零预测的重要贡献因素**。在原架构不变时，平衡曝光已使EXPLORE召回持续非零，因此“原架构完全不能表示EXPLORE”缺少支持；同时precision和Goal仍差，不能宣称主要问题已解决，也不能排除容量/标签/优化与部署memory。
平衡采样同时改变mode和goal损失的类别曝光，INTERCEPT训练曝光也从normal的{rows[0]['training_exposures'][3]:,}次降到{r['training_exposures'][3]:,}次。因此整体Goal下降不能独立归因于空间编码器或某一种loss。
训练类别先验约均匀，验证保留自然先验；这是误报上升的一个合理解释，尚未做校准/先验修正对照，不认定它为唯一原因。本轮不新增此类调参。
原始标签为启发式真值，所以这些指标只度量对标签的一致性，不能直接证明真实战略探索能力或线上收益。第4轮EXPLORE recall峰值67.49%不是预注册最佳模型，不据此换checkpoint。
E2结论为指标取舍而非统一胜出：weighted的mode Macro-F1最高，balanced的balanced accuracy/EXPLORE recall最高，normal与weighted的整体Joint几乎相同且高于balanced。
下一项严格相对E1_clean：**E3_res6，普通损失、自然采样，仅换6×64ch、stride1空间主干**。容量实验不叠加E2采样变化，最终组合留待后续E6。仍为NOT READY。
Balanced耗时{s['wall_seconds']/60:.2f}分钟，peak allocated={s['cuda_memory']['peak_allocated_mib']:.2f}MiB、reserved={s['cuda_memory']['peak_reserved_mib']:.2f}MiB；达到第6轮上限，非patience早停。MLflow={s['mlflow_run_id']}。
"""
        evidence['checks']=checks
        evidence['balanced_minus_normal_net_joint_correct']=round((c['joint_mode_goal_accuracy']-a['joint_mode_goal_accuracy'])*n)
        save_json(out/'comparison_evidence.json',evidence)
    report=f"""# E2：类别不平衡对照阶段结果

## 1. 实验假设与核心矛盾

固定clean数据、网络和训练预算，仅改变Mode梯度分配，检验EXPLORE零预测是否主要受类别不平衡影响。
**{conclusion}** {next_step}
这是一条seed42/fold0的有限预算结果，支持梯度分配有影响，不能据此排除标签/容量/优化/部署偏移。

## 2. 数据与防穿越边界

725场，580/145 match与可用source-seed隔离；259,507训练eligible单位行、64,414相同clean验证行。标签候选计数{meta['training_mode_counts']}，EXPLORE占{meta['training_mode_counts'][1]/meta['training_unit_rows']:.2%}。
当前特征仅≤t合法观测，previous意图赋值来自<t，未来轨迹只进入监督。数据/源码哈希、候选行、划分与验证标签一致。
normal与weighted的每轮实际采样类别直方图完全相同，六轮均96,000单位曝光，EXPLORE均1,091次；改变的是损失权重，没有额外喂更多稀有样本。balanced干预改变曝光直方图，预算不变。

## 3. 特征构建与数学表达

18通道×3帧ConvGRU、unit22、当前空间主干、无R head；π(m,g)=π(m)π(g|m)，推理(m,g)联合argmax。
Mode loss为Σconfidence_i·w_y·CE_i / Σconfidence_i·w_y，w_c=min(N/(5N_c),5)。实际权重（SCORE/EXPLORE/DEFEND/INTERCEPT/RECHARGE）={manifests['weighted']['class_weights']}。
目标区域NLL、Previous Goal dropout=25%、lr=1e-3、weight_decay=1e-4、batch16、每轮1000更新和最多6轮均不变；loss数值因加权改变，不直接按train_loss高低判断优化优劣。

## 4. 离线评估与指标对比

同一预注册checkpoint选择：mode-head Macro-F1，Joint打破平局。下列是选中checkpoint，不以EXPLORE峰值事后换模型。

| Arm | Best epoch | Mode accuracy | Mode Macro-F1 | Balanced acc. | 联合执行mode Macro-F1 | Joint | EXPLORE recall |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

加权相对normal：Macro-F1 {b['mode']['macro_f1']-a['mode']['macro_f1']:+.4f}，balanced accuracy {100*(b['mode']['balanced_accuracy']-a['mode']['balanced_accuracy']):+.2f}pp；联合执行mode Macro-F1 {b['executed_mode']['macro_f1']-a['executed_mode']['macro_f1']:+.4f}。原始mode head收益没有自动传到联合策略。
Joint只净增加{evidence['weighted_minus_normal_net_joint_correct']}个正确单位样本，不能描述为有意义的整体Goal收益。Goal条件Top1为normal={a['goal_top1']:.2%}、weighted={b['goal_top1']:.2%}，Manhattan误差{a['goal_manhattan']:.3f}/{b['goal_manhattan']:.3f}格。

{class_header}
{class_separator}
{classes}

加权模型原始mode-head混淆矩阵，行真值/列预测，SCORE/EXPLORE/DEFEND/INTERCEPT/RECHARGE：

```text
{np.asarray(b['mode']['confusion_matrix'])}
```

最佳模型EXPLORE原始head：预测23次、正确4次，precision17.39%、recall0.55%、F1=0.0107；联合执行mode：预测38次、mode正确3次，但这3次的goal均未命中，EXPLORE Joint仍0。
EXPLORE专家mode条件Goal Top1={b['per_mode']['EXPLORE']['goal_top1']:.2%}，误差{b['per_mode']['EXPLORE']['goal_error']:.3f}格、entropy={b['per_mode']['EXPLORE']['goal_entropy']:.3f}nat；INTERCEPT条件Goal Top1={b['per_mode']['INTERCEPT']['goal_top1']:.2%}。目标空间推断没有被Mode类别权重解决。

| Weighted epoch | Macro-F1 | EXPLORE recall | head预测数 | head正确数 | 联合mode正确数 | EXPLORE Joint |
|---|---:|---:|---:|---:|---:|---:|
{epoch_table}

第2轮EXPLORE recall短暂4.68%，最终0.55%，并非稳定学会；按EXPLORE峰值挑epoch2会改变预注册选择标准。模式头/执行mode/完整Joint必须分开解释。
无R head，本轮always-Replan的高正类F1不能作为学会R的证据。没有新增闭环赛程，completion与invalid persistence无可靠证据时保持null。
单seed且样本按录像相关，未做跨seed或逐match配对置信区间，微小Joint差异仅作描述。{balanced_evidence}

{balanced_section}

## 5. 性能瓶颈与物理开销

normal耗时{baseline['wall_seconds']/60:.2f}分钟，weighted耗时{results['weighted']['wall_seconds']/60:.2f}分钟；weighted峰值allocated={results['weighted']['cuda_memory']['peak_allocated_mib']:.2f}MiB、reserved={results['weighted']['cuda_memory']['peak_reserved_mib']:.2f}MiB。
两组都达到epoch6预算上限，非patience早停，也没有OOM。CPU峰值/能耗[待补充：未采集]。
weighted MLflow={results['weighted']['mlflow_run_id']}；数据manifest={baseline['dataset_manifest_sha256']}。本报告只读取现有产物，未训练、推理或执行比赛。
"""
    (out/'research_report.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for key,values in epochs.items():
        axes[0].plot(np.arange(1,len(values)+1),[e['mode']['macro_f1'] for e in values],label=key)
        axes[1].plot(np.arange(1,len(values)+1),[e['mode']['recall'][1] for e in values],label=key)
    axes[0].set_title('Validation mode Macro-F1');axes[1].set_title('EXPLORE mode-head recall')
    for ax in axes[:2]:ax.set_xlabel('Budget epoch (16k unit exposures)');ax.legend()
    max_recall=max(e['mode']['recall'][1] for values in epochs.values() for e in values)
    axes[0].set_ylim(0,1);axes[1].set_ylim(0,min(1,max(.1,max_recall*1.2)))
    x=np.arange(5);width=.8/len(results)
    for i,(key,s) in enumerate(results.items()):
        axes[2].bar(x+(i-(len(results)-1)/2)*width,s['metrics']['mode']['recall'],width,label=key)
    axes[2].set_ylim(0,1)
    axes[2].set_xticks(x,MODES,rotation=30);axes[2].set_title('Selected mode-head recall');axes[2].legend()
    fig.tight_layout();fig.savefig(out/'comparison_overview.png',dpi=160);plt.close(fig)
    print(json.dumps({'arms':list(results),'checks_pass':True,'ready':'NOT READY'}))


if __name__=='__main__':main()
