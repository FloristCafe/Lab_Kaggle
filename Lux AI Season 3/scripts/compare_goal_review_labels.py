"""Compare completed E1 label arms from immutable metrics; no model inference."""
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
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/goal_review/E1_label_comparison')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    folders={arm:ROOT/'experiments/goal_review'/f'E1_{arm}_seed42' for arm in ('clean','dirty_control')}
    summaries={arm:read(folder/'summary.json') for arm,folder in folders.items()}
    manifests={arm:read(folder/'manifest.json') for arm,folder in folders.items()}
    a,b=summaries['clean'],summaries['dirty_control'];ma,mb=manifests['clean'],manifests['dirty_control']
    ca,cb=copy.deepcopy(a['config']),copy.deepcopy(b['config'])
    assert ca['study'].pop('label_view')=='clean' and cb['study'].pop('label_view')=='dirty'
    checks={'config_differs_only_label_view':ca==cb,
        'same_dataset':a['dataset_manifest_sha256']==b['dataset_manifest_sha256'],
        'same_train_matches':set(ma['train_files'])==set(mb['train_files']),
        'same_validation_matches':set(ma['validation_files'])==set(mb['validation_files']),
        'same_validation_rows':ma['validation_unit_rows']==mb['validation_unit_rows'],
        'same_validation_labels':ma['validation_mode_counts']==mb['validation_mode_counts'],
        'same_training_sources':ma['source_sha256']==mb['source_sha256'] and ma['training_script_sha256']==mb['training_script_sha256'],
        'both_finished':all(s['status']=='FINISHED' and not s.get('smoke_only') for s in summaries.values())}
    assert all(checks.values()),checks
    epochs={arm:[read(folder/f'validation_epoch_{i}.json') for i in range(1,summaries[arm]['stopped_epoch']+1)] for arm,folder in folders.items()}
    exposure={arm:np.asarray([e['training_sampled_mode_counts'] for e in es]).sum(0).tolist() for arm,es in epochs.items()}
    metrics={'Mode accuracy':lambda m:m['mode']['accuracy'],'Macro-F1':lambda m:m['mode']['macro_f1'],
        'Balanced accuracy':lambda m:m['mode']['balanced_accuracy'],'Joint correctness':lambda m:m['joint_mode_goal_accuracy'],
        'Goal Top1':lambda m:m['goal_top1'],'Goal Top5':lambda m:m['goal_top5'],'Goal Top10':lambda m:m['goal_top10'],
        'EXPLORE recall':lambda m:m['per_mode']['EXPLORE']['recall']}
    comparison={name:{'clean':fn(a['metrics']),'dirty':fn(b['metrics']),
        'clean_minus_dirty':fn(a['metrics'])-fn(b['metrics'])} for name,fn in metrics.items()}
    evidence={'checks':checks,'best_checkpoint_comparison':comparison,'sampled_exposures':exposure,
        'last_epoch_comparison':{name:{'clean':fn(epochs['clean'][-1]),'dirty':fn(epochs['dirty_control'][-1])} for name,fn in metrics.items()},
        'summary_sha256':{arm:digest(folder/'summary.json') for arm,folder in folders.items()},
        'validation_samples':ma['validation_unit_rows'],'seed':42,'fold':0,'independent_seed_repeats':1,
        'readiness':'NOT READY'}
    save_json(out/'comparison_evidence.json',evidence)
    table='\n'.join(f"| {name} | {row['clean']:.2%} | {row['dirty']:.2%} | {100*row['clean_minus_dirty']:+.2f} pp |" for name,row in comparison.items())
    modes='\n'.join(f"| {name} | {a['metrics']['mode']['recall'][i]:.2%} | {b['metrics']['mode']['recall'][i]:.2%} | {a['metrics']['per_mode'][name]['goal_top1']:.2%} | {b['metrics']['per_mode'][name]['goal_top1']:.2%} |" for i,name in enumerate(MODES))
    report=f"""# E1：Clean 与 Dirty 训练标签对照

## 1. 实验假设与核心矛盾

在同一数据、网络、memory、seed和固定优化预算下，检验标签清理能否缓解EXPLORE零召回与Goal定位失败。
**结果没有统一优势：clean的选中检查点Joint更高，dirty的Macro-F1更高；两组EXPLORE均始终零预测。** 单seed结果不足以宣称清理无效、噪声有益或网络容量不足。

## 2. 数据与防穿越边界

两组均725场，580训练/145验证；同一组64,414个clean验证单位样本、相同类别支持。网络输入来自≤t合法观测，Previous Goal赋值仅来自<t；未来轨迹只参与监督。
配置仅study.label_view不同；源码哈希、分组文件集合、验证行数和标签计数匹配。训练候选行clean={ma['training_unit_rows']:,}、dirty={mb['training_unit_rows']:,}，候选模式计数分别为{ma['training_mode_counts']}、{mb['training_mode_counts']}。
清理改变标签、分段和样本保留，故此对照估计的是**整个标签清理流程**的效应；不能把差异全归因于SAP过滤，也不是固定相同单位行仅换标签值。
两组每轮16,000单位曝光、共六轮96,000次。EXPLORE曝光clean={exposure['clean'][1]:,}、dirty={exposure['dirty_control'][1]:,}；均未完整遍历所有候选六次。

## 3. 特征构建与数学表达

共用18通道×3帧ConvGRU、22维unit输入、当前空间主干与五类mode/576格goal。π(m,g)=π(m)π(g|m)，推理联合argmax。
两组普通confidence加权mode CE+专家mode条件目标区域NLL；25% previous-context dropout；无R head。clean先过滤非法SAP证据并处理删失；dirty仅在隔离训练视图保留旧标签，验证始终clean。

## 4. 离线评估与指标对比

按预注册mode-head Macro-F1选模型，Joint打破平局：clean最佳epoch={a['best_epoch']}，dirty最佳epoch={b['best_epoch']}；两组均在epoch6达到预算上限，均未发生patience早停。

| 指标 | Clean | Dirty control | Clean−Dirty |
|---|---:|---:|---:|
{table}

Macro-F1作为百分数展示，上表clean=0.4737、dirty=0.5141；不能将其与accuracy混用。

| Mode | Clean recall | Dirty recall | Clean条件Goal Top1 | Dirty条件Goal Top1 |
|---|---:|---:|---:|---:|
{modes}

选择规则影响解读：dirty的epoch6 Joint={epochs['dirty_control'][-1]['joint_mode_goal_accuracy']:.2%}，但因Macro-F1低于epoch5，没有被选中。固定第6轮比较，clean Joint={epochs['clean'][-1]['joint_mode_goal_accuracy']:.2%}，差值仅{100*(epochs['clean'][-1]['joint_mode_goal_accuracy']-epochs['dirty_control'][-1]['joint_mode_goal_accuracy']):+.2f}pp；Macro-F1分别{epochs['clean'][-1]['mode']['macro_f1']:.4f}/{epochs['dirty_control'][-1]['mode']['macro_f1']:.4f}。不可为某一结论事后改成按Joint挑模型。
六轮训练mixed-policy统计与每轮验证的原始mode head中，两组EXPLORE预测数均0；各自最佳检查点的联合执行mode也未选择EXPLORE。说明当前配置/预算下，整个清理流程尚未解决少数类任务选择问题。INTERCEPT条件Goal Top1两组均约5%，也是独立瓶颈。
目前仅同一seed/fold各一条优化轨迹；未保存逐单位预测用于逐match配对重采样，不给虚构的置信区间。帧/单位相关，不能把64,414个标签当独立重复实验。
R head未启用，因此约95.9%的原始Replan正类F1属于always-Replan对照，不纳入已学会R的证据。尚无新增闭环完成率或胜率结果。

下一项为**E2_weighted**：使用clean标签，保持模型、预算、seed和划分不变，仅启用cap=5的class-weighted mode CE；再执行E2_balanced。观察EXPLORE是否开始被预测以及其它mode的precision/recall代价。
该顺序检验梯度分配假设；当前证据不足以跳到容量重构或E6，继续保持NOT READY，不启动正式MAPPO。[待补充：E2两组summary.json]。

## 5. 性能瓶颈与物理开销

Clean耗时{a['wall_seconds']/60:.2f}分钟；Dirty耗时{b['wall_seconds']/60:.2f}分钟。两组peak allocated={a['cuda_memory']['peak_allocated_mib']:.2f}/{b['cuda_memory']['peak_allocated_mib']:.2f}MiB，peak reserved均132MiB。停止由轮数预算决定，未发生OOM。
两组总训练曝光192,000；验证每轮均64,414单位，最后另重载最佳检查点完整验证。CPU峰值/能耗[待补充：未采集]。
MLflow clean={a['mlflow_run_id']}；dirty={b['mlflow_run_id']}；数据manifest={a['dataset_manifest_sha256']}。原始训练产物保持不变，比较程序只读取现有指标，没有训练、推理或赛程。
"""
    (out/'research_report.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for arm,es in epochs.items():
        axes[0].plot(np.arange(1,len(es)+1),[e['mode']['macro_f1'] for e in es],label=arm)
        axes[1].plot(np.arange(1,len(es)+1),[e['joint_mode_goal_accuracy'] for e in es],label=arm)
    axes[0].set_title('Validation Macro-F1');axes[1].set_title('Validation Joint')
    for ax in axes[:2]:ax.set_xlabel('Budget epoch (16k unit exposures)');ax.legend();ax.set_ylim(0,1)
    x=np.arange(5)
    axes[2].bar(x-.2,a['metrics']['mode']['recall'],.4,label='clean')
    axes[2].bar(x+.2,b['metrics']['mode']['recall'],.4,label='dirty')
    axes[2].set_xticks(x,MODES,rotation=30);axes[2].set_title('Selected checkpoint mode recall');axes[2].legend();axes[2].set_ylim(0,1)
    fig.tight_layout();fig.savefig(out/'comparison_overview.png',dpi=160);plt.close(fig)
    print(json.dumps({'output':str(out),'matched_controls':True,'readiness':'NOT READY'}))


if __name__=='__main__':main()
