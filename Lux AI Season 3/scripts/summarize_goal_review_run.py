"""Read and verify a finished controlled run; never train or evaluate new games."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.config import load_config
from agents.strategic.experiment import digest,save_json
from agents.strategic.features import MODES


def read(path):return Path(path).read_text()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true',help='save checks without the E1_clean-specific narrative')
    args=p.parse_args();run=args.run_dir;out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    summary=json.loads(read(run/'summary.json'));manifest=json.loads(read(run/'manifest.json'))
    assert summary['status']=='FINISHED' and not summary.get('smoke_only')
    dataset=ROOT/'data/dataset_v3_goal_review/manifest.json'
    d=json.loads(read(dataset));m=summary['metrics'];cfg=summary['config']
    train=set(manifest['train_files']);validation=set(manifest['validation_files'])
    seeds=lambda names:{d['files'][name]['source_seed'] for name in names}-{None}
    checkpoint=torch.load(run/'best_model.pth',weights_only=False,map_location='cpu')
    checks={
        'dataset_manifest_match':digest(dataset)==summary['dataset_manifest_sha256'],
        'checkpoint_dataset_match':checkpoint['dataset_manifest_sha256']==summary['dataset_manifest_sha256'],
        'checkpoint_epoch_match':checkpoint['epoch']==summary['best_epoch'],
        'checkpoint_macro_f1_match':abs(checkpoint['metrics']['mode']['macro_f1']-m['mode']['macro_f1'])<1e-12,
        'checkpoint_not_smoke':not checkpoint.get('smoke_only'),
        'train_validation_disjoint':not bool(train&validation),
        'source_seed_disjoint':not bool(seeds(train)&seeds(validation)),
        'config_matches_preregistered':cfg==load_config(ROOT/'configs/goal_review'/f'{run.name.removesuffix("_seed42")}.yaml'),
        'training_script_unchanged':digest(ROOT/'scripts/train_goal_review.py')==manifest['training_script_sha256'],
        'model_pipeline_sources_unchanged':all(digest(ROOT/name)==expected for name,expected in manifest['source_sha256'].items()),
        'dataset_producers_unchanged':all(digest(ROOT/name)==expected for name,expected in d['pipeline_sha256'].items())}
    assert all(checks.values()),checks
    import mlflow
    client=mlflow.MlflowClient(tracking_uri='sqlite:////home/issue/ml-workspace/artifacts/mlflow.db')
    info=client.get_run(summary['mlflow_run_id']).info
    assert info.status=='FINISHED',info.status
    epochs=[json.loads(read(run/f'validation_epoch_{i}.json')) for i in range(1,summary['stopped_epoch']+1)]
    exposures=np.asarray([e['training_sampled_mode_counts'] for e in epochs]).sum(0)
    trains=[e['training_epoch_mixed_policy'] for e in epochs]
    verification={'checks':checks,'run_name':run.name,'mlflow_status':info.status,'mlflow_run_id':info.run_id,
        'summary_sha256':digest(run/'summary.json'),'checkpoint_sha256':digest(run/'best_model.pth'),
        'manifest_sha256':digest(run/'manifest.json'),'dataset_manifest_sha256':digest(dataset),
        'train_games':len(train),'validation_games':len(validation),'training_unit_rows':manifest['training_unit_rows'],
        'validation_unit_rows':manifest['validation_unit_rows'],'sampled_exposures':exposures.tolist(),
        'training_mode_head_explore_predictions':sum(t['mode']['predicted_count'][1] for t in trains),
        'validation_mode_head_explore_predictions_each_epoch':[e['mode']['predicted_count'][1] for e in epochs],
        'budget_exhausted':summary['stopped_epoch']==cfg['study']['epochs'],
        'mappo_ready':'NOT READY'}
    save_json(out/'verification.json',verification)
    if args.verify_only:
        print(json.dumps({'run':run.name,'verified':True,'status':info.status}))
        return
    perclass='\n'.join(f"| {mode} | {m['mode']['support'][i]:,} | {m['mode']['precision'][i]:.2%} | {m['mode']['recall'][i]:.2%} | {m['mode']['f1'][i]:.2%} | {m['per_mode'][mode]['goal_top1']:.2%} | {m['per_mode'][mode]['joint_mode_goal_accuracy']:.2%} |" for i,mode in enumerate(MODES))
    epoch_table='\n'.join(f"| {i} | {e['mode']['macro_f1']:.4f} | {e['mode']['balanced_accuracy']:.2%} | {e['joint_mode_goal_accuracy']:.2%} | {e['training_sampled_mode_counts'][1]} | {e['mode']['predicted_count'][1]} |" for i,e in enumerate(epochs,1))
    report=f"""# {run.name}：正式结果复核

## 1. 实验假设与核心矛盾

在当前空间主干不变、普通 CE+目标区域 NLL 的条件下，检验严格合法 SAP 与删失清理后的有限预算 Goal IL baseline。
**运行已正常完成，但 EXPLORE 仍从未被选择，NOT READY。** 最佳第 {summary['best_epoch']} 轮；停止第 {summary['stopped_epoch']} 轮，达到配置上限，不是早停收敛。
本项不是 Replan/Continue 训练：model.replan={cfg['model']['replan']}。原始日志中约95.9%的 Replan正类F1来自 always-Replan，对 R 能力没有证明力。

## 2. 数据与防穿越边界

725 场，580/145 match/source-seed 隔离；训练 eligible 单位行 {manifest['training_unit_rows']:,}，验证 {manifest['validation_unit_rows']:,}，固定每四帧可监督单位行。网络特征仅≤t合法观测，历史 memory 赋值来自<t；未来轨迹只参与监督标签。
数据 manifest、模型/标签/训练代码哈希、检查点 epoch/分数与预注册配置均匹配，MLflow状态FINISHED。复核只读取现有产物，没有额外训练或赛程。
每轮16,000单位曝光，六轮合计 {int(exposures.sum()):,} 次；其中EXPLORE {int(exposures[1]):,} 次（{exposures[1]/exposures.sum():.2%}），不是对所有训练标签完整遍历六次。

## 3. 特征构建与数学表达

18通道×3帧ConvGRU，unit22→query32，π(m,g)=π(m)π(g|m)，推理联合argmax。
无R head；Previous Goal为因果规则memory，训练按帧25%清零该上下文。损失为confidence加权mode CE和专家mode条件目标区域NLL。
标签清理同时覆盖SAP合法性和删失尾段，因此即使后续matched dirty对照有收益，也不能在没有进一步消融时全归因于SAP一项。

## 4. 离线评估与指标对比

| 指标 | 最佳检查点 |
|---|---:|
| Mode-head accuracy | {m['mode']['accuracy']:.2%} |
| Mode-head Macro-F1 | {m['mode']['macro_f1']:.4f} |
| Mode-head balanced accuracy | {m['mode']['balanced_accuracy']:.2%} |
| 联合argmax执行mode accuracy | {m['executed_mode']['accuracy']:.2%} |
| 执行mode Macro-F1 / balanced accuracy | {m['executed_mode']['macro_f1']:.4f} / {m['executed_mode']['balanced_accuracy']:.2%} |
| Joint Mode+Goal correctness | {m['joint_mode_goal_accuracy']:.2%} |
| Goal Top1 / Top5 / Top10（专家mode条件） | {m['goal_top1']:.2%} / {m['goal_top5']:.2%} / {m['goal_top10']:.2%} |
| Goal最短Manhattan误差 / entropy | {m['goal_manhattan']:.3f}格 / {m['goal_entropy']:.3f} nat |
| 预测目标距离 / 专家region最短距离 | {m['predicted_goal_distance']:.3f} / {m['expert_region_distance']:.3f}格 |
| 段内相邻样本预测意图一致率 | {m['within_segment_prediction_consistency']:.2%}（{m['within_segment_pairs']:,}对） |

| Mode | support | Precision | Recall | F1 | 专家mode条件Goal Top1 | Joint |
|---|---:|---:|---:|---:|---:|---:|
{perclass}

混淆矩阵，行真值、列预测，顺序 SCORE/EXPLORE/DEFEND/INTERCEPT/RECHARGE：

```text
{np.asarray(m['mode']['confusion_matrix'])}
```

| Epoch | Val Macro-F1 | Balanced accuracy | Joint | 训练EXPLORE曝光 | 验证EXPLORE预测数 |
|---|---:|---:|---:|---:|---:|
{epoch_table}

主要解释：
- 六轮训练中的mixed-policy统计与每轮验证均无EXPLORE预测；这不是仅在陌生验证地图上失败，也不是单纯goal head把EXPLORE位置猜错。
- 普通损失下少数类梯度分配不足仍是有力假设，但尚未做weighted/balanced对照，不能宣判主因，也不能排除标签语义、容量和优化问题。
- INTERCEPT mode recall={m['mode']['recall'][3]:.2%}，但其专家mode条件Goal Top1仅{m['per_mode']['INTERCEPT']['goal_top1']:.2%}，对应joint仅{m['per_mode']['INTERCEPT']['joint_mode_goal_accuracy']:.2%}。长期攻击目标定位是独立的主要瓶颈。
- mode-head与联合执行mode指标不同；DEFEND条件Goal较好，但mode-head recall低，不能混成一个accuracy。
- 旧400场mode accuracy86.20%、joint18.81%只作历史参考；新580/145隔离、clean标签、memory和采样改变，19.14%不能当成可归因的提升。
- 六轮末仍选出最佳macro-F1，且训练/验证并无旧实验那种明显整体过拟合分离；有限96k曝光不能证明已充分优化。
- 离线goal persistence/segment consistency是在已标注采样行的相邻预测比较，不是闭环option完成。completion、invalid persistence、geodesic无可靠证据时保持null。

下一项为E1_dirty_control：相同扩展数据、划分、memory、网络与预算，仅训练标签视图改为旧噪声标签，验证仍严格clean。该对照完成后进入E2_weighted和E2_balanced，分离清理与梯度分配的影响。[待补充：这三项正式summary.json]。
E3容量、E4独立R、E5部署memory仍待执行；本项结果不足以修改架构或选E6，也不进入正式MAPPO。

## 5. 性能瓶颈与物理开销

完整训练+验证 {summary['wall_seconds']:.2f}s（{summary['wall_seconds']/60:.2f}分钟）；GPU峰值 allocated={summary['cuda_memory']['peak_allocated_mib']:.2f}MiB、reserved={summary['cuda_memory']['peak_reserved_mib']:.2f}MiB，实际B16/BF16已测量。
训练预算6,000次优化、96,000单位曝光；每轮完整验证64,414单位。CPU峰值/能耗[待补充：未采集]。显存远未达到8GiB限制，本次终止原因是轮数预算，不是OOM。
记录：MLflow {summary['mlflow_run_id']}；数据哈希 {summary['dataset_manifest_sha256']}。检查点和原始指标保持不变，复核证据见verification.json。
"""
    (out/'research_report.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    x=np.arange(1,len(epochs)+1)
    axes[0].plot(x,[e['mode']['macro_f1'] for e in epochs],label='Val macro-F1')
    axes[0].plot(x,[e['joint_mode_goal_accuracy'] for e in epochs],label='Val joint')
    axes[0].set_xlabel('Budget epoch (16k unit exposures)');axes[0].legend();axes[0].set_ylim(0,1)
    axes[1].bar(MODES,m['mode']['recall']);axes[1].set_title('Mode-head recall');axes[1].set_ylim(0,1)
    axes[1].tick_params(axis='x',rotation=30)
    axes[2].bar(MODES,[m['per_mode'][mode]['goal_top1'] for mode in MODES]);axes[2].set_title('Oracle-mode goal Top1')
    axes[2].set_ylim(0,1);axes[2].tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(out/'result_overview.png',dpi=160);plt.close(fig)
    print(json.dumps({'run':run.name,'verified':True,'status':info.status,'readiness':'NOT READY'}))


if __name__=='__main__':main()
