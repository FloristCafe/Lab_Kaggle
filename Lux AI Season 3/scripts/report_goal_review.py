"""Assemble measured E0 evidence and pending controlled-run handoff; no training."""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.metrics import classification
from agents.goal_review.config import load_config
from agents.strategic.experiment import digest,save_json
from agents.strategic.features import MODES

ARMS=('E1_clean','E1_dirty_control','E2_weighted','E2_balanced','E3_res6','E3_res8',
      'E3_dilated','E4_without_r','E4_with_r','E5_corrupted','E5_hysteresis','E5_combination')


def read(path):return json.loads(Path(path).read_text())


def command(arm):
    return f"""cd '/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3'
mkdir -p outputs/goal_review
set -o pipefail
/home/issue/ml-workspace/.venv/bin/python -u scripts/train_goal_review.py \\
  --config configs/goal_review/{arm}.yaml \\
  --data-dir data/dataset_v3_goal_review \\
  --output-dir experiments/goal_review/{arm}_seed42 \\
  --seed 42 --run-training 2>&1 | tee outputs/goal_review/{arm}_seed42.log"""


def runbook():
    text="""# Goal Review 受控实验执行手册

运行位置：VS Code 的 Ubuntu-26.04 / WSL Bash 终端。数据、模型和 MLflow 均留在 Linux ext4。
不再下载，不删除原始 JSON，不启动最终留出集或正式 MAPPO。

## 第一步只运行 E1_clean

实验 E1_clean；配置 configs/goal_review/E1_clean.yaml；数据 dataset_v3_goal_review；seed 42。
725 场分为 580 train / 145 validation，旧 80 个 validation ID 保留在 validation，按 Match ID 和可用 source seed 隔离。
此命令预计超过两分钟，由用户执行。**本次交付后停在这里；先检查 E1 结果再启动后续单项。**

[WSL Bash]
```bash
"""+command('E1_clean')+"""
```

默认预算为 6 个 epoch，每轮 1,000 次更新 × 16 个单位样本；BF16、Adam、lr=1e-3、weight_decay=1e-4。
这里的 epoch 是固定采样预算，**不是遍历全部 129 万标签一次**。所有同组实验预算一致；这是有限预算比较，不代表充分收敛。
监控 mode-head macro-F1，joint correctness 打破平局，patience=5；达到 6 轮上限会停止，不能自动称作早停收敛。
验证读取完整该 arm 的验证样本，不使用训练采样器，不按帧随机拆分。全量训练及验证实际用时待本机首轮记录。

输出：
- experiments/goal_review/E1_clean_seed42/best_model.pth：最佳检查点（不是 smoke）。
- 同目录 summary.json：最佳模型指标、耗时、显存、MLflow run ID。
- manifest.json / config.yaml：数据哈希、源码哈希、训练/验证文件、类别计数和配置。
- validation_epoch_N.json / validation_log.json：逐类 precision/recall/F1、混淆矩阵、Goal、R 指标。
- train_log.csv：曲线原始值；outputs/goal_review/E1_clean_seed42.log：完整终端输出。
- MLflow：lux-s3-goal-review；SQLite 跟踪库在 ml-workspace/artifacts。

重跑必须另设 output-dir 和 log 文件名，已有非空 run 目录会被拒绝。运行期间日志可实时读取，进程退出后仍保留。
判断完成以 summary.json + 正常退出为准；显存探针打印不等于训练完成。

## 后续实验顺序（逐个执行，不自动批量启动）

E1_dirty_control 是隔离的旧噪声标签训练对照；验证永远使用严格 clean 标签，不将 dirty 标签用于最终候选。
E2_weighted / E2_balanced 与 E1_clean 只改变梯度分配。
E3 三项均相对 E1_clean，保留普通损失和同一预算，不能同时叠加 E2 的胜出项来声称纯容量收益。
E4_without_r / E4_with_r 为单独配对：两者都用 review epochs、关闭 previous dropout 和旧无界 Previous Bias，仅 R head/Continue 语义不同。
E5 以 E4_with_r 为 A，分别加入 memory corruption、bounded hysteresis、组合。
E6 必须等上述实际结果后选候选并新训，当前没有“最佳组合”的证据。
"""
    for arm in ARMS[1:]:
        text+=f'\n### {arm}\n\n[WSL Bash]\n```bash\n{command(arm)}\n```\n'
    text+="""
## E5 的三种 memory condition 与短闭环

下面命令在 E4_with_r 正式训练完成后使用。scripted 是因果规则 memory，绝不是未来专家 Goal；corrupted 为快照扰动；policy_replay 为策略历史输出回流到 memory，但单位轨迹仍来自录像。
最后一项是真实环境短前缀，沿用已使用的 dev seed 15001，不构成整场胜率实验。

[WSL Bash]
```bash
cd '/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3'
set -o pipefail
/home/issue/ml-workspace/.venv/bin/python -u scripts/evaluate_goal_review.py \\
  --checkpoint experiments/goal_review/E4_with_r_seed42/best_model.pth \\
  --data-dir data/dataset_v3_goal_review \\
  --output-dir experiments/goal_review/E4_with_r_memory_eval \\
  --conditions scripted corrupted policy_replay --closed-loop-steps 64 \\
  2>&1 | tee outputs/goal_review/E4_with_r_memory_eval.log
```

对 E4_without_r、E5 三个 checkpoint 分别换 checkpoint/output-dir/log 名称执行，保持相同验证文件和 dev seed。
review epoch 分布可能随策略变化；同时比较覆盖率、previous-mode 分布、age、review frequency，不能只比较条件准确率。
在线没有专家 r 真值，因此 precision/recall/F1 为 null；不能拿协议强制 invalid→Replan 的零 invalid persistence 当作学习效果。

## 更新正式实验对照表

下面仅汇总现有结果，不训练。拒绝将 SMOKE_ONLY、不同数据哈希或未完成 run 填入正式表。

[WSL Bash]
```bash
cd '/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3'
/home/issue/ml-workspace/.venv/bin/python scripts/report_goal_review.py \\
  --smoke-dir experiments/goal_review/interface_smoke_final
```

输出 experiments/goal_review/research_report/{research_report.md,runbook.md,comparison.csv,comparison.json,evidence.json}。
最终 readiness 仍需按报告 Q1–Q8 人工审核，汇总脚本不会仅凭最高 accuracy 自动宣布 READY。
"""
    finished=[]
    for arm in ARMS:
        summary=ROOT/'experiments/goal_review'/f'{arm}_seed42/summary.json'
        if summary.exists():
            result=read(summary)
            if result.get('status')=='FINISHED' and not result.get('smoke_only'):finished.append(arm)
    if finished:
        next_arm=next((arm for arm in ARMS if arm not in finished),None)
        progress=f'已完成：{", ".join(finished)}。下方 E1_clean 是初始流程参考，不要重复运行。'
        if next_arm:
            progress+=f'当前下一项为 **{next_arm}**，仍由用户启动：\n\n[WSL Bash]\n```bash\n{command(next_arm)}\n```\n'
        text=text.replace('# Goal Review 受控实验执行手册','# Goal Review 受控实验执行手册\n\n'+progress,1)
    if (ROOT/'experiments/goal_review/autonomous_progress/plan.json').exists():
        text='# 自主执行状态与复现记录\n\n用户已在2026-10-17明确授权由agent执行后续必要实验，覆盖此前由用户启动长训练的分工。以下命令仅供复现归档，**无需用户逐项执行**。实时进度见autonomous_progress/latest.json，完成状态以各run的summary.json为准。\n\n'+text
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke-dir',type=Path,default=ROOT/'experiments/goal_review/interface_smoke_final')
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/goal_review/research_report')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    final=ROOT/'experiments/goal_review/final_research_report'
    if (final/'summary.json').exists():
        summary=read(final/'summary.json')
        for name in ('research_report.md','runbook.md'):
            source=out/name;archive=out/f'phase1_{name}'
            if source.exists() and not archive.exists():archive.write_text(source.read_text())
        (out/'research_report.md').write_text(
            '# Goal Review 完成结果索引\n\n'
            '本轮11个正式IL训练及完整冻结记忆研究已结束，结论NOT READY。\n\n'
            '完整结论、Q1–Q8及全部对照见[最终研究报告](../final_research_report/research_report.md)。\n\n'
            'phase1_*文件为历史阶段快照，其pending状态与启动指令已过期。\n')
        (out/'runbook.md').write_text(
            '# 当前执行分工\n\n'
            '本批实验已完成，当前没有待自动启动的实验。后续实验由用户在VS Code执行；'
            'Agent负责代码、检查与结果分析，并提供具体指令。\n\n'
            '不启动最终留出集或正式MAPPO。历史命令保留于phase1_runbook.md，仅用于审计。\n')
        print(json.dumps({'output':str(out),'completed_controlled_runs':summary['training_runs'],
                          'readiness':summary['readiness'],'execution_owner':'user'}))
        return
    audit=read(ROOT/'experiments/goal_review/E0_audit/summary.json')
    clean=audit['clean_dataset'];stats=clean['data_statistics'];smoke=read(args.smoke_dir/'summary.json')
    manifest=read(ROOT/'data/dataset_v3_goal_review/manifest.json')
    rows=[]
    for arm in ARMS:
        path=ROOT/'experiments/goal_review'/f'{arm}_seed42'/'summary.json'
        row={'experiment':arm,'status':'PENDING_USER_RUN','mode_accuracy':None,'macro_f1':None,
             'balanced_accuracy':None,'explore_recall':None,'joint_accuracy':None,'free_replan_f1':None}
        if path.exists():
            result=read(path)
            if result.get('smoke_only') or result.get('status')!='FINISHED':
                row['status']='EXCLUDED_SMOKE_OR_INCOMPLETE'
            elif result['dataset_manifest_sha256']!=digest(ROOT/'data/dataset_v3_goal_review/manifest.json'):
                row['status']='EXCLUDED_DIFFERENT_DATASET'
            elif result['config']!=load_config(ROOT/'configs/goal_review'/f'{arm}.yaml'):
                row['status']='EXCLUDED_DIFFERENT_CONFIG'
            else:
                m=result['metrics'];row.update(status='FINISHED',mode_accuracy=m['mode']['accuracy'],
                    macro_f1=m['mode']['macro_f1'],balanced_accuracy=m['mode']['balanced_accuracy'],
                    explore_recall=m['per_mode']['EXPLORE']['recall'],joint_accuracy=m['joint_mode_goal_accuracy'],
                    free_replan_f1=m['free_replan']['f1'][1] if result['config']['model']['replan'] else None)
        rows.append(row)
    save_json(out/'comparison.json',rows)
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    corpus=ROOT/'data/dataset_v3_goal_review'
    disk={suffix:sum(p.stat().st_size for p in corpus.glob('*'+suffix)) for suffix in ('.npz','.json')}
    sources=[*sorted((ROOT/'agents/goal_review').glob('*.py')),*sorted((ROOT/'scripts').glob('*goal_review*.py')),
             *sorted((ROOT/'tests').glob('test_goal_review*.py')),*sorted((ROOT/'configs/goal_review').glob('*.yaml'))]
    pipeline_ok={name:digest(ROOT/name)==expected for name,expected in manifest['pipeline_sha256'].items()}
    assert all(pipeline_ok.values()),'frozen producer changed'
    evidence={'audit_sha256':digest(ROOT/'experiments/goal_review/E0_audit/summary.json'),
        'dataset_manifest_sha256':digest(corpus/'manifest.json'),'producer_hashes_match':pipeline_ok,
        'smoke_sha256':digest(args.smoke_dir/'summary.json'),'disk_bytes':disk,
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'raw_json_count':len(list((ROOT/'data/raw_replays').glob('*.json'))),
        'formal_completed_runs':sum(r['status']=='FINISHED' for r in rows),'ready':'NOT READY'}
    (out/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    (out/'tracked.diff').write_text(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT,text=True))
    evidence['tests_log_sha256']=digest(ROOT/'outputs/goal_review_all_tests.log')
    matches=re.findall(r'Ran (\d+) tests', (ROOT/'outputs/goal_review_all_tests.log').read_text())
    test_count=int(matches[-1]) if matches else None
    handoff=('用户已授权由agent自主执行后续必要实验，不再交接启动指令。各项预算/划分保持冻结；hysteresis按同权重推理干预评估，最终结合完整对照解释E6与READY状态。'
             if (ROOT/'experiments/goal_review/autonomous_progress/plan.json').exists() else
             '后续按 comparison.json 中首个未完成对照推进，附件第十九条要求预计超过两分钟的完整训练由用户在 VS Code 启动。不能因为 E1 结果返回就自动启动下一项长训练，也不能跳过同数据、同预算的标签清理/不平衡/容量控制而预选 E6。')
    try:
        import mlflow
        client=mlflow.MlflowClient(tracking_uri='sqlite:////home/issue/ml-workspace/artifacts/mlflow.db')
        info=client.get_run(stats['mlflow_run_id']).info
        evidence['build_wall_seconds']=(info.end_time-info.start_time)/1000 if info.end_time else None
    except Exception as error:
        evidence['build_wall_seconds']=None;evidence['mlflow_read_error']=str(error)
    save_json(out/'evidence.json',evidence)
    old=classification(audit['legacy_mode_confusion']);total=stats['total_samples'];valid=stats['valid_samples']
    counts=stats['mode_counts'];r_n=stats['replan_supervised_units'];memory=stats['scripted_memory']
    table='\n'.join(f"| {name} | {counts[name]:,} | {counts[name]/valid:.2%} |" for name in MODES)
    cap='\n'.join(f"| {name} | {c['parameters']:,} | {c['conv_modules']} | {c['downsampling_stages']} | {c['effective_gradient_radius_80']} / {c['effective_gradient_radius_95']} | {c['gradient_mass_within_3']:.2%} | {c['parity_sublattice_density_cv']:.3f} |" for name,c in audit['capacity'].items())
    reasons='\n'.join(f'| {k} | {v:,} | {v/stats["invalid_samples"]:.2%} |' for k,v in stats['filter_reasons'].items() if v)
    perclass='\n'.join(f"| {name} | {old['support'][i]:,} | {old['precision'][i]:.2%} | {old['recall'][i]:.2%} | {old['f1'][i]:.2%} |" for i,name in enumerate(MODES))
    controlled='\n'.join(f"| {r['experiment']} | {r['status']} | {r['macro_f1'] if r['macro_f1'] is not None else '[待补充]'} | {r['explore_recall'] if r['explore_recall'] is not None else '[待补充]'} |" for r in rows)
    conditions='\n'.join(f"| {name} | {s['macro_review_frequency']:.3f} | {s['previous_goal_valid_fraction']:.2%} | {s['previous_age_mean']:.2f} | {s['previous_mode_counts']} |" for name,s in smoke['memory_conditions'].items())
    gpu='\n'.join(f"| {name} | {s['parameters']:,} | {s['cuda_memory']['peak_allocated_mib']:.1f} | {s['cuda_memory']['peak_reserved_mib']:.1f} | {s['logprob_error']:.2e} |" for name,s in smoke['architectures'].items())
    online=smoke['online_interface'];reviewfreq=stats['macro_decision_epochs']/stats['environment_frames']
    distance=np.asarray(stats['goal_distance_hist']);dmean=float((np.arange(len(distance))*distance).sum()/distance.sum())
    report=f"""# Goal-Level IL：标签与部署接口审计、受控实验交付

## 1. 实验假设与核心矛盾

本轮以合法 SAP、删失轨迹处理、共享因果 Goal memory 和独立 Replan head 建立可解释的 IL→Semi-MDP 接口；目前已完成 {evidence['formal_completed_runs']}/{len(ARMS)} 项正式对照，当前结论为 **NOT READY**。
旧模型高 mode accuracy 与 EXPLORE 零召回并存，且离线规则 memory 与策略部署 memory 的任务分布明显不同。标签、类别不平衡、表示能力、优化和部署偏移仍是竞争假设，不能预先指定主因。

### 当前实现审计（修改前）

- 当前 Actor 并非旧 35 通道 waypoint：每帧 18 个因果空间通道，三帧 ConvGRU；分组卷积、U 形两级下采样/上采样、坐标、瓶颈全局均值。单位 query 输入 32 维局部编码加 22 维物理/memory，输出 query 32；mode=[B,16,5]，goal=[B,16,5,24,24]。
- 推理使用 log P(m)+log P(g|m) 的联合 argmax；不是先选 mode，再选该 mode 的 goal。
- Previous Goal 原来来自当前及更早公开观测的规则规划，不是未来专家目标；训练/部署均有相同字段，但规则规划与 Actor 历史输出分布不同。
- 旧 Dataset 只为 label_valid 单位提供任务监督。过滤单位仍能出现在共享地图中；64.03% 单位被过滤不等于 64.03% 地图帧完全未见。
- 旧 GoalManager 有目标持久化和事件触发雏形，但没有可学习的独立二元 r；普通 score、路由停顿容易打开宏决策。
- Goal 标签来自较长语义轨迹段及目标区域，多证据启发式打分；不是 12 步 Gaussian waypoint。旧 SAP attempt 未先验验证能量与 Chebyshev 范围，可污染事件和分段。
- 已有 Semi-MDP MAPPO、centralized critic、duration-aware GAE 和实际环境桥接。上轮正式 RL/在线结果保留，本轮仅接口检查，不做 MAPPO 优化。
- 重写范围为新增 goal_review 名字空间、clean label wrapper、因果 memory、R 概率语义与对照实验入口；复用现有特征、标签器、战术控制和环境桥。**必须重建 NPZ，已完成。**
- 优先假设：类别不平衡/梯度分配、标签语义/团队探索归因、规则 Previous Goal 与部署分布偏移；容量问题必须通过 E3 检验。先 E0，再 E1。

## 2. 数据与防穿越边界

725 场现有 Kaggle EpisodeId 原始录像，{stats['environment_frames']:,} 个环境帧，{total:,} 个存活单位状态；580/145 按 Match ID 训练/验证隔离，可用 source seed 也不重叠，旧验证 80 个 ID 留在验证。没有读取本地赛程或最终 50 种子留出集。
所有 {evidence['raw_json_count']} 个原始 JSON 保留；没有下载、删除或向 Windows 复制高频训练 NPZ。

特征边界是 t_feature≤t_decision：当前合法观测允许用于当前决策；历史 memory 的目标赋值必须来自 <t 的动作输出，当前公开观测只更新存活/合法性与 age。
监督标签使用未来轨迹作 hindsight inference，t_feature<t_future_label，但标签不能回流到 memory。没有速度回填或未来投影输入。前 110 帧独立重建与整场 NPZ 的输入 memory/review 位逐值一致；这是一项具体前缀检查，不是所有路径的形式证明。

### 标签清理

SAP 必须满足单位存活、energy≥sap_cost、相对目标 Chebyshev 范围合法、绝对坐标在地图内，再参与事件、分段、任务打分和锚点选择。
{stats['invalid_sap_attempts']:,} 次非法尝试已剔除，{stats['legal_sap_attempts']:,} 次合法尝试保留；NPZ 审计发现 illegal SAP 作为事件的次数为 **0**。原动作只用于标签隔离副本，原始记录不修改。
right-censored 尾段和 episode 截断不再产生 Goal；死亡位置不是目标。未知过滤状态保持 mask，绝不默认 Continue。

严格有效 Goal {valid:,}（{valid/total:.2%}），过滤 {stats['invalid_samples']:,}（{stats['invalid_samples']/total:.2%}）。

| Mode | 标签数 | 占有效标签 |
|---|---:|---:|
{table}

INTERCEPT/EXPLORE={counts['INTERCEPT']/counts['EXPLORE']:.1f}:1；清理后不平衡仍然存在。
过滤原因采用**互斥诊断优先级**，不是人工语义真值；一次事件可能符合多个原因，不能把此表解释为因果贡献分解。

| 首要过滤原因 | 单位状态 | 占过滤状态 |
|---|---:|---:|
{reasons}

episode_boundary=0 因本批尾部均已归入 right_censored；无法可靠识别的 tactical avoidance 不伪造类别。
旧过滤审计见 E0 summary：1,033,836 no stable anchor；268,368 death/right-censored；105,023 ambiguous；17,472 insufficient duration；58,152 after-anchor/other。
旧 EXPLORE 标签 8,657 中 7,990 有附近友军，4,684 同时全局 rank 增加；团队探索收益的个体归因需人工核验，不能仅凭相关性判错。已生成 20 条盲审队列；[待补充：人工语义标注，见 human_annotation_queue.json]。

### 决策 epoch 与 R 分布

| 项目 | 计数/比例 |
|---|---:|
| raw environment frames | {stats['environment_frames']:,} |
| 有至少一个 Goal 标签的帧 | {stats['frames_with_valid_goal']:,} |
| 有至少一个被过滤单位的帧 | {stats['filtered_environment_frames']:,} |
| scripted macro review epochs | {stats['macro_decision_epochs']:,} / {reviewfreq:.2%} |
| 可靠语义段起点/恢复点 | {stats['semantic_boundaries']:,} |
| review 上可监督的 r 单位样本 | {r_n:,} |
| Replan / Continue | {stats['replan_states']:,} / {stats['continuation_states']:,} |
| Replan 正例比例 | {stats['replan_states']/r_n:.2%} |
| 其中因 memory 不兼容而需纠正的 Replan | {stats['corrective_replans']:,} |
| 被安全桥接恢复的 filtered Continue | {stats['bridged_filtered_continues']} |

安全桥接要求同一生命期、短缺口、两端可靠同 mode 且目标区域相交、无非法 SAP 证据、因果 previous 仍兼容；本批没有满足全部条件的新增 Continue，不能声称已经补齐 filtered-state coverage。
不能一概 segment interior→Continue：因果规则 memory 往往与未来专家段不兼容，强标 Continue 会训练错误目标。实现同时保存 boundary_replan（语义标签）和 corrective_replan（当前 memory 条件目标），并在报告分开计数。语义 boundary 不是在线输入。
scripted memory 执行 {memory['decisions']:,} 次单位级决策；只有 {r_n/memory['decisions']:.2%} 有可靠 R 监督，未覆盖部分仍是主要阻塞。
review 间隔均值/中位/P90/P95={clean['scripted_review_option_duration']['mean']:.3f}/3/3/3 步（跨整场含首尾边界）。生命变化与重大得分事件仍频繁触发，阈值重构没有自动消除分布偏移。

## 3. 特征构建与数学表达

π(r,m,g|s,Iprev)：r=0 仅执行 (mprev,gprev)，r=1 执行五类 mode 与 576 格条件 goal；没有第 577 个 KEEP。
Continue 的行为 logprob=log P(r=0)；Replan=log P(r=1)+log P(m)+log P(g|m)。无有效 previous 时协议强制 Replan，P(r)=1，该维不伪造 Bernoulli 梯度。R=0 不训练新 mode/goal。
greedy 实现先按 r 概率≥0.5 做二元判定，若 Replan 再做 **(m,g) 联合 argmax**；不宣称跨 r 与全部 2,880 个新目标做完整联合 MAP。PPO 采样路径严格按分解概率，单独保存 r 与行为 logprob。

宏 review 条件为生命周期/重置、首次抵达、V2 证伪、持续阻塞、持续严重威胁、重大得分/信息变化、最长 16 步；通常最小 3 步，生命周期/无效记忆可提前。单次 A* reroute、局部闪避和普通敌人经过不自动结束 Goal。持续阻塞用公开观测静止计数近似；未知地图上没有“已证明不可达”的可靠证书，不伪造该事件。

offline 与 online 共用 ReviewMemory。scripted 条件只用因果规则，corrupted 只扰动已有 memory（坐标/age/mode/valid），policy_replay 把 Actor 历史输出送到后续 memory，但环境位置仍是固定录像；只有 actual closed-loop probe 由实际动作改变后续状态。
bounded hysteresis 只提升前一个 **(mode,goal) 联合对**，η≤0.5、age≥80 时衰减至 0，invalid previous 时关闭，memory age≥100 强制失效；不会永久锁住旧目标。

### 网络容量与有效空间敏感性

| 架构（无 R） | 参数 | Conv 模块 | 下采样级数 | 梯度半径 R80/R95 | 3 格内梯度质量 | 奇偶子网格密度 CV |
|---|---:|---:|---:|---:|---:|---:|
{cap}

current 为 stem+2 down+2 up，宽度 32→48→64→48→32；res6 为 6×64ch 残差块；res8 为 8×128ch；dilated6 为 64ch、d=[1,2,3,5,8,4]，均保持 24×24。每块两层 3×3 卷积；ConvGRU 仍展开三帧、hidden32；query32，mode Linear32→5，goal 五个 query 与空间 key 点积。
四组共用原始 grouped history encoders、ConvGRU、单位输入与 mode/goal heads，仅替换空间主干；空间主干 treatment 包括去掉降采样和对应瓶颈全局均值路径，不能把结果解释为“仅参数量”收益。
GroupNorm 在空间维聚合，current 还有显式全局均值，理论信息支持均可全图，不能继续声称远处资源在数学上完全不可见。current 的 6×6 压缩会损失细定位，32 维 temporal/query 瓶颈在各组保持，是另一个可能限制。
梯度审计是在一个真实 EXPLORE 验证状态上，对 mode EXPLORE logit 的输入梯度取绝对值并跨时间/通道求和；仅 current_trained 加载旧已训练权重，其余为随机初始化。该单点诊断与奇偶网格 CV 不能证明训练后 gridding/容量优劣，需 E3 和 Goal 细定位指标验证。

## 4. 离线评估与指标对比

### 已存在的旧 400 场基线

mode accuracy={old['accuracy']:.2%}，macro-F1={old['macro_f1']:.4f}，balanced accuracy={old['balanced_accuracy']:.2%}；joint mode+goal=18.81%。该结果只作历史参照，数据规模、清理、memory 和采样协议改变后不能作为单变量因果对照。

| Mode | 验证 support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
{perclass}

混淆矩阵（行真值、列预测，顺序 SCORE/EXPLORE/DEFEND/INTERCEPT/RECHARGE）：

```text
{np.asarray(audit['legacy_mode_confusion'])}
```

EXPLORE 预测列总和=0：旧 **mode head 从未选择 EXPLORE**，因此 0 recall 不能只归咎于 goal head。Joint argmax 的 mode 与单独 mode-head argmax另行记录，避免混用。

### 本轮正式受控实验（不填 smoke accuracy）

| 实验 | 状态 | Macro-F1 | EXPLORE recall |
|---|---|---:|---:|
{controlled}

E1 noisy/clean 对照共享扩展数据、隔离验证、因果 memory、网络与计算预算，验证一律 clean；E2 normal/weighted/balanced 只变梯度分配；E3 保持 normal loss 只变空间主干。
E4 without/with R 两组均在 review epochs 训练、关闭 context dropout 与旧无界 Previous Bias，只在组内归因 R。不能把 E1→E4 差值全部归因于 R。
E5 以 E4 with R 为 scripted A，再做 corrupted B、policy-feedback C 与 bounded hysteresis/组合。E6 尚无胜出结果，**不预选或训练“最佳组合”**。
所有正式指标输出 per-class、混淆矩阵、expert-mode Goal Top1/5/10、Manhattan、entropy/peak、执行 joint accuracy、EXPLORE goal error/count、R/free-R、goal distance、段内一致性和样本持久性。未启用 R head 的 run 产生的是 always-Replan 对照，正式对照表的 free_replan_f1 留空，不把该常量策略的高正类 F1 当作学习效果。动态/未知障碍的 geodesic error 未定义，明确 null。

### 仅接口 smoke：不是泛化证据

四个架构各两个 B=4 优化步；总耗时 {smoke['wall_seconds']:.2f}s，formal training executed=false，MAPPO optimizer updates=0。
跨单条采样与批量重算使用 FP32 并关闭 TF32；先前 TF32 版本未过 1e-4 阈值的失败日志保留，不通过放宽阈值掩盖。

| 架构（含 R） | 可训练参数 | Peak allocated MiB | Peak reserved MiB | 最大 Logprob 误差 |
|---|---:|---:|---:|---:|
{gpu}

短录像前 32 帧 memory 条件：

| 条件 | review/frame | previous valid | age 均值 | previous-mode counts |
|---|---:|---:|---:|---|
{conditions}

实际环境前缀 {online['primitive_steps']} 步、{online['macro_decisions']} 个高层 transition，review frequency={online['review_frequency']:.3f}，τ均值={online['option_duration']['mean']:.3f}；随机/两步权重产生的 behavior 不作为性能结论，未跑完整比赛。primitive step budget 允许当前 option 结束时少量超出。
宏 transition 包含 (B,A,logπ,R,Δ,Bnext,done)，R=Σγ^j rj，bootstrap=γ^ΔV(Bnext)，GAE λ 按 primitive timebase；critic 仍为共享公开全队观测，不接入隐藏真实状态。只检查兼容性，未做 RL 参数更新。新 R checkpoint 不能直接传给旧 train_goal_mappo.py；本轮交付新的 Actor/环境/行为概率接口，正式 R-aware PPO rollout buffer 与 optimizer 接线留到 IL 通过后，不能宣称已经完成新 R 策略的正式 MAPPO 训练入口。
在线无专家 r 真值，Replan precision/recall/F1=null。invalid-goal persistence=0 来自强制合法性协议，不是策略学会纠错的证据。Goal completion 按每个非原地新意图首次到达计数，已在目标上的赋值不计完成率分母。

### Q1–Q8 的当前回答

| 问题 | 结论与证据 / 尚缺证据 |
|---|---|
| Q1 EXPLORE=0 主因？ | 尚不能确定。mode head 从未选择、约 70:1 不平衡、团队探索归因和规则 memory 任务偏移均有证据；[待补充：E1/E2/E3/E5 受控结果]。 |
| Q2 编码器过浅？ | 已有全图支持，不能以局部感受野推断失败；容量/下采样与 32 维瓶颈可疑；[待补充：E3 的 EXPLORE、Goal 结果]。 |
| Q3 filtered 导致 mismatch？ | 任务监督 coverage 确实不足。旧 64.03%、新 {stats['invalid_samples']/total:.2%} 单位过滤；但旧 82.17% 帧仍有至少一个 Goal 标签，不能等价成整幅地图未见。offline scripted review={reviewfreq:.3f}，旧 online IL-only≈0.3495；不同数据/状态分布不能据此归因。新短 smoke非稳态，需正式 checkpoint 的三个 condition。 |
| Q4 R 是否解决？ | 未验证；目前 {stats['replan_states']/r_n:.2%} 为 R1，多数 corrective。必须看 E4配对和free-R precision/recall/F1、false Replan、switch/persistence/completion，不能看整体 R accuracy。 |
| Q5 Previous Goal exposure bias？ | 旧 offline Previous Mode 绝大多数 EXPLORE/SCORE/RECHARGE，DEFEND/INTERCEPT均0；online 旧 Actor 几乎不选 EXPLORE，是强分布差异证据，不等于已证明其导致失败。需冻结同权重 A/B/C 与实际短闭环。 |
| Q6 86.20% 是否误导？ | 是，若解读为五类均学会则错误；macro-F1={old['macro_f1']:.3f}、balanced={old['balanced_accuracy']:.3f}，EXPLORE=0，少数类差异被多数 INTERCEPT 掩盖。 |
| Q7 长期战略还是 waypoint？ | 标签距离均值 {dmean:.2f} 格，含原地/保持类；采用≤80步语义段及目标区域而非固定12步，但这不能证明 Actor 学会长期规划。[待补充：正式预测距离、段内一致性、持久性、切换、非平凡完成率和盲审]。 |
| Q8 进入 MAPPO？ | **NOT READY**：清理未证明语义正确；过滤覆盖仍低；R标签严重失衡且偏纠错；受控训练尚未完成；部署 memory 与真实闭环尚无可靠性能证据。接口 sanity 通过不等价训练准备就绪。 |

## 5. 性能瓶颈与物理开销

紧凑 NPZ 总计 {disk['.npz']/2**20:.2f} MiB，segment/manifest/summary JSON {disk['.json']/2**20:.2f} MiB；无需把全部数据展开到内存，LRU=1 shard，sampler 对各组同样按 shard 分组访问，平衡发生在单位梯度样本级。
此 schema 临时保留 dirty labels 作为隔离对照，因此不能与旧 123MB waypoint 数据的体积直接相除比较。
725 场数据构建 MLflow wall_seconds={evidence['build_wall_seconds']}；CPU峰值内存/能耗 [待补充：本次未采集]。完整训练耗时、B16 峰值 [待补充：E1 summary.json]；B4显存实测不能冒充B16测量，按激活近似线性外推各组 <1GiB，仍以运行时resource guard为准。
训练正式执行每轮16,000单位梯度曝光、最多6轮，最大96,000次单位曝光，少于全库有效标签；有限预算不能支持“模型已充分收敛/容量无效”的结论。验证 IO/前向很可能超过训练步耗时，首轮记录后再定预算，禁止悄悄更改各 arm 配额。
{test_count}项单元测试、项目解析测试与静态检查见验证清单。Git tracked.diff 对未跟踪文件为空不表示无改动，因此 evidence.json 同时保存所有新源码/配置哈希，交付包包含源文件快照。
Git={audit['provenance']['git_commit']}；官方环境={audit['provenance']['environment_commit']}。V2和官方 vendor无改动，原始725录像和旧实验保留，最终holdout未执行。
MLflow：build={stats['mlflow_run_id']}；E0={audit['mlflow_run_id']}；interface={smoke['mlflow_run_id']}。终端详细日志单独保留，不以显存打印当作实验完成。

{handoff}
"""
    (out/'research_report.md').write_text(report)
    (out/'runbook.md').write_text(runbook())
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        axes[0].bar(MODES,[counts[m]/valid for m in MODES]);axes[0].set_title('Clean label distribution')
        axes[0].tick_params(axis='x',rotation=30)
        axes[1].bar(MODES,old['recall']);axes[1].set_ylim(0,1);axes[1].set_title('Legacy mode recall (trained)')
        axes[1].tick_params(axis='x',rotation=30)
        image=axes[2].imshow(np.log10(np.asarray(audit['capacity']['current_trained']['gradient_map'])+1e-12),origin='lower')
        axes[2].set_title('Legacy EXPLORE input gradient (one state)');fig.colorbar(image,ax=axes[2],label='log10 normalized mass')
        fig.tight_layout();fig.savefig(out/'audit_overview.png',dpi=160);plt.close(fig)
    except ImportError:
        pass
    print(json.dumps({'output':str(out),'completed_controlled_runs':evidence['formal_completed_runs'],'readiness':'NOT READY'}))


if __name__=='__main__':main()
