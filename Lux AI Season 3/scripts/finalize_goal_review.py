"""Synthesize existing artifacts only: no model forward, optimizer or environment."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.strategic.experiment import digest,save_json
from agents.strategic.features import MODES

BASE=ROOT/'experiments/goal_review'
ARMS=('E1_clean','E1_dirty_control','E2_weighted','E2_balanced','E3_res6','E3_res8',
      'E3_dilated','E4_without_r','E4_with_r','E5_corrupted','E6_candidate')


def read(path):return json.loads(Path(path).read_text())


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    out=BASE/'final_research_report';out.mkdir(exist_ok=True)
    runs={arm:read(BASE/f'{arm}_seed42/summary.json') for arm in ARMS}
    assert all(s['status']=='FINISHED' and not s['smoke_only'] for s in runs.values())
    evidence=read(BASE/'research_report/evidence.json')
    stats=read(ROOT/'data/dataset_v3_goal_review/summary.json')
    audit=read(BASE/'E0_audit/summary.json')
    baseline=read(BASE/'causal_baseline_audit/summary.json')
    support=read(BASE/'replan_support_audit/summary.json')
    aliasing=read(BASE/'query_aliasing_audit/summary.json')
    selection=read(BASE/'autonomous_progress/candidate_selection.json')
    studies={key:read(BASE/f'{folder}/summary.json') for key,folder in (
        ('without_r','E5_memory_without_r'),('with_r','E5_memory_with_r'),
        ('corrupted','E5_memory_corrupted'),('candidate','E6_memory_candidate'))}
    plans=[s['plan'] for s in studies.values()]
    keys=('validation_files','dataset_manifest_sha256','conditions','dev_seed','sides','prefix_steps',
          'batch_games','tf32','source_sha256','evaluation_sources','runtime_memory_sha256')
    checks={k:all(p[k]==plans[0][k] for p in plans) for k in keys}
    checks['full_validation']=len(plans[0]['validation_files'])==145
    checks['FP32']=all(s['inference_precision']=='FP32' for s in studies.values())
    checks['no_training_in_studies']=all(s['training_updates']==0 for s in studies.values())
    manifest=read(ROOT/'data/dataset_v3_goal_review/manifest.json')
    checks['dataset_producers']=all(digest(ROOT/name)==expected for name,expected in manifest['pipeline_sha256'].items())
    checks['runtime_unchanged']=digest(ROOT/'agents/goal_review/runtime.py')==plans[0]['runtime_memory_sha256']
    checks['evaluation_unchanged']=all(digest(ROOT/'scripts'/name)==expected for name,expected in plans[0]['evaluation_sources'].items())
    checks['study_wrapper_unchanged']=digest(ROOT/'scripts/evaluate_goal_review_study.py')==plans[0]['source_sha256']
    checks['study_checkpoint_matches']=all(s['plan']['checkpoint_sha256']==digest(BASE/f'{arm}_seed42/best_model.pth')
        for arm,s in zip(('E4_without_r','E4_with_r','E5_corrupted','E6_candidate'),studies.values(),strict=True))
    assert all(checks.values()),checks
    run_rows=[];mode_rows=[];goal_rows=[];memory_rows=[];online_rows=[]
    for arm,s in runs.items():
        m=s['metrics'];r=m['free_replan'];enabled=s['config']['model']['replan']
        run_rows.append({'arm':arm,'scope':s['config']['study']['scope'],'context':s['config']['study']['context'],
            'best_epoch':s['best_epoch'],'stopped_epoch':s['stopped_epoch'],'mode_accuracy':m['mode']['accuracy'],
            'macro_f1':m['mode']['macro_f1'],'balanced_accuracy':m['mode']['balanced_accuracy'],
            'joint':m['joint_mode_goal_accuracy'],'oracle_goal_top1':m['goal_top1'],
            'EXPLORE_recall':m['mode']['recall'][1],'EXPLORE_joint':m['per_mode']['EXPLORE']['joint_mode_goal_accuracy'],
            'continue_recall':r['recall'][0] if enabled else None,'false_replan':m['false_replan_rate'] if enabled else None,
            'wall_seconds':s['wall_seconds'],'peak_allocated_mib':s['cuda_memory']['peak_allocated_mib'],
            'peak_reserved_mib':s['cuda_memory']['peak_reserved_mib'],'mlflow_run_id':s['mlflow_run_id']})
        for i,mode in enumerate(MODES):
            mode_rows.append({'arm':arm,'mode':mode,'support':m['mode']['support'][i],
                'predicted':m['mode']['predicted_count'][i],'precision':m['mode']['precision'][i],
                'recall':m['mode']['recall'][i],'f1':m['mode']['f1'][i],**m['per_mode'][mode]})
        goal_rows.append({'arm':arm,**{k:m[k] for k in ('goal_top1','goal_top5','goal_top10','goal_manhattan',
            'goal_entropy','goal_peak','anchor_rate','predicted_goal_distance','expert_region_distance',
            'within_segment_prediction_consistency','goal_persistence','within_segment_pairs')}})
    for arm,s in studies.items():
        assert s['status']=='FINISHED' and not s['smoke_only']
        for eta,values in s['conditions'].items():
            counts=[]
            for condition in ('scripted','corrupted','policy_replay'):
                m=values[condition];h=m['all_frame_held_intent'];r=m['free_replan'];counts.append(h['units'])
                memory_rows.append({'arm':arm,'eta':float(eta),'condition':condition,'units':h['units'],
                    'held_joint':h['joint'],'held_mode_accuracy':h['mode_accuracy'],
                    'held_anchor_rate':h['anchor_prediction_rate'],'review_joint':m['joint_mode_goal_accuracy'],
                    'review_frequency':m['macro_review_frequency'],'coverage':m['goal_label_coverage'],
                    'previous_valid_fraction':m['previous_goal_valid_fraction'],'previous_age':m['previous_age_mean'],
                    'previous_modes':json.dumps(m['previous_mode_counts']),
                    'free_continue_precision':r['precision'][0],'free_continue_recall':r['recall'][0],
                    'free_R1_precision':r['precision'][1],'free_R1_recall':r['recall'][1],
                    'free_R1_f1':r['f1'][1],'false_replan':m['false_replan_rate']})
            assert len(set(counts))==1
            folder=BASE/('E6_memory_candidate' if arm=='candidate' else f'E5_memory_{arm}')/f'eta_{float(eta):g}'
            for p in values['closed_loop']:
                trace=read(folder/f"online_trace_p{p['side']}.json")
                free=free_continue=forced=0
                for t in trace:
                    active=np.asarray(t['start_alive'],bool);valid=np.asarray(t['previous_valid'],bool)
                    r=np.asarray(t['replan'],bool);free+=int((active&valid).sum())
                    free_continue+=int((active&valid&~r).sum());forced+=int((active&~valid).sum())
                online_rows.append({'arm':arm,'eta':float(eta),'side':p['side'],'steps':p['primitive_steps'],
                    'review_frequency':p['review_frequency'],'replan_rate':p['replan_rate'],
                    'free_decisions':free,'free_continue':free_continue,'forced_decisions':forced,
                    'goal_switch_rate':p['goal_switch_rate'],'goal_persistence':p['goal_persistence'],
                    'intent_persistence':p['intent_persistence'],'nontrivial_completion':p['goal_completion_rate'],
                    'completion_events':p['goal_completion_events'],'navigational_intents':p['navigational_intents'],
                    'previous_age':p['mean_goal_age'],'score':p['metrics'].get('score'),
                    'opponent_score':p['metrics'].get('opponent_score'),'invalid_persistence':p['invalid_goal_persistence']})
    assert len({r['units'] for r in memory_rows})==1
    resource=list(csv.DictReader((BASE/'autonomous_progress/resource_samples.csv').open()))
    resource_summary={k:{'min':min(float(r[k]) for r in resource),'max':max(float(r[k]) for r in resource)}
        for k in resource[0] if k!='utc'}
    resource_summary.update(start=resource[0]['utc'],end=resource[-1]['utc'],samples=len(resource),
        scope='sampled interval of four concurrent training runs only; excludes later evaluation and E6')
    summary={'status':'FINISHED','readiness':'NOT READY','formal_RL_updates':0,'training_runs':len(runs),
        'run_rows':run_rows,'mode_rows':mode_rows,'goal_rows':goal_rows,'memory_rows':memory_rows,
        'online_rows':online_rows,'protocol_checks':checks,'resource_samples':resource_summary,
        'candidate_selection':selection,'baseline':baseline,'R_support':support,'query_aliasing':aliasing,
        'dataset_manifest_sha256':digest(ROOT/'data/dataset_v3_goal_review/manifest.json'),
        'source_evidence_sha256':digest(BASE/'research_report/evidence.json'),
        'run_summary_sha256':{a:digest(BASE/f'{a}_seed42/summary.json') for a in runs},
        'study_summary_sha256':{k:digest(BASE/('E6_memory_candidate' if k=='candidate' else f'E5_memory_{k}')/'summary.json') for k in studies},
        'execution_owner':'user','next_experiments':'provide commands to user; no autonomous launches',
        'report_generator_sha256':digest(Path(__file__))}
    save_json(out/'summary.json',summary)
    for name,rows in (('runs',run_rows),('per_mode',mode_rows),('goals',goal_rows),('memory',memory_rows),('online',online_rows)):
        write_csv(out/f'{name}.csv',rows)
    run_table=table(['Arm','scope / memory','Best','Mode Acc','Macro-F1','Balanced Acc','Joint','EXP recall'],[
        [r['arm'],r['scope']+' / '+r['context'],r['best_epoch'],f"{r['mode_accuracy']:.2%}",f"{r['macro_f1']:.4f}",
         f"{r['balanced_accuracy']:.2%}",f"{r['joint']:.2%}",f"{r['EXPLORE_recall']:.2%}"] for r in run_rows])
    mem_table=table(['Arm','η','Memory','Held Joint','review/frame','Continue recall','False Replan'],[
        [r['arm'],r['eta'],r['condition'],f"{r['held_joint']:.2%}",f"{r['review_frequency']:.3f}",
         '—' if r['arm']=='without_r' else f"{r['free_continue_recall']:.2%}",
         '—' if r['arm']=='without_r' else f"{r['false_replan']:.2%}"] for r in memory_rows])
    online_table=table(['Arm','η','Side','Steps','review/frame','Free Continue/N','Goal persist','Completion','score/opponent'],[
        [r['arm'],r['eta'],r['side'],r['steps'],f"{r['review_frequency']:.3f}",f"{r['free_continue']}/{r['free_decisions']}",
         f"{r['goal_persistence']:.2%}",f"{r['nontrivial_completion']:.2%}",f"{r['score']}/{r['opponent_score']}"] for r in online_rows])
    goal_table=table(['Arm','Goal@1','@5','@10','L1','Entropy','Pred distance','Expert distance','Segment consistency'],[
        [r['arm'],f"{r['goal_top1']:.2%}",f"{r['goal_top5']:.2%}",f"{r['goal_top10']:.2%}",f"{r['goal_manhattan']:.3f}",
         f"{r['goal_entropy']:.3f}",f"{r['predicted_goal_distance']:.2f}",f"{r['expert_region_distance']:.2f}",
         f"{r['within_segment_prediction_consistency']:.2%}"] for r in goal_rows])
    costs=table(['Arm','Wall min','Peak allocated MiB','Peak reserved MiB','MLflow'],[
        [r['arm'],f"{r['wall_seconds']/60:.2f}",f"{r['peak_allocated_mib']:.1f}",f"{r['peak_reserved_mib']:.1f}",r['mlflow_run_id']] for r in run_rows])
    cap=table(['Architecture','Parameters','Conv modules','Widths','Downsample','R80 / R95'],[
        [k,v['parameters'],v['conv_modules'],v['channel_widths'],v['downsampling_stages'],
         f"{v['effective_gradient_radius_80']:.2f} / {v['effective_gradient_radius_95']:.2f}"]
        for k,v in audit['capacity'].items()])
    candidate=runs['E6_candidate']['metrics']
    perclass=table(['Mode','Support','Predicted','Precision','Recall','F1','Goal@1','Joint'],[
        [m,candidate['mode']['support'][i],candidate['mode']['predicted_count'][i],f"{candidate['mode']['precision'][i]:.2%}",
         f"{candidate['mode']['recall'][i]:.2%}",f"{candidate['mode']['f1'][i]:.4f}",
         f"{candidate['per_mode'][m]['goal_top1']:.2%}",f"{candidate['per_mode'][m]['joint_mode_goal_accuracy']:.2%}"] for i,m in enumerate(MODES)])
    confusions='\n\n'.join(f"### {a}\n\nMode head (true rows, predicted columns; SCORE/EXPLORE/DEFEND/INTERCEPT/RECHARGE)\n\n```text\n{np.asarray(s['metrics']['mode']['confusion_matrix'])}\n```\n\nFree R (Continue/Replan; {'learned head' if s['config']['model']['replan'] else 'constant always-Replan control'})\n\n```text\n{np.asarray(s['metrics']['free_replan']['confusion_matrix'])}\n```" for a,s in runs.items())
    (out/'confusion_matrices.md').write_text(confusions)
    (out/'research_report.md').write_text(f"""# Goal-Level IL：标签、容量、Replan 与部署记忆受控研究

## 1. 实验假设与核心矛盾

本轮分别检验严格标签、类别梯度、空间主干和意图记忆是否改善 Goal IL，再按事前登记规则组合候选。**结论：NOT READY；E6 组合未解决 EXPLORE 与持续意图缺失，暂不进入正式 MAPPO。**
已完成 11 个正式 IL run（同 seed42），4 个完整145场冻结记忆研究、14个双侧/η组合的实际短前缀；没有增加下载、最终留出集或正式RL训练。后续实验由用户启动。

## 2. 数据与防穿越边界

725场胜方合法视角，580训练/145验证；按Match ID划分，已知source seeds均唯一且跨集合不相交，旧80验证ID保持验证。4,202,800存活单位状态，1,293,473有效Goal，过滤{stats['invalid_samples']/stats['total_samples']:.2%}；有效类计数={stats['mode_counts']}。EXPLORE仅1.11%，INTERCEPT71.73%。过滤状态没有被默认标成Continue，实际恢复filtered Continue=0。
输入为≤T的公开观测，previous intent只来自<T的规则/策略输出；标签可使用>T的专家轨迹推导语义段、区域和纠错证据，不能回流输入。事件分段前剔除8,021次非法SAP，合法148,151次；非法SAP派生宏事件审计为0。死亡、截尾、证据不足不伪造Goal。
数据、标签规则、memory生产器的8项哈希一致，manifest={summary['dataset_manifest_sha256']}。标签仍是启发式推断，没有人工意图真值；source seed隔离也不保证地图结构/目标分布完全新颖。
E1/E2/E3验证64,414单位（每4帧采样）；E4/E5/E6验证105,267单位（review epochs），两协议的绝对差值不能归因于单项改动。E5/E6 corrupted配置同时改变训练与选模validation的记忆上下文，不能将原始summary之间的差异解释为纯训练corruption因果效应。以下共用FP32冻结评估缓解评估条件差异，但不消除选模条件差异。
同一145场验证被反复用于选epoch、选组件、选η，没有独立测试、种子复现或显著性推断。E6是验证选择的探索性组合。

## 3. 特征构建与数学表达

每帧18个因果物理通道，三帧分组编码与ConvGRU(hidden32)融合；单位22维状态形成query32。输出 mode[B,16,5]、goal[B,16,5,24,24]及独立Bernoulli r。π(r,m,g|s,Iprev)：r=0保持原(mode,goal)，r=1联合选择新(mode,goal)；无有效previous强制r1。Continue logπ仅包含logP(r0)，Replan额外包含mode和goal概率。没有577类KEEP。
训练为confidence-weighted mode CE + expert-mode区域NLL +自由previous上的R BCE；r0样本不为新mode/goal贡献梯度。三项E2仅改变mode梯度分配/单位采样，loss权重上限5。E3保持普通loss，只替换空间主干；移除下采样也改变全局汇聚路径，不能等同纯参数量实验。

{cap}

current为stem+两次下采样/上采样；res6/res8各6×64/8×128残差块；dilated6为6×64、d=[1,2,3,5,8,4]。GroupNorm和current全局均值使理论信息支持可全图；不能由局部卷积公式断言模型看不到对角。梯度半径为单状态局部敏感性，除current_trained外为初始化权重诊断，不能当成已训练容量胜负。
bounded hysteresis仅对上一个(mode,goal)联合对加η≤0.5，80步衰减为0、100步使memory失效。η不改变训练loss/梯度，因此E5_hysteresis与combination按同checkpoint的η=0/.25推理对照完成，没有重复训练相同目标。
执行器修正：同目标r1不再重置age、blocked、reached；新目标/失效目标仍正常初始化。冻结scripted规则只在目标改变/失效时r1，故该修正不改变已生成特征和标签。133项单元测试、真实前缀特征一致性、240帧逐流/分批一致性通过。

## 4. 离线评估与指标对比

### 正式受控训练

{run_table}

选模=mode-head Macro-F1，joint只破平局；每轮1,000更新×16单位，最多6轮、patience5。所有run跑到6轮预算上限，这不是充分收敛证据。每个run96,000单位曝光，远少于全库129万有效状态。mode-head argmax与执行joint argmax的mode不同，分别记录。
严格clean未获得对dirty的Macro-F1优势（0.4737 vs0.5141），joint略高（19.14% vs18.18%）；清理同时改变样本和分段，且无人工语义真值，不能宣称清理已验证正确。dirty仅隔离对照，不进入候选。
E2 weighted提高Macro-F1到0.5281，joint仅净增2/64,414；balanced把EXPLORE召回从0推到43.25%，但precision5.94%、joint降到17.85%。不平衡对模式塌缩有直接干预证据，重采样仍未解决目标质量。
E3 dilated与current的joint差仅7/64,414（19.148% vs19.137%）；res6/res8更差，全部EXPLORE召回为0。事前规则机械选dilated为E6组件，不是统计上可靠的架构胜出。

### Goal细定位（给定专家mode）

{goal_table}

Goal TopK按真实模式的条件目标头评分，Joint同时要求执行模式与目标区域正确，不能混为同一Accuracy。动态未知地图geodesic未定义，保持null。段内一致性/离线persistence来自采样时点，不能冒充真实闭环保持。
同协议因果基线：给定专家mode时当前坐标命中19.80%，E1 Goal@1=22.87%；previous目标8.46%，因果previous完整意图joint7.90%。当前坐标是有效SCORE/DEFEND/RECHARGE常见目标，不应全部当垃圾；该oracle-mode基线不可直接和模型joint比较。平均目标区域仅1.013格，除SCORE外四类均单格，INTERCEPT专家目标距当前均值12.48格，其Goal@1仍仅约5%，精确单格伪标签可识别性是待解决瓶颈。

### 冻结同权重的A/B/C记忆研究

所有行共有257,305个有效单位状态。review时发出意图，其他帧保持，在全部clean物理帧计分；分母不随策略review改变。A=因果scripted，B=标签盲corrupted快照，C=Actor历史输出回流memory。A/B输出保持仅用于计分，输入memory各自固定。C仍沿用专家物理轨迹，因此不是环境闭环。

{mem_table}

E4配对加R后，规则条件Continue召回79.49%、False Replan20.51%，但固定帧joint从18.01%降到16.22%；C进一步从without-R16.97%降到with-R12.51%。R分类学会了一部分规则兼容性，未解决部署错配。
E5 corruption使C joint回升到16.77%，仍未超过without-R16.97%，且C Continue召回仅14.94%。A→C退化仍存在，不能宣称解决exposure bias。η=.25对A/B有微小帮助，对C普遍恶化；E6事前选择η=0。E6本身η=.25的C也更差（17.07% vs17.16%）。
E6选择={selection['architecture']}、{selection['loss']}、{selection['context_source']}、R=True、η={selection['hysteresis']}。其C fixed-frame joint17.16%略高without-R16.97%，同时**最佳checkpoint自由R完全始终Replan、Continue召回0**，不能作为最终成功候选；组件优点未必可叠加。

### E6逐类结果与误差

{perclass}

原始选模105,267单位下mode82.88%、Macro-F1 .5104、balanced55.10%、joint17.92%；EXPLORE head predicted=0，执行joint模式会少量选EXPLORE但其正确目标joint仍0。完整11组逐类、混淆矩阵、R矩阵见per_mode.csv和confusion_matrices.md；冻结条件全量JSON保留。

### 实际dev短闭环（seed15001）

{online_table}

每行预算128 primitive步，当前option结束后实际129步，均非终局；score只为前缀计数，不是整局胜率。所有14前缀的自由previous决策也没有Continue，不能归咎于全部强制r1。目标身份持久性与r0不同，r1重新选择相同坐标同样会保持目标。非原地首次抵达率通常<3%，E6η0双侧1/172和4/184；前缀短、单dev种子且已有使用，不能外推整局质量。在线没有专家R真值，precision/recall/F1=null。Invalid persistence=0由协议保证，不是学习成绩。

### Q1–Q8研究结论

| 问题 | 结论及边界 |
|---|---|
| Q1 EXPLORE=0主因 | **类别不平衡是已实证的主要贡献因素**：同网络balanced把recall推到43.25%，排除完全无法表达EXPLORE。precision仅5.94%、E6仍0，标签归因/状态混淆/记忆支持缺口仍可能重要，无人工意图真值不能确定唯一主因。 |
| Q2 spatial encoder过浅 | **未证明是主要瓶颈**：当前已有全图信息支持，放大res6/res8没有改善，dilated仅7个正确样本优势；有限预算不能证明更大架构长期无效。固定32维temporal/query、目标单格噪音仍需分离验证。 |
| Q3 filtered造成部署错配 | **错配实测存在，不能全部归因过滤率**。新过滤69.22%，恢复filtered Continue=0；规则训练previous从不DEFEND/INTERCEPT，真实Actor会产生这些mode。训练自由R0样本SCORE27578、EXPLORE11、DEFEND0、INTERCEPT0、RECHARGE1705，支持域明显不足。offline A约.390 review/frame、C约.398–.404，实际prefix约.256–.341且状态不同；差值不是过滤的单变量因果效应。 |
| Q4 R真正解决了吗 | **没有**。A下R分类改善，但完整意图joint降低，C降幅更大；真实prefix有有效previous也全r1，E6又退化为常量r1。R1高F1主要受正类占比支配。 |
| Q5 Previous Goal exposure bias | **有直接分布差异与冻结干预证据**：with-R A16.22%→C12.51%，Continue79.49%→44.87%；C含previous DEFEND/INTERCEPT而A均0。corruption缓解joint退化但未带来可靠Continue；C并非互动rollout，尚不能归因全部在线损失。 |
| Q6 accuracy86.20%误导吗 | **若理解为五类均已学会则误导**。E1 mode87.04%但Macro-F1 .4737、balanced45.78%、EXPLORE0；balanced提高少数类召回却降低joint。须同时报告逐类、两种mode定义及joint。 |
| Q7 学会长期战略吗 | **证据不足**。E6预测距离6.97vs专家9.67格、段内一致性50.78%；目标精确单格占主导，INTERCEPT Goal@1约5%，真实非平凡完成低，且全r1。不能仅凭长语义段或预测离当前较远就宣称长期规划，也不能仅凭anchor率断言恒等映射。精确相同query冲突仅3/64,414不可避免joint错误，不能解释总体失败；这不证明表示充分。 |
| Q8 正式MAPPO | **NOT READY**。阻塞项：EXPLORE精确意图仍失效、规则R0支持不足、memory反馈鲁棒性差、真实prefix无Continue且目标完成低、标签语义无人审。R-aware行为概率/SMDP接口sanity通过，但正式R-aware rollout buffer与PPO optimizer尚未接完；旧MAPPO入口不能直接接新checkpoint。 |

优先修复方向是合法、标签盲且具DEFEND/INTERCEPT持续状态覆盖的memory行为策略与兼容监督，另做分层意图盲审来评估单格标签噪音；当前结果不支持继续盲目加宽网络或直接用RL掩盖标签问题。**本报告没有启动这些后续实验；实现完成后将给用户可执行指令。**

## 5. 性能瓶颈与物理开销

{costs}

后四组受控训练并行共享GPU，其wall不能与前六组独占运行直接做吞吐比较；E6随后独立运行。30秒设备采样仅覆盖四组并行区间：{resource_summary['start']}至{resource_summary['end']}，共{len(resource)}点，GPU总显存峰值{resource_summary['gpu_memory_mib']['max']:.0f}MiB，温度最高{resource_summary['temperature_c']['max']:.0f}°C，host available最低{resource_summary['host_available_mib']['min']:.0f}MiB。单进程peak allocated与整卡进程外开销不同；后续评估/E6无连续资源采样，不能把该采样冒充全流程峰值，能耗未独占归因。
数据NPZ277.65MiB、JSON106.97MiB；读取LRU1 shard、按shard组织样本，无全库展开。725原始录像与旧数据保留，无删除与下载。冻结研究单条件按游戏16流分批且各流时间顺序独立，输入FP32/TF32关闭，所有完整验证145场，额外训练更新数0；对应wall和MLflow见summary.json，原始输出保留。
Git={audit['provenance']['git_commit']}；官方环境={audit['provenance']['environment_commit']}；V2与vendor未修改。训练/评估checkpoint、源码、配置、数据hash均独立保存；源码未提交，不能只凭HEAD复现，必须使用交付快照。133项单元测试、静态检查与diff检查已通过，软件一致性不等同泛化质量。
此前授权自动批次已结束，当前执行归属已恢复为用户；自动接续入口检查execution_owner，当前拒绝agent继续启动。旧授权plan单独归档，完成产物不删除、不覆写、不重新标成pending。
""")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    labels=[r['arm'] for r in run_rows];x=np.arange(len(labels))
    axes[0].plot(x,[r['macro_f1'] for r in run_rows],'o-',label='mode Macro-F1')
    axes[0].plot(x,[r['joint'] for r in run_rows],'o-',label='review Joint')
    axes[0].set_xticks(x,labels,rotation=65,ha='right');axes[0].legend();axes[0].set_title('Different scopes marked in report')
    for arm in studies:
        rows=[r for r in memory_rows if r['arm']==arm and r['eta']==0]
        axes[1].plot([0,1,2],[r['held_joint'] for r in rows],'o-',label=arm)
    axes[1].set_xticks([0,1,2],['scripted','corrupted','policy memory']);axes[1].legend();axes[1].set_title('Fixed denominator: 257,305 unit states')
    for arm in ('with_r','corrupted','candidate'):
        rows=[r for r in memory_rows if r['arm']==arm and r['eta']==0]
        axes[2].plot([0,1,2],[r['free_continue_recall'] for r in rows],'o-',label=arm)
    axes[2].set_xticks([0,1,2],['scripted','corrupted','policy memory']);axes[2].set_ylim(0,1)
    axes[2].legend();axes[2].set_title('Free Continue recall; replay only')
    fig.tight_layout();fig.savefig(out/'study_overview.png',dpi=180);plt.close(fig)
    print(json.dumps({'output':str(out),'status':'FINISHED','training_runs':len(runs),'readiness':'NOT READY',
                      'additional_training_updates':0,'model_forward_passes':0,'environment_steps':0}))


if __name__=='__main__':main()
