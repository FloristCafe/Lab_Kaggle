"""Experiment 0: existing filtered states, memory mismatch and measured gradients."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.config import load_config
from agents.goal_review.model import ReviewActor
from agents.strategic.data import history_at,unpack
from agents.strategic.experiment import digest,provenance,save_json,setup
from agents.strategic.features import MODES


def filtered_audit():
    folder=ROOT/'data/dataset_v2_goal'
    manifest=json.loads((folder/'manifest.json').read_text())
    reasons=Counter();samples=[];prev=Counter();explore=Counter()
    for name,row in manifest['files'].items():
        with np.load(folder/name,allow_pickle=False) as z:
            active,valid=z['active'],z['label_valid']
            filtered=active&~valid
            why=np.full(active.shape,'other',object)
            events=z['semantic_events'];positions=z['positions'];mode=z['mode']
            regions=z['goal_regions'];uv=z['unit_vectors']
            segments=json.loads((folder/name).with_suffix('.segments.json').read_text())
            for sid,s in enumerate(segments):
                sl=(slice(s['start'],s['end']+1),s['unit_id'])
                scores=np.asarray(s['mode_scores']);p=np.exp(scores-scores.max());p/=p.sum()
                if s['right_censored']:reason='death_right_censored'
                elif s['end']-s['start']+1<6:reason='insufficient_duration'
                elif not regions[sid].any():reason='no_stable_anchor'
                elif s['confidence']<.45 or np.sort(p)[-1]-np.sort(p)[-2]<.12:reason='ambiguous_intent'
                elif (positions[s['start']:s['end']+1,s['unit_id']]==positions[s['start'],s['unit_id']]).all():reason='stationary_waiting'
                else:reason='after_anchor_or_other'
                why[sl]=reason
            reasons.update(why[filtered].tolist())
            prev['alive_units']+=int(active.sum());prev['filtered_units']+=int(filtered.sum())
            prev['environment_frames']+=len(active);prev['frames_with_any_goal']+=int(valid.any(1).sum())
            prev['frames_with_any_filtered_unit']+=int(filtered.any(1).sum())
            prev['frames_with_alive_but_no_goal']+=int((active.any(1)&~valid.any(1)).sum())
            prev['valid_previous']+=int(((uv[:,:,11]>.5)&active).sum())
            for i,m in enumerate(MODES):prev['previous_'+m]+=int(((uv[:,:,12+i]>.5)&active).sum())
            e=valid&(mode==1)
            explore['labels']+=int(e.sum());explore['games']+=bool(e.any())
            explore['with_nearby_friends']+=int((e&(events[:,:,17]>0)).sum())
            explore['with_global_rank_gain']+=int((e&(events[:,:,23]>0)).sum())
            ids=np.argwhere(e)
            if len(samples)<20 and len(ids):
                t,u=map(int,ids[0]);sid=int(z['segment_id'][t,u])
                samples.append({'source':name,'unit':u,'step':t,'position':positions[t,u].tolist(),
                    'goal':z['goal'][t,u].tolist(),'segment':segments[sid],
                    'human_mode':None,'human_goal':None,'human_valid':None,
                    'annotation_boundary':'inspect legal observed trajectory only; blind to model predictions'})
    assert sum(reasons.values())==1482851
    return {'filtered_primary_proxy_reasons':dict(reasons),'interpretation':'exclusive diagnostics, NOT human semantic ground truth; tactical avoidance not identifiable',
            'previous_context':dict(prev),'explore':dict(explore),'explore_examples':samples}


def capacity_audit(config):
    val=json.loads((ROOT/'experiments/B_strategic_goal_il/manifest.json').read_text())['validation_files']
    sample=None
    for name in val:
        with np.load(ROOT/'data/dataset_v2_goal'/name) as z:
            ids=np.argwhere(z['label_valid']&(z['mode']==1))
            if not len(ids):continue
            t,u=map(int,ids[0]);d=unpack(z)
            sample=(history_at(d['features'],d['steps'],d['match_steps'],t),d['positions'][t],d['unit_vectors'][t],d['active'][t],u,name,t)
            break
    if sample is None:raise ValueError('no EXPLORE validation probe')
    h,p,v,a,u,name,t=sample;results={}
    for arch in ('current','res6','res8','dilated6','current_trained'):
        torch.manual_seed(42)
        cfg={**config['model'],'architecture':'current' if arch=='current_trained' else arch,'replan':False}
        model=ReviewActor(cfg).cuda().eval()
        if arch=='current_trained':model.load_state_dict(torch.load(ROOT/'experiments/B_strategic_goal_il/best_model.pth',map_location='cpu',weights_only=False)['actor_state_dict'])
        history=torch.tensor(h[None],device='cuda',requires_grad=True)
        positions=torch.tensor(p[None].astype(np.int64),device='cuda')
        units=torch.tensor(v[None].astype(np.float32),device='cuda')
        alive=torch.tensor(a[None],device='cuda')
        _,ml,_=model(history,positions,units,alive)
        gradient=torch.autograd.grad(ml[0,u,1],history)[0].abs().sum((0,1,2)).detach().cpu().numpy()
        mass=gradient/max(float(gradient.sum()),1e-30)
        coords=np.stack(np.indices((24,24)),-1);radius=np.abs(coords-p[u]).sum(-1)
        cumulative=np.array([mass[radius<=r].sum() for r in range(47)])
        convs=[m for m in model.modules() if isinstance(m,torch.nn.Conv2d)]
        bins=np.array([mass[(coords[:,:,0]%2==i)&(coords[:,:,1]%2==j)].mean() for i in (0,1) for j in (0,1)])
        results[arch]={'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'leaf_layers':sum(not list(m.children()) for m in model.modules()),'conv_modules':len(convs),
            'linear_modules':sum(isinstance(m,torch.nn.Linear) for m in model.modules()),
            'channel_widths':sorted(set(m.out_channels for m in convs)),
            'spatial_path':'stem + 2 downs + 2 ups' if arch.startswith('current') else ('6 residual blocks' if arch!='res8' else '8 residual blocks'),
            'temporal_unroll':3,'query_dimension':32,'normalization':'GroupNorm, spatial/global coupling',
            'activation':'SiLU; ConvGRU sigmoid/tanh','downsampling_stages':2 if arch.startswith('current') else 0,
            'theoretical_support':'full 24x24 due GroupNorm; current also has explicit global mean',
            'effective_gradient_radius_80':int(np.searchsorted(cumulative,.8)),
            'effective_gradient_radius_95':int(np.searchsorted(cumulative,.95)),
            'gradient_mass_within_3':float(mass[radius<=3].sum()),
            'tiny_gradient_fraction':float((mass < mass.max()*1e-4).mean()),
            'parity_sublattice_density_cv':float(bins.std()/max(bins.mean(),1e-30)),
            'weight_status':'trained legacy' if arch=='current_trained' else 'random initialization, NOT a trained ablation result',
            'gradient_map':mass.tolist(),'probe':{'file':name,'step':t,'unit':u,'target':'EXPLORE mode logit'}}
        del model,history,positions,units,alive,ml
    return results


def clean_audit(folder):
    m=json.loads((folder/'manifest.json').read_text())
    if not m.get('complete'):return {'status':'PENDING_DATASET_COMPLETION'}
    transitions=np.zeros((6,6),int);bad_event=0;durations=[];modecounts=np.zeros(5,int)
    previous_count=0;previous_modes=np.zeros(5,int);previous_ages=[];alive_units=0
    for name,row in m['files'].items():
        path=folder/name
        assert digest(path)==row['npz_sha256']
        with np.load(path,allow_pickle=False) as z:
            bad_event+=int(((z['semantic_events'][:,:,21]>.5)&~z['sap_legal']).sum())
            old=np.where(z['dirty_label_valid'],z['dirty_mode'],5)
            new=np.where(z['label_valid'],z['mode'],5)
            active=z['active'];np.add.at(transitions,(old[active],new[active]),1)
            vectors=z['unit_vectors'].astype(np.float32);pv=(vectors[:,:,11]>.5)&active
            previous_count+=int(pv.sum());alive_units+=int(active.sum())
            previous_modes+=np.bincount(vectors[pv,12:17].argmax(1),minlength=5)
            previous_ages.extend((vectors[pv,10]*100).tolist())
            if name==next(iter(m['files'])):
                from agents.goal_review.memory import causal_contexts
                from agents.goal_review.data import unpack_review
                d=unpack_review(z);d['env_cfg']=row['env_cfg']
                prefix={k:(v[:110] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(d['steps']) else v) for k,v in d.items()}
                checked,review_checked,_=causal_contexts(prefix,m['config'])
                np.testing.assert_array_equal(checked,d['unit_vectors'][:110])
                np.testing.assert_array_equal(review_checked,d['review_frames'][:110])
            ids=np.flatnonzero(z['review_frames']);steps=z['steps']
            durations.extend(np.diff(np.r_[0,steps[ids],steps[-1]]).tolist())
            modecounts+=np.bincount(z['mode'][z['label_valid']],minlength=5)
    assert bad_event==0
    durations=np.asarray([x for x in durations if x>0])
    return {'complete':True,'games':len(m['files']),'split_counts':dict(Counter(r['split'] for r in m['files'].values())),
            'invalid_sap_used_as_event':bad_event,'dirty_to_clean_transition_6x6':transitions.tolist(),
            'transition_axes':list(MODES)+['FILTERED'],'mode_counts':modecounts.tolist(),
            'context_prefix_equal_110':True,
            'previous_context':{'valid_fraction':previous_count/max(1,alive_units),
                'mode_counts':previous_modes.tolist(),'mean_age':float(np.mean(previous_ages)),
                'median_age':float(np.median(previous_ages)),'p90_age':float(np.quantile(previous_ages,.9))},
            'scripted_review_option_duration':{'mean':float(durations.mean()),'median':float(np.median(durations)),
                'p90':float(np.quantile(durations,.9)),'p95':float(np.quantile(durations,.95))},
            'data_statistics':m['statistics'],'dataset_manifest_sha256':digest(folder/'manifest.json')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'data/dataset_v3_goal_review')
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/goal_review/E0_audit');args=p.parse_args()
    setup();args.output_dir.mkdir(parents=True,exist_ok=True)
    from ml_workspace.tracking import tracked_run
    import mlflow
    with tracked_run('E0_audit',experiment_name='lux-s3-goal-review',params={'operation':'audit_no_training','seed':42}) as run:
        config=load_config(ROOT/'configs/goal_review/common.yaml')
        former=filtered_audit();capacity=capacity_audit(config);clean=clean_audit(args.data_dir)
        old=json.loads((ROOT/'experiments/B_strategic_goal_il/summary.json').read_text())
        result={'provenance':provenance(),'current_implementation_audit':former,'capacity':capacity,'clean_dataset':clean,
                'legacy_mode_confusion':old['metrics']['mode_confusion'],'legacy_explore_predicted_count':0,
                'legacy_online_reference':json.loads((ROOT/'experiments/CDE_goal_mappo_v2/il_only/summary.json').read_text())['evaluations'],
                'causal_claim':'No controlled training run executed; imbalance/label/representation/mismatch remain competing hypotheses',
                'readiness':'NOT READY','mlflow_run_id':run.info.run_id}
        save_json(args.output_dir/'summary.json',result)
        save_json(args.output_dir/'human_annotation_queue.json',former.pop('explore_examples'))
        mlflow.log_artifacts(str(args.output_dir))
        print(json.dumps({'output':str(args.output_dir),'readiness':'NOT READY','clean':clean.get('complete',False)}),flush=True)


if __name__=='__main__':main()
