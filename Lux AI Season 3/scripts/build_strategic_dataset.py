"""Local raw JSON -> causal features and event-based strategic goals; never burn raw."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.strategic.data import MASKS, VALUES
from agents.strategic.experiment import digest, provenance, save_json, setup
from agents.strategic.features import CausalObservationBuilder, MODES, causal_context_update, unit_vectors
from agents.strategic.labeler import EVENT_NAMES, StrategicGoalLabeler
from scripts.replay_parser import _as_action, _as_obs
from scripts.stream_replays import select_local_games


def parse_raw(path,side,config):
    raw = json.loads(path.read_text())
    cfg = raw['configuration']['env_cfg']
    builder = CausalObservationBuilder(side,cfg)
    T = len(raw['steps'])
    result = dict(features=[],positions=[],energies=[],active=[],enemy_positions=[],enemy_energies=[],
                  enemy_active=[],steps=[],match_steps=[],team_points=[],rank=[],unit_vectors=[])
    goals = np.zeros((16,2),np.int16)
    modes,ages,valid = np.zeros(16,np.int8),np.zeros(16,np.int16),np.zeros(16,bool)
    for t,records in enumerate(raw['steps']):
        frame = builder.observe(_as_obs(records[side]))
        ep,ee,ea = np.full((16,2),-1,np.int16),np.zeros(16,np.int16),np.zeros(16,bool)
        for uid,xy,e in frame['enemy']:
            ep[uid],ee[uid],ea[uid] = xy,e,True
        values = dict(features=frame['base'],positions=frame['positions'],energies=frame['energy'],active=frame['alive'],
                      enemy_positions=ep,enemy_energies=ee,enemy_active=ea,steps=frame['step'],match_steps=frame['match_step'],
                      team_points=frame['points'],rank=frame['rank'],
                      unit_vectors=unit_vectors(frame,goals,modes,ages,valid,cfg).astype(np.float16))
        for k,value in values.items():
            result[k].append(value)
        causal_context_update(frame,goals,modes,ages,valid,config['tactical'])
    result = {k:np.asarray(v) for k,v in result.items()}
    # Recorded action T+1 is applied from observation T, not T+1.
    result['actions'] = np.stack([_as_action(raw['steps'][min(t+1,T-1)][side],16)
                                  if t<T-1 else np.zeros((16,3),np.int16) for t in range(T)])
    result['team_id'] = side
    result['env_cfg'] = cfg
    result['source_seed'] = raw['configuration'].get('seed')
    return result


def statistics(data,segments):
    valid = data['label_valid']
    distances = np.abs(data['positions'][valid]-data['goal'][valid]).sum(1)
    return dict(total_samples=int(data['active'].sum()),valid_samples=int(valid.sum()),
                invalid_samples=int(data['active'].sum()-valid.sum()),
                mode_counts={name:int(((data['mode']==m)&valid).sum()) for m,name in enumerate(MODES)},
                confidence_hist=np.histogram(data['label_confidence'][valid],bins=np.linspace(0,1,11))[0].tolist(),
                goal_distance_hist=np.bincount(distances,minlength=47).tolist(),
                segment_duration_hist=np.bincount([s['end']-s['start']+1 for s in segments],minlength=102).tolist(),
                valid_segment_duration_hist=np.bincount([s['end']-s['start']+1 for s in segments if s['valid']],minlength=102).tolist(),
                segments=len(segments),valid_segments=sum(s['valid'] for s in segments),
                censored_segments=sum(s['right_censored'] for s in segments))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=400)
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw_replays')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'data/dataset_v2_goal')
    parser.add_argument('--reuse-dir',type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/strategic_baseline.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    selected = select_local_games(ROOT/'data/kaggle_downloads/86411/selection_top25_wins45_scan200.json',args.raw_dir,args.limit,25)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    setup()
    from ml_workspace.tracking import tracked_run
    import mlflow
    labeler = StrategicGoalLabeler(config['labeler'])
    manifest_path = args.output_dir/'manifest.json'
    source = provenance()
    config_hash = digest(args.config)
    pipeline_hashes = {name:digest(ROOT/name) for name in (
        'agents/strategic/features.py','agents/strategic/labeler.py','agents/strategic/data.py',
        'scripts/build_strategic_dataset.py')}
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and (previous['config_sha256']!=config_hash or previous.get('pipeline_sha256')!=pipeline_hashes):
        raise ValueError('configuration changed; use a new dataset directory')
    manifest = dict(dataset_version='dataset_v2_goal',schema_version=4,feature_version='causal_grouped_v2',
                    config_sha256=config_hash,pipeline_sha256=pipeline_hashes,config=config,provenance=source,files=previous.get('files',{}),
                    event_names=EVENT_NAMES,observation_source='steps[side].observation.obs only',
                    feature_cutoff='prefix<=T; no teacher future goals in previous context',
                    label_source='future event segments, no 12-step mode or Gaussian',
                    previous_context='causal scripted context, not expert segment goals',
                    views='top25 locally available strict winning side',
                    replay_action_alignment='record T+1 action is outbound at observation T',
                    old_corpus_overlap=len(set(selected)&{p.stem.rsplit('_p',1)[0] for p in (ROOT/'data/processed_replays_unitcentric25').glob('*.npz')}),
                    selected_games=list(selected))
    with tracked_run(f'dataset_v2_goal_{args.limit}',experiment_name='lux-s3-strategic-baseline',params={'games':args.limit,'seed':42}) as run:
        for index,(key,entry) in enumerate(selected.items()):
            raw_path = args.raw_dir/f'{key}.json'
            raw_hash = digest(raw_path)
            expected_raw = entry['episode'].get('expected_raw_sha256')
            if expected_raw and expected_raw!=raw_hash:
                raise ValueError('raw file differs from downloaded source hash')
            side = entry['winner_side']
            name = f'{key}_p{side}.npz'
            output = args.output_dir/name
            if name in manifest['files']:
                if digest(output)!=manifest['files'][name]['npz_sha256']:
                    raise ValueError('changed cached shard')
                continue
            reuse_manifest = args.reuse_dir/'manifest.json' if args.reuse_dir else None
            reused = False
            if reuse_manifest and reuse_manifest.exists():
                rm = json.loads(reuse_manifest.read_text())
                row = rm['files'].get(name)
                if (row and rm['config_sha256']==config_hash and rm.get('pipeline_sha256')==pipeline_hashes
                        and row['raw_sha256']==raw_hash):
                    assert digest(args.reuse_dir/name)==row['npz_sha256']
                    shutil.copyfile(args.reuse_dir/name,output)
                    shutil.copyfile((args.reuse_dir/name).with_suffix('.segments.json'),output.with_suffix('.segments.json'))
                    manifest['files'][name] = row
                    reused = True
            if not reused:
                data = parse_raw(raw_path,side,config)
                labels,segments = labeler.label(data)
                data.update(labels)
                stat = statistics(data,segments)
                features = data.pop('features')
                meta = dict(source_match=key,team_id=side,source_seed=data.pop('source_seed'),env_cfg=data.pop('env_cfg'),
                            raw_sha256=raw_hash,expert_rank=entry['source']['rank'],label_version='event_goal_v2')
                with output.with_suffix('.npz.part').open('wb') as file:
                    np.savez_compressed(file,schema_version=np.uint8(4),feature_masks=features[:,MASKS].astype(np.uint8),
                                        feature_values=features[:,VALUES].astype(np.float16),
                                        metadata=np.asarray(json.dumps(meta)),**data)
                output.with_suffix('.npz.part').replace(output)
                save_json(output.with_suffix('.segments.json'),segments)
                manifest['files'][name] = dict(**meta,npz_sha256=digest(output),statistics=stat)
            save_json(manifest_path,manifest)
            print(f'[{index+1}/{args.limit}] {name} valid={manifest["files"][name]["statistics"]["valid_samples"]}',flush=True)
        rows = [r['statistics'] for r in manifest['files'].values()]
        aggregate = {k:int(sum(r[k] for r in rows)) for k in ('total_samples','valid_samples','invalid_samples','segments','valid_segments','censored_segments')}
        aggregate['mode_counts'] = {name:sum(r['mode_counts'][name] for r in rows) for name in MODES}
        for key in ('confidence_hist','goal_distance_hist','segment_duration_hist','valid_segment_duration_hist'):
            maxlen = max(len(r[key]) for r in rows)
            aggregate[key] = np.sum([np.pad(r[key],(0,maxlen-len(r[key]))) for r in rows],axis=0).tolist()
        aggregate.update(games=len(rows),mlflow_run_id=run.info.run_id)
        manifest['statistics'] = aggregate
        save_json(manifest_path,manifest)
        save_json(args.output_dir/'summary.json',aggregate)
        mlflow.log_metrics({k:v for k,v in aggregate.items() if isinstance(v,(int,float))})
        mlflow.log_artifact(str(manifest_path))
        print(json.dumps(aggregate),flush=True)


if __name__=='__main__':
    main()
