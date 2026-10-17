"""All local Kaggle wins -> clean goal/review data; no download or raw deletion."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.goal_review.config import load_config
from agents.goal_review.labels import REASONS, clean_labels, replan_targets
from agents.goal_review.memory import causal_contexts
from agents.strategic.data import MASKS, VALUES, unpack
from agents.strategic.experiment import digest, provenance, save_json, setup
from scripts.build_strategic_dataset import parse_raw, statistics
from scripts.download_kaggle_replays import validate_replay


BASE_KEYS = ('features', 'positions', 'energies', 'active', 'enemy_positions', 'enemy_energies',
             'enemy_active', 'steps', 'match_steps', 'team_points', 'rank', 'actions', 'team_id')


def inventory(raw_dir, legacy):
    entries, rejected = {}, {}
    for path in sorted(raw_dir.glob('*.json')):
        try:
            raw = json.loads(path.read_text())
            rewards = np.asarray(raw['rewards'], float)
            if rewards.shape != (2,) or not np.isfinite(rewards).all() or rewards[0] == rewards[1]:
                raise ValueError('not a strict win')
            side = int(rewards.argmax())
            validate_replay(raw, int(path.stem), side)
            if len(raw['steps']) != 506:
                raise ValueError('require complete official 506-observation game')
            key = f'{path.stem}_p{side}.npz'
            entries[key] = {'source_match': path.stem, 'team_id': side,
                            'source_seed': raw['configuration'].get('seed'),
                            'env_cfg': raw['configuration']['env_cfg'], 'raw_sha256': digest(path),
                            'legacy_feature_available': key in legacy['files']}
        except (ValueError, KeyError, TypeError) as exc:
            rejected[path.name] = str(exc)
    return entries, rejected


def split_entries(entries):
    reference = json.loads((ROOT/'experiments/B_strategic_goal_il/manifest.json').read_text())
    known = {name: split for split, names in (('train', reference['train_files']), ('validation', reference['validation_files'])) for name in names}
    assigned, seed_split = {}, {}
    for name, split in known.items():
        if name in entries:
            seed = entries[name]['source_seed']
            token = f'seed:{seed}' if seed is not None else name
            if token in seed_split and seed_split[token] != split:
                raise ValueError('legacy split has seed leakage')
            assigned[name], seed_split[token] = split, split
    groups = {}
    for name, row in entries.items():
        if name in assigned:
            continue
        token = f'seed:{row["source_seed"]}' if row['source_seed'] is not None else name
        if token in seed_split:
            assigned[name] = seed_split[token]
        else:
            groups.setdefault(token, []).append(name)
    import hashlib
    needed = round(len(entries)*.2) - sum(v == 'validation' for v in assigned.values())
    for token in sorted(groups, key=lambda x: hashlib.sha256(('42:'+x).encode()).hexdigest()):
        names = groups[token]
        split = 'validation' if 0 < len(names) <= needed else 'train'
        if split == 'validation':
            needed -= len(names)
        assigned.update({name: split for name in names})
    assert len(assigned) == len(entries)
    return assigned


def process(d, config):
    labels, segments = clean_labels(d, config)
    vectors, reviews, memory_stats = causal_contexts(d, config)
    r, rv, comp = np.zeros_like(d['active'], np.int8), np.zeros_like(d['active']), np.zeros_like(d['active'])
    for t in range(len(reviews)):
        regions = np.zeros((16,24,24), bool)
        for u in np.flatnonzero(labels['semantic_valid'][t]):
            regions[u] = labels['goal_regions'][labels['segment_id'][t,u]]
        r[t], rv[t], comp[t] = replan_targets(labels['mode'][t], regions, labels['semantic_valid'][t],
            labels['boundary_replan'][t], labels['bridge_continue'][t], vectors[t], labels['label_valid'][t])
    d = {**d, **labels, 'unit_vectors': vectors, 'review_frames': reviews, 'replan': r, 'replan_valid': rv}
    stat = statistics(d, segments)
    eligible = rv & reviews[:,None]
    filtered = d['active'] & ~d['label_valid']
    stat.update(environment_frames=len(reviews), frames_with_valid_goal=int(d['label_valid'].any(1).sum()),
                filtered_unit_states=int(filtered.sum()), filtered_environment_frames=int((filtered.any(1)).sum()),
                macro_decision_epochs=int(reviews.sum()), replan_supervised_units=int(eligible.sum()),
                replan_states=int((eligible & (r==1)).sum()), continuation_states=int((eligible & (r==0)).sum()),
                semantic_boundaries=int(d['boundary_replan'].sum()),
                corrective_replans=int((eligible & (r==1) & ~d['boundary_replan']).sum()),
                bridged_filtered_continues=int((eligible & (r==0) & filtered).sum()),
                legal_sap_attempts=int(d['sap_legal'].sum()), invalid_sap_attempts=int((d['sap_attempt'] & ~d['sap_legal']).sum()),
                filter_reasons={why:int(((d['filter_reason']==i)&filtered).sum()) for i,why in enumerate(REASONS)},
                scripted_memory=memory_stats)
    return d, segments, stat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/goal_review/common.yaml')
    p.add_argument('--raw-dir', type=Path, default=ROOT/'data/raw_replays')
    p.add_argument('--output-dir', type=Path, default=ROOT/'data/dataset_v3_goal_review')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    setup()
    config = load_config(args.config)
    old_dir = ROOT/'data/dataset_v2_goal'
    legacy = json.loads((old_dir/'manifest.json').read_text())
    for name,h in legacy['pipeline_sha256'].items():
        if digest(ROOT/name) != h:
            raise ValueError('legacy feature producer changed; cached features not reusable')
    entries, rejected = inventory(args.raw_dir, legacy)
    split = split_entries(entries)
    if args.limit:
        entries = dict(list(entries.items())[:args.limit])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dependency_files = [ROOT/'agents/goal_review'/name for name in ('labels.py','memory.py','config.py')]
    dependency_files += [Path(__file__)] + [ROOT/name for name in legacy['pipeline_sha256']]
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in dependency_files}
    manifest_path = args.output_dir/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    identity = {'pipeline_sha256':hashes, 'config':config, 'selected_files':list(entries)}
    if manifest and any(manifest[k] != v for k,v in identity.items()):
        raise ValueError('producer/config/source selection changed: choose new dataset directory')
    manifest.update(identity, dataset_version='dataset_v3_goal_review', schema_version=5,
                    feature_version='causal_grouped_v2', label_version='legal_sap_censored_v1',
                    observation_boundary='legal player observations <=T; previous assignments <T',
                    label_boundary='future segments/short compatible bridges are supervision ONLY',
                    previous_goal='ReviewMemory shared online/offline, scripted causal policy',
                    holdout_boundary='only Kaggle EpisodeId-validated raw_replays; no local schedules/replays loaded, no environment final evaluation',
                    split_policy='preserve old 320/80 IDs; extend seed-grouped 80/20 with hash42',
                    raw_count=len(list(args.raw_dir.glob('*.json'))), pilot=bool(args.limit),
                    rejected=rejected, provenance=provenance(), files=manifest.get('files',{}))
    save_json(manifest_path, manifest)
    from ml_workspace.tracking import tracked_run
    import mlflow
    with tracked_run(args.output_dir.name, experiment_name='lux-s3-goal-review', params={'games':len(entries),'seed':42,'operation':'dataset_build_no_training'}) as run:
        for index,(name, meta) in enumerate(entries.items()):
            path = args.output_dir/name
            if name in manifest['files']:
                assert digest(path)==manifest['files'][name]['npz_sha256']
                continue
            raw_path = args.raw_dir/(meta['source_match']+'.json')
            assert digest(raw_path)==meta['raw_sha256']
            if meta['legacy_feature_available']:
                assert digest(old_dir/name)==legacy['files'][name]['npz_sha256']
                with np.load(old_dir/name, allow_pickle=False) as z:
                    cached = unpack(z)
                d = {key:cached[key] for key in BASE_KEYS}
                d.update(env_cfg=meta['env_cfg'], source_seed=meta['source_seed'])
            else:
                d = parse_raw(raw_path, meta['team_id'], config)
            d, segments, stat = process(d, config)
            features = d.pop('features')
            d.pop('env_cfg');d.pop('source_seed')
            file_meta = {**meta, 'split':split[name], 'label_version':manifest['label_version']}
            temporary = path.with_suffix('.npz.part')
            with temporary.open('wb') as file:
                np.savez_compressed(file, schema_version=np.uint8(5), feature_masks=features[:,MASKS].astype(np.uint8),
                                    feature_values=features[:,VALUES].astype(np.float16), metadata=np.asarray(json.dumps(file_meta)), **d)
            temporary.replace(path)
            save_json(path.with_suffix('.segments.json'), segments)
            manifest['files'][name] = {**file_meta, 'npz_sha256':digest(path), 'statistics':stat}
            save_json(manifest_path, manifest)
            print(json.dumps({'completed':index+1,'games':len(entries),'file':name,'valid':stat['valid_samples'],
                              'replan':stat['replan_states'],'continue':stat['continuation_states']}), flush=True)
        rows = [r['statistics'] for r in manifest['files'].values()]
        summary = {}
        for key in rows[0]:
            values = [r[key] for r in rows]
            if isinstance(values[0], dict):
                total = Counter()
                for v in values:total.update(v)
                summary[key] = dict(total)
            elif isinstance(values[0], list):
                length = max(map(len,values))
                summary[key] = np.sum([np.pad(v,(0,length-len(v))) for v in values],axis=0).tolist()
            else:summary[key] = sum(values)
        summary.update(games=len(rows), split_counts=dict(Counter(split[n] for n in entries)),
                       mlflow_run_id=run.info.run_id, complete=True)
        manifest['statistics'], manifest['complete'] = summary, True
        save_json(manifest_path,manifest);save_json(args.output_dir/'summary.json',summary)
        mlflow.log_artifact(str(manifest_path))
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
