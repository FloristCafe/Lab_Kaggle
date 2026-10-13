# V3 current: causal three-frame normalized Gaussian KL training

UnitFrameDataset now normalizes each in-memory Gaussian target over 576 cells
to sum to one (sigma=1.5). The network returns raw [K,1,24,24] logits, with no
sigmoid/softmax in forward. Active objective: KLDivLoss(reduction="none") on
FP32 log_softmax(logits), summed over cells and averaged over valid units.
Invalid units are selected out before loss evaluation, including NaN padding.

The legal 35-channel history input is unchanged: original hindsight channels
6..8 and projections 21..24 are never inputs. No projection dropout is needed.
History, target probabilities, unit anchors and coordinates remain memory-only.
The existing 400 compact NPZ and heuristic label validity are unchanged.

Active entry scripts/train_spatial_bc.py defaults to all 400 manifest games,
locks the previous 320/80 fold-0 split/NPZ hashes, and stops after five epochs
without lower raw validation KL. It records raw and decoding-masked accuracy,
moving/stay accuracy, distances, peak probabilities and entropy. Destination
masking remains decoding-only: neither positive Gaussian tails nor KL logits
are hard truncated during training. KL fits a deliberately smooth target and
does not guarantee sharp top-1 or identification of a unique drift speed.

--preflight-only validates data/split and executes a tracked synthetic and
real-batch forward/backward without optimizer updates; full training is handed
to the user. --min-games 400 rejects short or expanded collections instead of
silently selecting a subset. Prior MSE entry is train_spatial_mse_legacy.py,
using normalize_targets=False; its recorded artifacts remain unchanged.

## Archived Gaussian MSE notes

# V3 current: causal three-frame Gaussian heatmap regression

Active UnitFrameDataset reads the unchanged compact NPZ and returns 35-channel
queries, peak-1 Gaussian targets (Euclidean sigma=1.5), and unit-valid masks.
Current channels use original indices 0..5,9..20 (18); original hindsight
terrain 6..8 and projections 21..24 are never model inputs. Past T-1/T-2 each
add visible, visible terrain (3 masks), energy, enemy count and enemy energy
(7 planes). Missing/gapped/reset history is zero padded. Anchor index=32,
coordinate indices=33/34. History and heatmaps exist only in memory.

Network defaults to 35 -> 1 U-Net. Active training uses sigmoid output and
masked per-unit mean MSE, with no CE/focal or destination mask in its objective.
Optional optimistic destination masking is evaluated separately when decoding.
Raw MSE/accuracy/distance and zero, training-mean, anchor-Gaussian baselines
guard against declaring a nearly blank heatmap a successful policy.

Entry point: scripts/train_spatial_bc.py, deliberately capped at the first 10
manifest episodes, 8/2 grouped split. This is a quick validation, not a 400-game
run. Prior projection/CE experiments are archival: old Dataset is named
LegacyProjectionFrameDataset, and its exact 28-channel model is unet28_network.py.
Archived scripts explicitly import those legacy definitions. Existing artifacts
and NPZ hashes are preserved. Labels remain future heuristic supervision;
legal input evidence cutoff is T, and no target/validity information is input.

## Archived 28-channel experiment notes

# V3 current: CoordConv U-Net + Focal Loss

The active network is 28 -> 1 channels. Dataset generates x/y planes at indices
26/27 in memory, following the existing [x,y] axes; unit anchor stays at index
25. The compact 25-channel NPZ files are unchanged. U-Net downsamples 24 -> 12
-> 6, adds globally pooled bottleneck context, then uses skip connections to
restore 24x24. GroupNorm replaces replay-dependent BatchNorm running stats.

The training objective is multiclass Focal Loss, gamma 2. Logits are cast to
FP32 and provably impossible cells are masked with -1e9 before optimization.
Validation logs both masked and raw CE/accuracy, actual anchor prediction rate,
moving/stay target accuracy and projection-zeroed performance. Early stopping
continues to use masked validation CE, patience 5.

Full-label audit uncovered continuously active reused slots and reset-transition
labels. The loader excludes 9,588 such archived supervision windows (0.70%),
leaving 1,358,977 labels. All remaining labels pass the optimistic energy/horizon
mask. NPZ hashes and the 320/80 episode split are locked to the previous run;
label validity is corrected only in memory. The frozen old checkpoint is
re-evaluated on this same subset before full training, without optimizer updates.
Its exact old definition is archived in residual_network.py.

Mask assumptions are explicit: unknown costs use official optimistic bounds
(move >=1, recovery <=20, energy <=400, unknown nebula drain >=0). Legacy NPZ
has tile-summed energy and no match-specific inferred nebula drain, so strict
per-unit deep-nebula exclusion is unavailable. Unknown cells stay allowed when
reachability cannot be disproven. The mask never reads projected terrain or
hidden replay state; existing feature backcast leakage still remains.

The full current command and limits are in outputs/unet_coord_focal_retrain.md.
Training output: outputs/bc_unet_coord_focal400_fold0_seed42. Run the command
with tee as documented to preserve terminal errors and normal stdout.

References: [CoordConv](https://arxiv.org/abs/1807.03247),
[U-Net](https://arxiv.org/abs/1505.04597),
[Focal Loss](https://arxiv.org/abs/1708.02002).

The sections below preserve historical experiment provenance. Their 26-channel
network descriptions and old commands do not describe the current architecture.

## Historical 26-channel compact data pipeline

The current model is `[K,26,24,24] -> [K,1,24,24]`. Channel 25 (zero based)
is a temporary one-hot unit anchor; it is never stored in NPZ. Convolutions
share weights across queries, but finite-map zero padding still causes boundary
effects, so absolute translation equivariance is not guaranteed.

`replay_parser.py` schema 3 stores 18 binary channels as uint8 and seven
continuous/count/energy channels as float16, in separate arrays. It uses
`np.savez_compressed`, with no duplicate dense feature array. Labels remain
uint8 one-hot targets and boolean validity masks. Legacy schema-2 files and
25-to-16 checkpoints are incompatible and must not be mixed with this version.

Intent filtering uses the maximum Manhattan displacement from the current
position across t+1 through t+12. Displacement below three rejects the label,
unless the current player's observation contains a cardinal-adjacent enemy or
the unit is on a map edge. Move evidence and at least two future observations
are still required. Windows stop at death, match reset or observation gaps.
A move away and back is therefore preserved. No hidden enemy state is used.

`agents/v3/dataset.py` loads compact replay shards through a bounded cache,
indexes only valid frames and creates `[K,26,24,24]` unit queries in memory.
Training shuffles replay order and frames within each replay to avoid repeatedly
inflating ZIP members. The frame collator handles variable K. Future dropout is
sampled once per frame (20%), before copying its shared map to unit queries.

`stream_replays.py --offline` selects 400 locally available strict wins from
an existing downloader selection manifest, round-robin across ranks <=100.
The current local collection has 1,125 replays from the first 25 teams; selecting
400 gives 16 games per team. Offline mode never loads credentials or makes
network requests, and missing files fail explicitly. JSON is parsed directly
in place, with no staging copy. `--burn-existing` removes selected originals
only after verification; omit it to retain them. An online mode is also available
but is not invoked by the current commands. It retains only the
selected winning team's view to avoid cloning low-quality opposing submissions.
At most ten newly staged JSON files accumulate before parsing. It verifies each
NPZ's schema and ZIP members, hashes it, persists NPZ and manifest to disk, then
calls `os.remove` on the corresponding JSON. Failed parses/writes keep JSON.
The same command resumes; verified completed NPZ files are not downloaded again.
`--reuse-dir` uses existing JSON first; `--burn-existing` also removes each
selected original after success. Unselected old JSON is left intact.

Pipeline storage is capped at 40 GiB, with a five-GiB free-space reserve checked
on ext4 and `/mnt/c` (WSL's virtual disk free size is not physical Windows free
space). This cap describes this output directory, not unrelated preexisting data.

`train_full_bc.py` requires at least 400 games by default. GroupKFold(5), fold 0,
keeps entire episodes apart: 320 training games and 80 validation games for the
400-game, one-view dataset. It validates manifest-referenced shards, trains
with <=32-unit BF16 microbatches, accumulates unit-weighted gradients over four
frames per optimizer update, uses Adam weight decay 1e-4 and stops after five
consecutive epochs without lower validation CE. The best model is saved when
loss improves and reloaded before final accuracy reporting. Every epoch logs
both normal and projection-zeroed validation metrics to MLflow and metrics.csv.

[WSL Bash]

```bash
cd "/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3"
.venv/bin/python scripts/stream_replays.py \
  --offline --target-games 400 --batch-size 10 \
  --selection-manifest data/kaggle_downloads/86411/selection_top25_wins45_scan200.json \
  --reuse-dir data/raw_replays --burn-existing \
  --data-dir data/processed_replays_unitcentric25 \
  --max-total-gib 40 --reserve-gib 5
/home/issue/ml-workspace/.venv/bin/python scripts/train_full_bc.py \
  --data-dir data/processed_replays_unitcentric25 \
  --output-dir outputs/bc_unitcentric_400_fold0_seed42 \
  --min-games 400 --fold 0 --seed 42 \
  --batch-size 32 --frames-per-batch 4 --max-epochs 60 \
  --patience 5 --dropout-probability 0.20 --weight-decay 1e-4
```

Dataset resume reuses the same command and directory. Training requires a fresh
output directory if interrupted; it does not claim optimizer-state resume.
No official match schedule or final 50-seed evaluation is invoked.

Leakage boundaries remain explicit: drift speed is inferred from later public
observations and backfilled into earlier frames. Removing projections does not
remove hindsight from the backcast current-terrain channels. Match-ID grouping
prevents identical episodes from crossing the split, but does not verify unique
map seeds. Validation here measures offline pseudo-label imitation, not online
policy strength or competition win rate.

## Historical 25-channel experiments (previous architecture)

The descriptions and commands below record completed earlier experiments.
They are historical provenance; those old datasets/checkpoints cannot be used
with the current unit-centric network and schema-3 loaders.

# V3 fixed-batch behavior-cloning check

`network.py` implements a 25-channel, 64-hidden-channel, four-block
residual network. Every convolution has stride 1. The 16 output planes are raw
macro-target logits indexed by replay unit slot; there is no unit query channel.
This model is currently a tensor/optimization sanity check, not a deployed agent.
Four added channels project known nebula and asteroid tiles 5 and 10 steps ahead.
The replay parser backcasts a speed inferred later in the game, so early training
features contain future-observation leakage and cannot be used for online inference.
The previous 21-channel checkpoint is incompatible with this network.

`scripts/train_bc.py` samples 32 frames without replacement from 25-channel NPZ
files using seed 42, requiring at least one valid intent per frame. It loads
that batch once and performs 500 Adam updates on exactly those same tensors.
There is no augmentation, dropout, validation set, or strategy modification.
The CUDA default is BF16 convolution with FP32 cross-entropy. Invalid unit slots
have zero loss and zero gradients; an entirely invalid batch skips the update.
Spatial class indices follow the parser's `[x, y]` convention: `class = x*24+y`.

Run in the existing ML workspace interpreter; no installation in the match-runner
virtual environment is needed. Choose an empty output directory for every run.

[WSL Bash]

```bash
cd "/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3"
/home/issue/ml-workspace/.venv/bin/python -m unittest discover -s tests -p test_bc.py -v
/home/issue/ml-workspace/.venv/bin/python scripts/train_bc.py \
  --data-dir data/processed_replays_backcast_future25 \
  --batch-size 32 --epochs 500 --seed 42 --lr 0.001 \
  --device cuda --precision bf16 --output-dir outputs/bc_backcast_future25_seed42
```

Artifacts include the 501-row loss/accuracy CSV (initial state plus 500 updates),
curve, checkpoint, summary, and manifest containing exact batch indices,
channel order, data/source hashes, runtime, Git state and MLflow identity.
The first run's report is at `outputs/bc_single_batch_seed42/report.md`.

Memorizing this batch cannot validate replay action/observation timing, expert
intent semantics, a consistent global x/y transpose, or unseen unit-slot identity.
Those remain separate questions before broader imitation learning.

## Group-held-out full behavior cloning

`scripts/train_full_bc.py` loads all 10 NPZ files from the 25-channel dataset.
`GroupKFold(n_splits=5)` keeps both player views of each replay in the same fold;
fold 0 trains on four replays and validates on `71903094`. Training independently
zeros all four future-projection channels for 20% of frames, with Adam L2 weight
decay `1e-4`. It logs validation metrics both with projections and with all four
projection channels removed. Validation masked CE selects `best_model.pth`, and
five non-improving epochs trigger early stopping.

[WSL Bash]

```bash
cd "/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3"
/home/issue/ml-workspace/.venv/bin/python scripts/train_full_bc.py \
  --data-dir data/processed_replays_backcast_future25 \
  --output-dir outputs/bc_full_group_fold0_seed42 --fold 0 --seed 42
```

The fold-0 run stopped at epoch 6; its best checkpoint was epoch 1, with only
4.73% valid-unit accuracy on the held-out replay. The learning curves therefore
show a generalization failure, not successful validation convergence. The
backcast speed also remains unavailable online before it is inferred.

### Inconclusive drift overlap regression fix

Replay 71902713 (winner view 1), observation step 8, has only six comparable
cells for the correct +0.15 drift candidate. That is below the eight-cell
threshold. Other, sufficiently observed candidates fail, so absence of an exact
compared candidate is inconclusive rather than a model conflict. Only refuting
all candidates after sufficient overlap reports a conflict. Candidate update
rules, feature planes and labels are unchanged for already accepted replays.

The stream runner permits only the explicit old/new parser SHA transition for
this fix. It verifies completed NPZ hashes, backs up the old manifest, records
per-replay parser provenance and resumes without rewriting committed shards.
Unknown parser edits remain incompatible. With 107 committed replays, rerun the
offline stream command first; training still requires 400 complete groups.
