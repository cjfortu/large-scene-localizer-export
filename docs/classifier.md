# Stage 2: the classifier (`localizer.classifier`)

> **This package ships no trained weights.** The weights described on this
> page (`classifier_baseline_30k_ep64.safetensors`) are not included; `checkpoints/classifier_baseline_30k_ep64.json`
> is their manifest, and its `val_metrics` are the target for a retrain
> (with the tolerances of `docs/results.md`, section 7). Retrain command,
> time and target: "Retrain" below; the whole day-one path: `README.md`. If
> the weight file arrives separately, put it in `checkpoints/`, check it
> against the manifest hash (`docs/runbook.md`, step 3), and everything else
> on this page applies unchanged.

Pipeline status: gate -> classifier is measured end to end on 512² synthetic
data; the segmenter is validated standalone only (shipped weights: epoch-19
checkpoint, val Dice 0.391), and its integration is open work.

| Item | Value |
|---|---|
| Weights measured (not in this package) | `classifier_baseline_30k_ep64.safetensors`, 69,099,252 bytes |
| sha256 (identity) | `2f9c96d643b6321465cc7bd92b19c2d6e9d98196ef32a742e9b0147c362a989f` |
| Converted from | `classifier_baseline_30k_ep64.pt` (PyTorch, 69,112,200 bytes, sha256 `03639c42...59d5d6`; not shipped), tensor-for-tensor identical |
| Manifest | `checkpoints/classifier_baseline_30k_ep64.json` |
| Config that produced it | `configs/classifier/baseline_30k.yaml` (hyperparameters identical to the training-time config at commit `487f385`; only data/run locations were made relative) |
| Build | `Mover3DCNNTrans(t_dim=64, dropout_rate=0.5, attn_dropout=0.1, max_attn_len=2200, head_type="flatten", return_logits=False, attn_type="full")` |

## What it is

A binary mover / no-mover classifier for one 128 x 128 x 64 chip: a 3-stage
3D CNN (32, 64, 128 channels, each stage halving T, H, W), one global
self-attention block with relative position bias over the 8 x 16 x 16 = 2048
bottleneck tokens, and an MLP head. Input `(B, 1, 64, 128, 128)` in `[0, 1]`;
output `(B,)` probabilities (sigmoid inside `forward`). Threshold 0.5. In the
pipeline it runs on every chip the gate proposes and is the stage that
removes the gate's false firing.

The input size is fixed: the head's linear layer assumes a 16 x 16 spatial
bottleneck, so only 128 x 128 chips work.

## What it was trained on

- Data: `swarm_128x128x64`, 30,000 synthetic 128² scenes, 15,000 with a
  target and 15,000 without, stratified 90/10 split with `val_split_seed 91`
  (27,000 train / 3,000 val; 1,687 / 188 batches of 16). These are whole
  128² scenes from the generator's `single` sampler: a background swarm of
  identical moving disks, with one anomalous mover (different direction or
  speed) in target scenes.
- Recipe: Adam, lr 1e-3, no weight decay, batch 16, BCE on the sigmoid
  output, no scheduler, no augmentation, seed 0. Planned 100 epochs; the run
  was stopped deliberately after epoch 65. Checkpoint selected by best val
  accuracy: epoch 64, val accuracy 0.9973 (val loss 0.0077, train accuracy
  0.9956).
- Training stack (inferred, not shipped): aarch64 NVIDIA GB10 node, torch
  `2.12.0.dev20260322+cu128`.

**The training set regenerates bit-identically** from
`configs/data/swarm_128.yaml` (six invocations; `docs/data.md`). To change
what the classifier sees, build the deployment-style crop set described under
"known wrong" below.

<!-- Sources: checkpoints/classifier_baseline_30k_ep64.json (bytes, sha256,
config_commit, epoch, selection_criterion, val_metrics.val_acc 0.9973,
val_loss 0.0077, train_acc 0.9956, dataset "30000 scenes, 15000/15000 labels;
1687/188 train/val batches", notes "deliberately stopped after ep65 of 100",
inferred_fields.torch); recipe: configs/classifier/baseline_30k.yaml and
src/localizer/classifier/train.py (optim.Adam, BCELoss when return_logits
false, "No scheduler. No augmentation."); stratified split:
src/localizer/classifier/data.py (train_test_split(stratify=labels)); no
metadata: the canonical dataset directory holds data_all and label_msk only
(30000 files in data_all). -->

## What it has been measured to do (and on what)

**Standalone, on its own val split** (3,000 whole 128² scenes, 1,500 / 1,500):

| TP | TN | FP | FN | specificity | sensitivity | accuracy |
|---|---|---|---|---|---|---|
| 1494 | 1498 | 2 | 6 | 0.9987 | 0.9960 | 0.9973 |

**In the system** (gate -> classifier on the 300-scene val split of
`multi_target_swarm_512x512x64`, 128² crops of 512² scenes):

| Measure | Value |
|---|---|
| Rejects the gate's non-target proposals (specificity on gate hard negatives) | 0.988 (over 1,531 chips) |
| False detections per scene, absent / present (mean) | 0.00 / 0.08 (present max 2) |
| Target groups lost to a classifier reject | 186 of 4,143 (87% of all 213 lost groups) |
| Lost groups that span a single chip | 168 of 213 |

The classifier is near perfect on the distribution it was trained on and
very specific in the pipeline; its weakness is recall on off-distribution
crops.

**Reproduction on x86.** Scored on Linux x86-64, Tier 1, against the
canonical copy of its val data, the confusion counts reproduce exactly
(1494 / 1498 / 2 / 6). A separately generated copy of the same 30k set
(same generator and indices, not bit-identical) gives 1491 / 1499 / 1 / 9;
no sample sits within 0.04 of the threshold and batch size or TF32 do not
change the counts, so the difference is the data, not the platform. Check
dataset hashes before treating a mismatch as a regression.

<!-- Sources: standalone counts: checkpoints/classifier_baseline_30k_ep64.json
val_metrics.specificity_eval and reproduced_on_x86.specificity_eval; system:
originating run artifacts analysis/stage1_v0_10_percell/stage2_system_eval.json
results.A (stage2_specificity_on_gate_hardnegs 0.98824, gate_hardneg_chips_total
1531, fp_per_scene_absent.mean 0.0, fp_per_scene_present.mean 0.08257 max 2.0)
and per_target_recall.json results.A (stage2_reject 186, lost 213,
stage2_reject_pct_of_loss 0.87324, footprint_lost.1chip 168); the
independent-draw result: originating repo documents/notes/
validation_2026_11_05_export.md section 4. -->

## What is known to be wrong with it

1. **Distribution mismatch (the largest lever in the system).** Trained on
   whole 128² scenes holding a single anomalous mover; deployed on 128²
   crops of 512² scenes, where targets are off-centre, cut by the chip edge,
   sparse within the chip, or clustered. 87% of the system's lost target
   groups are classifier rejects, and the lost groups are mostly
   single-chip groups, consistent with this. The fix is a retrain or
   fine-tune on deployment-style crops (128² crops of 512² scenes, cut on
   the gate's 4 x 4 chip grid). Expect both the near-zero false-detection
   rate and the recall to move; the current operating point is not the
   system's ceiling.
2. **No per-sample validation predictions or confusion matrix were saved**
   by the trainer, only `val_acc`. The confusion counts above were recovered
   afterwards by `scripts/classifier/specificity.py`. Save both on the next
   training run.
3. **Fixed 128 x 128 x 64 input.** A different chip size needs a new head.
4. **Same byte size, different weights.** Three other classifier runs in the
   originating project produced files of exactly this size with different
   hashes. Identify the checkpoint by sha256 only.

## Run it

Standalone confusion counts (needs `swarm_128x128x64` under `$LSL_DATA_ROOT`
and the original weight file, not shipped; for your own checkpoint add
`--checkpoint <path>`; prints whether the counts match the table above and
exits with code 2 if they do not):

```bash
python scripts/classifier/specificity.py
```

In the system, with the gate (needs the 512² dataset and both weight files;
for retrained ones add `--gate-ckpt A=<path>` and `--classifier-ckpt <path>`):

```bash
lsl-system-eval --out system_eval.json
lsl-per-target-recall --out per_target_recall.json
```

### Retrain

With the shipped recipe (any tier with a GPU; no Mamba blocks):

```bash
lsl-train-classifier --config configs/classifier/baseline_30k.yaml 2>&1 | tee cls_train.log
```

- Data: `swarm_128x128x64`, regenerated with the six-invocation layout
  (`docs/data.md`); a different layout gives different scenes.
- Time: 1,052 s per epoch, **measured** in the original run on an NVIDIA
  GB10 (65 epochs in 19.0 h; the selected epoch 64 at about 18.7 h). The
  config plans 100 epochs, about 29 h (**estimate**). Not timed on a 24 GB
  GPU.
- Target (`checkpoints/classifier_baseline_30k_ep64.json` `val_metrics`):
  val accuracy 0.9973 (val loss 0.0077); TP / TN / FP / FN 1494 / 1498 / 2 / 6
  on the 3,000-scene val split.
- Validation accuracy sat at chance (0.50-0.52) for the first 12 epochs of
  the original run, then rose (0.661 at epoch 13, 0.755 at 15). A flat start
  is expected; the original run was stopped after epoch 65.
- Score your checkpoint:
  `python scripts/classifier/specificity.py --checkpoint $LSL_RUNS_ROOT/classifier/baseline_30k/models/best_classifier.pt`.

`configs/classifier/default.yaml` is the same model with 75 epochs. More in
`docs/runbook.md`.
