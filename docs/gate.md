# Stage 1: the gate (`localizer.gate`)

> **This package ships no trained weights.** The weights described on this
> page (`gate_A_stem2_bs1_ep42.safetensors`) are not included; `checkpoints/gate_A_stem2_bs1_ep42.json`
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
| Weights measured (not in this package) | `gate_A_stem2_bs1_ep42.safetensors`, 14,337,404 bytes |
| sha256 (identity) | `eb7932b53ec34ed931db94eddbd2cade5dfa0490d825a49538cd3ad826019a3e` |
| Converted from | `gate_A_stem2_bs1_ep42.pth` (PyTorch, 14,357,593 bytes, sha256 `d741b42d...08cdb`; not shipped), tensor-for-tensor identical |
| Manifest | `checkpoints/gate_A_stem2_bs1_ep42.json` |
| Config that produced it | `configs/gate/A_stem2_bs1.yaml` (hyperparameters identical to the training-time config at commit `5ced936`; only data/run locations were made relative) |
| Build | `build_segmenter_v0_7(arm=2, nfeats=32, fusion="sum", stem_stride=2)`, low-res head, `low_res_pool = (2, 4, 4)` |

## What it is

A region-proposal network for whole scenes: a 3D CNN encoder with a stride-2
stem and one bidirectional temporal Mamba block, ending in a 1x1x1 head that
outputs one logit per 2 frames x 4 x 4 px. Thresholded at 0.5, its output
decides which 128 x 128 px chips go to the classifier. Shapes and the chip
rule are in `docs/architecture.md`. Internally this is "v0.10 Arm A", the
no-stabilizer recipe.

## What it was trained on

- Data: `multi_target_swarm_512x512x64` (3000 synthetic scenes, 2250 with
  targets / 750 without; `docs/data.md`), split 2700 train / 300 val by
  `val_split 0.1`, `val_split_seed 0`. The val split holds 218 target-present
  and 82 target-absent scenes.
- Recipe: AdamW, batch 1, 50 epochs, lr 1e-3, weight decay 1e-2 (0 on SSM
  parameters), ReduceLROnPlateau on val loss (factor 0.5, patience 10), weighted BCE on
  the max-pooled low-res mask with `pos_weight` about 180, fp32, seed 0.
- Checkpoint selection ("gate criterion"): from epoch 30 on, maximize val
  `cluster_recall` subject to the mean absent-scene positive-voxel fraction
  (`abs_pos_frac`) <= 0.25. Selected epoch 42 (`cluster_recall` 0.8510,
  `abs_pos_frac` 0.1346).
- Training stack (not shipped, not reproducible on x86): aarch64 NVIDIA
  GB10 node with 128 GB unified memory, torch `2.12.0.dev20260322+cu128`
  (nightly), `mamba_ssm 2.3.1`, `causal_conv1d 1.6.1`. Peak allocated GPU
  memory 8.05 GiB (8,648,129,536 bytes; PyTorch `max_memory_allocated`, not
  the full process footprint).

<!-- Sources: checkpoints/gate_A_stem2_bs1_ep42.json (bytes, sha256,
config_commit, selection_criterion, dataset, host, torch, mamba_ssm,
causal_conv1d, val_metrics.val_cluster_recall 0.851, abs_pos_frac_mean 0.1346,
peak_vram_bytes 8648129536); recipe: configs/gate/A_stem2_bs1.yaml
(batch_size 1, num_epochs 50, lr 1.0e-3, weight_decay 1.0e-2,
weight_decay_ssm 0.0, scheduler_factor 0.5, scheduler_patience 10,
gate_settle_min_epoch 30, gate_abs_frac_bar 0.25, tppf 91 -> pw 180);
val split 218 / 82: originating run artifact
analysis/stage1_v0_10_percell/rescore.json results.A n_present 218 n_absent 82. -->

## What it has been measured to do (and on what)

All on the 300-scene val split of `multi_target_swarm_512x512x64`, threshold
0.5, scored on the 8 x 8 grid of 64 x 64 px cells (time-collapsed). A cell is
flagged when at least k gate voxels in it are positive.

| k | per-cell recall (target cells flagged) | absent scenes: cells flagged of 64, mean / p50 / p95 | present scenes: non-target cells flagged, mean | caught rate (present scenes with >= 1 target cell flagged) |
|---|---|---|---|---|
| 1 | 0.9485 | 25.76 (40.2%) / 0 / 64 | 20.27 | 1.000 |
| 2 | 0.9456 | 25.68 (40.1%) / 0 / 64 | 20.18 | 0.995 |
| 4 | 0.9413 | 25.33 (39.6%) / 0 / 64 | 19.84 | 0.995 |
| 8 | 0.9250 | 25.02 (39.1%) / 0 / 64 | 19.48 | 0.995 |

Read this as: the gate finds target cells (about 95%), but its false firing
is **bimodal per scene**: on more than half of empty scenes it flags nothing,
and on the worst ones it flags the whole grid. Raising k barely changes
either axis, so the flood is solid in time, not scattered single voxels.

Where the flood sits (k = 1, absent scenes): of the 25.8 cells flagged on the
average absent scene, 14.1 are interior (inner 6 x 6) and 11.6 are on the
outer ring, so about 45% of the false firing is the border ring, an
architectural boundary effect of zero padding and the stride-2 stem, and the
majority is interior. The per-cell flag-rate contrast between target-present
and target-absent scenes is weak (interior mean 0.167).

In the system (gate -> classifier, same val split): the gate proposes on
average 7.0 of 16 chips per absent scene and 11.0 per present scene; it
misses 27 of 4143 target groups outright (0.65%). The classifier rejects
98.8% of the gate's non-target proposals, so the flood costs essentially no
false detections. See `docs/results.md`.

Training-time val metrics at epoch 42 (as logged): `cluster_recall` 0.851,
`recall@3` 0.6881, unsplit Dice 0.3622, present-scene Dice 0.1878, mean
false-positive proposals per scene 40.19 (from `extract_proposals`). These are loose coverage metrics
(see "known wrong" below); prefer the per-cell and system numbers.

**Reproduction on x86.** On Linux x86-64, Tier 1 (torch 2.12.0+cu130), both
with the fused CUDA kernels and with the pure-PyTorch reference, every value
in the table above (all k) and every system number reproduced with zero
delta. The two backends are not bit-equal per voxel (216 of 15.7 M voxels
flip at 0.5 over 30 scenes; max logit difference 0.18 against logits up to
about 1795 in magnitude), but no cell decision changes. The 300-scene rescore
took 43 s with the kernels and 81 s with the reference on a 24 GB laptop
Blackwell GPU (NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU).

<!-- Sources: per-cell table: originating run artifact
analysis/stage1_v0_10_percell/rescore.json results.A.per_k["1","2","4","8"]
(per_cell_recall_strict 0.94850, 0.94564, 0.94134, 0.92504;
absent_cells_flagged mean 25.756, 25.683, 25.329, 25.024, p50 0, p95 64;
absent_cells_flagged_rate64 mean 0.40244, 0.40130, 0.39577, 0.39101;
present_nontarget_cells_flagged mean 20.271, 20.183, 19.844, 19.477;
caught_rate 1.0, 0.99541 x3). Ring/interior:
analysis/stage1_v0_10_percell/border_attribution.json results.A
interior_absent_flagged_mean 14.134 (ring = 25.756 - 14.134 = 11.62, 45.1%),
interior_mean_contrast 0.16689. System: analysis/stage1_v0_10_percell/
stage2_system_eval.json results.A gate_proposals_per_scene_absent.mean 7.0366,
_present.mean 10.9587, stage2_specificity_on_gate_hardnegs 0.98824;
per_target_recall.json results.A gate_miss 27, total_targets 4143.
Logged val metrics: checkpoints/gate_A_stem2_bs1_ep42.json val_metrics.
x86 reproduction, logit drift, wall times: the manifest's reproduced_on_x86
block and the originating repo's validation record
documents/notes/validation_2026_11_05_export.md section 2. -->

## What is known to be wrong with it

1. **Weak per-cell precision.** About 40% of cells flagged on the average
   empty scene (p95: all 64). Harmless at the system level today because the
   classifier rejects it, but it means the gate does little filtering on bad
   scenes and the classifier carries the precision.
2. **Border-ring flood.** About 45% of the absent-scene false firing is the
   outer ring of cells. Switching the encoder to replicate padding did *not*
   fix it in a controlled test (it raised firing; see `docs/history.md`,
   precision ladder). An optional fix, low priority while the classifier
   absorbs it.
3. **Seed bistability.** The same config, node and software environment
   (internally "L0") landed in the opposite basin: per-cell miss rate 28.4%
   with absent-cell flagging 15.3%, against this checkpoint's 5.2% / 40.2%.
   The trigger is seed or GPU (Mamba kernel) non-determinism; it is
   unidentified. A retrain of this config is not guaranteed to land where
   this checkpoint did. Multi-seed any hard-binary retrain.
4. **Coverage metrics in the training log are loose.** `cluster_recall` and
   `recall@K` credit a proposal landing near a target among the top K, which
   is near-trivial once the gate fires at all, and `abs_pos_frac` is a voxel
   fraction, not a proposal count. The checkpoint was selected on them.
   Select future gates on system recall.
5. **1024² is unmeasured end to end** and the shipped architecture needs
   31.8 GiB allocated to train one 1024² scene, more than a 24 GB card
   (`docs/environment.md`).
6. **Training was not re-run on the shipped (Tier 1) stack** beyond one
   full 512² epoch, run to time it (1,478 s on a 24 GB laptop GPU), and a
   1-epoch smoke run. Equivalence is established for inference only.
7. **The weight file carries no metadata.** Its `safetensors_sha256` in the
   manifest is its identity. Stem keys are `down1.down.*` (stride-2 stem); stride-1
   checkpoints are not drop-in compatible.

<!-- Sources: item 3: originating run artifact
analysis/stage1_gate_ladder_512/rescore_L0_L1_border.json results.L0 per_k 1
(per_cell_recall_strict 0.71617 -> miss 28.4%, absent_cells_flagged_rate64
mean 0.15282) and the ladder record's section 4 item 2 (same node, unchanged
env, cause unidentified); item 2: rescore_L0_L1_border.json results.L1
(rate64 mean 0.41159 vs L0 0.15282); item 5: originating repo
documents/notes/v0_7_step11_measurement.md "Peak alloc (real data) 31.82 GiB";
item 6: validation record section 7; item 7: manifest notes. -->

## Run it

Rescore the original gate per cell (needs the 512² dataset under
`$LSL_DATA_ROOT` and the weight file under `$LSL_CKPT_ROOT`, which this
package does not ship; for a retrained gate see "Retrain"):

```bash
python scripts/gate/rescore_percell.py --gate-ckpt A=gate_A_stem2_bs1_ep42.safetensors --out rescore_gate.json
```

### Retrain

With the shipped recipe (Tier 1, about 9 GiB GPU memory; the reference
Mamba of Tier 0 on GPU needs 33.4 GiB):

```bash
lsl-train-gate --config configs/gate/A_stem2_bs1.yaml 2>&1 | tee gate_train.log
```

- Data: `multi_target_swarm_512x512x64` (`docs/data.md`).
- Time: 50 epochs x about 1,478 s = about 20.5 h on a 24 GB GPU with the
  fused kernels. **Estimate**: one epoch was measured (RTX PRO 5000 laptop
  GPU), times 50. The original runs took 2,731-2,780 s per epoch on a GB10.
- Target (`checkpoints/gate_A_stem2_bs1_ep42.json` `val_metrics`): per-cell
  recall (k = 1) 0.9485, absent-cell flag rate 0.4024, caught rate 1.0
  (rescore); per-target(group) recall 0.9486 (3930 / 4143), gate misses 27,
  with the classifier. Logged at selection: `cluster_recall` 0.851,
  `abs_pos_frac` 0.1346.
- Seed-bistable (item 3 above): train at least 3 seeds (`--seed`) and select
  on per-target recall with the classifier. `models/best_gate_model.pth` is
  written only from epoch 30 on, when the absent-scene criterion is met.
- Score your checkpoint:
  `python scripts/gate/rescore_percell.py --gate-ckpt A=$LSL_RUNS_ROOT/gate/A_stem2_bs1/models/best_gate_model.pth --out rescore_gate.json`.

Other gate configs: `configs/gate/ladder_L3.yaml` (cell-focal loss, stride-1
stem; the recipe named for the next decoder round; needs about 35 GiB),
`configs/gate/anchor_128_arm_h.yaml` (the 128² architecture anchor),
`configs/gate/gatediag_512_base.yaml` (512² diagnostic base). More in
`docs/runbook.md`.
