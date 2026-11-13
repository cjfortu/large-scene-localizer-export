# Results: what has been measured, on what, and how far to trust it

> **This package ships no trained weights.** The numbers below were
> measured with the weights described by `checkpoints/<name>.json`, which are
> not included. They are the targets for a retrain (`README.md`, "No weights
> in this package: day one"), with the tolerances of section 7.

**What this is.** Every measured number for the shipped models, with its
caveats and its source. Numbers come from result files written by the scripts
in this repository; each section names the file in an HTML comment. "Original"
means measured in the originating environment (aarch64, torch nightly);
"x86" means re-measured on Linux x86-64 with the shipped Tier 1 environment.

**State in one sentence.** Gate -> classifier is measured end to end on 512²
synthetic data; the segmenter is validated standalone only, its shipped
weights are the epoch-19 checkpoint (val Dice 0.391), and integration is open
work.

## Caveats that bound every number below

1. **512² only.** All system numbers are on the 300-scene val split of
   `multi_target_swarm_512x512x64` (218 target-present, 82 target-absent
   scenes). 1024² end to end is unmeasured.
2. **"Target" = connected-component group, not mover.** A target is a 3D
   6-connected component of the truth mask; crossing or touching tracks merge,
   so a group can hold several movers and be "caught" via one of them.
   Per-mover recall is unmeasured and could be lower.
3. **0.949 is at the line, not past it.** The stricter centroid rule gives
   0.917. Margin is thin even per group.
4. **87% of the residual loss is the classifier**, rejecting true-target
   chips. It was trained on whole 128² scenes and is evaluated on 128² crops
   of 512² scenes. This distribution mismatch is load-bearing: the near-zero
   false-detection rate and the recall loss are two faces of the same
   conservative shift, and a retrain on deployment-style crops would move
   both. The current operating point is not the system's ceiling.
5. **The gate's flood is absorbed by the classifier.** The gate flags about
   40% of cells on the average empty scene, but the classifier rejects 98.8%
   of the gate's non-target chips, giving about zero false detections per
   scene. Gate precision matters little to the system *today*; it would
   matter if the classifier changed.
6. **Synthetic, abstract data only.** No real imagery has been evaluated.

## 1. System: gate -> classifier (512²)

| Measure | Gate A (shipped) | Gate B1 (EMA variant, not shipped; for context) |
|---|---|---|
| Per-target(group) recall, any-chip rule | **0.9486** (3930 / 4143) | 0.9124 (3780 / 4143) |
| Per-target(group) recall, centroid-chip rule | 0.9172 (3800 / 4143) | 0.8767 |
| Lost groups | 213 | 363 |
| ... lost at the gate (no proposal) | 27 (12.7% of loss; 0.65% of groups) | 167 (46%) |
| ... lost at the classifier (rejected) | 186 (87.3% of loss) | 196 (54%) |
| False detections per absent scene, mean / p95 / max | 0.00 / 0 / 0 | 0.00 / 0 / 0 |
| False detections per present scene, mean / p95 / max | 0.08 / 1 / 2 | 0.15 / 1 / 3 |
| Gate proposals per scene (of 16 chips), absent / present mean | 7.04 / 10.96 | n/a |
| Classifier specificity on gate non-target proposals | 0.988 (1531 chips) | 0.978 |
| Chip-level system recall (target-occupied chips detected) | 0.7147 (1047 / 1465) | 0.6867 |

The chip-level 0.715 is lower than per-target recall because many target
groups only graze a chip; a group counts as caught if any of its chips is
detected. Of A's 213 lost groups, 168 occupy a single chip, consistent with
small off-centre targets in a crop being the classifier's failure mode.

<!-- Sources: originating run artifacts analysis/stage1_v0_10_percell/
per_target_recall.json results.{A,B1} (per_target_recall 0.94859 / 0.91238,
caught 3930 / 3780, lenient 0.91721 / 0.87666, lenient_centroid_caught 3800,
lost 213 / 363, gate_miss 27 / 167, stage2_reject 186 / 196,
gate_miss_pct_of_loss 0.12676, stage2_reject_pct_of_loss 0.87324,
footprint_lost.1chip 168) and stage2_system_eval.json results.{A,B1}
(system_recall 0.71468 / 0.68669, target_chips 1047/1465,
fp_per_scene_absent, fp_per_scene_present, gate_proposals_per_scene_*,
stage2_specificity_on_gate_hardnegs 0.98824 / 0.97772,
gate_hardneg_chips_total 1531). -->

## 2. Gate alone (512², per cell)

8 x 8 grid of 64 x 64 px cells, time-collapsed, threshold 0.5, k = 1 (a cell
is flagged if any gate voxel in it is positive):

| Gate | per-cell recall | per-cell miss | absent scenes: cells flagged, mean / p50 / p95 | caught rate |
|---|---|---|---|---|
| **A, epoch 42 (shipped)** | 0.9485 | 5.2% | 40.2% / 0% / 100% | 1.000 |
| B1 (EMA), epoch 33 | 0.8489 | 15.1% | 31.7% / 0% / 93.8% | 0.995 |
| B2 (EMA + LR floor), epoch 33 | 0.7594 | 24.1% | 23.9% / 0% / 92.0% | 0.968 |

Sweeps over k = 2, 4, 8 and the present-scene false load are in
`docs/gate.md`. The stabilizers (B1, B2) buy fewer false flags by losing
recall; at the system level, where the classifier absorbs false flags, the
lowest-miss gate (A) wins.

<!-- Sources: analysis/stage1_v0_10_percell/rescore.json
results.{A,B1,B2}.per_k["1"] (per_cell_recall_strict 0.94850 / 0.84893 /
0.75937; absent_cells_flagged_rate64 mean 0.40244 / 0.31669 / 0.23895,
p50 0, p95 1.0 / 0.9375 / 0.9203; gate_epoch 42 / 33 / 33; caught_rate 1.0 / 0.99541 / 0.96789). -->

## 3. Classifier alone (128² scenes, its own val split)

| N | TP | TN | FP | FN | specificity | sensitivity | accuracy |
|---|---|---|---|---|---|---|---|
| 3000 (1500 / 1500) | 1494 | 1498 | 2 | 6 | 0.9987 | 0.9960 | 0.9973 |

<!-- Source: checkpoints/classifier_baseline_30k_ep64.json
val_metrics.specificity_eval and reproduced_on_x86.specificity_eval. -->

## 4. Segmenter alone (128² scenes, its own val split)

| Weights | N | val loss | val Dice | val IoU |
|---|---|---|---|---|
| epoch 19 (shipped; best val loss) | 250 | 0.0339 | **0.391** | 0.318 |
| epoch 150 (training log only; weights never saved) | 250 | 0.0892 | 0.8985 | 0.8403 |

Dice is the mean per-sample Dice at threshold 0.5; target-absent samples
score 1 for an empty prediction and about 0 otherwise. See
`docs/segmenter.md`.

<!-- Source: checkpoints/segmenter_volseg_ep19.json val_metrics and
reproduced_on_x86. -->

## 5. Gate precision levers tried at 512² (closed)

A controlled ladder of cheap levers against goals of per-cell miss < 1% and
absent-cell flagging < 25% (8 x 8 grid, k = 1). None reached both.

| Arm | Lever (cumulative) | Checkpoint scored | per-cell miss | absent cells flagged, mean | p95 |
|---|---|---|---|---|---|
| A (shipped) | none (stride-2 stem, zero padding) | gate epoch 42 | 5.2% | 40.2% | 100% |
| L0 | none; same config as A | gate epoch 34 | 28.4% | 15.3% | 98.4% |
| L1 | replicate padding | gate epoch 47 | 10.2% | 41.2% | 100% |
| L2 | + stride-1 stem | early peak, epoch 1 | 2.2% | 41.5% | 100% |
| L3 | + cell-focal loss (max-pool cells) | early peak, epoch 3 | 14.9% | 15.4% | 81.3% |

Findings: replicate padding does not remove the perimeter flood; the
stride-1 stem collapses after about epoch 3 and never yields a qualifying
gate checkpoint (L2 and L3 are scored at their early peaks); the cell-focal
loss (L3, `configs/gate/ladder_L3.yaml`) gives the best-balanced point and is
the recipe named for the next decoder round; L0 and A share a config and
landed in opposite basins (seed or kernel non-determinism, trigger
unidentified). Caveat: L2 trained without the fused `causal_conv1d` kernel
(non-fused convolution path), unlike the other arms.

<!-- Sources: analysis/stage1_gate_ladder_512/rescore_L0_L1_border.json
results.{L0,L1}.per_k["1"] (per_cell_recall_strict 0.71617 / 0.89814;
absent_cells_flagged_rate64 mean 0.15282 / 0.41159, p95 0.98359 / 1.0) and
rescore_L2_L3_early.json results.{L2,L3}.per_k["1"] (0.97768 / 0.85150;
mean 0.41502 / 0.15415; p95 1.0 / 0.8125); epochs and the collapse finding:
the ladder results record (originating repository), sections 2 and 4.
NOTE: that record's table gives L3 p95 as 40.3%; the JSON gives 0.8125, used here. -->

## 6. GPU memory and wall time

| Workload | Peak allocated GPU memory | Where measured |
|---|---|---|
| Gate A recipe (stride-2 stem), train, 512², batch 1, fp32 | 8.05 GiB | original training run |
| Gate L1 recipe (stride-2 stem, replicate padding), train, 512² | 8.85 GiB | ladder preflight |
| Gate L2 recipe (stride-1 stem), train, 512² | 38.4 GiB | ladder preflight |
| Gate L3 recipe (stride-1 stem, cell-focal), train, 512² | 34.6 GiB | ladder preflight |
| Gate, stride-2 stem, train, 1024², batch 1 | 31.8 GiB | 1024² measurement |
| Gate, stride-1 stem, train, 1024², batch 1 | about 111.6 GiB, out of memory on a 128 GB unified-memory GPU | 1024² measurement |

These are PyTorch `max_memory_allocated` peaks; the process footprint
(context, allocator cache, kernel workspaces) is higher and was not
captured. The two 1024² shape tests in the suite (forward + backward of the
shipped architecture at 1024²) run out of memory on the 24 GB GPU used here
(NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU, a laptop part).

Wall time, gate per-cell rescore of 300 val scenes on that laptop Blackwell
GPU (RTX PRO 5000, 24 GB, x86 host): 43 s with the fused kernels, 81 s with the pure-PyTorch reference.
One training epoch of the shipped gate recipe at 512² on that GPU with the
fused kernels: 1,478 s (0.393 s per batch, peak 8.05 GiB). Training with
the reference does not fit 24 GB (it needs 33.4 GiB, measured on a GB10,
where it ran 3.4x slower per batch than the `mamba_ssm` path); details in
`docs/environment.md`.
Original 50-epoch gate training at 512² was budgeted at about 32 h per arm on
the development GPUs.

<!-- Sources: 8.05 GiB: checkpoints/gate_A_stem2_bs1_ep42.json
peak_vram_bytes 8648129536; ladder: analysis/stage1_gate_ladder_512/
preflight_L1.json vram_peak_gib 8.849, preflight_L2.json 38.402,
preflight_L3.json 34.574 (the L1 preflight is stride-2, see the ladder
record: "VRAM 8.9/38.4/34.6 GiB" for L1/L2/L3); 1024²: originating repo
documents/notes/v0_7_step11_measurement.md ("Peak alloc (real data) 31.82 GiB";
"~111.61 GiB (OOM at 121.69 GiB total)"); 1024² tests OOM and rescore
walls: validation record sections 1 and 2; 32 h: ladder record section 5
("L0/L1 32 h"). -->

## 7. Reproduction on x86-64 (the shipped Tier 1 environment)

Linux x86-64, NVIDIA RTX PRO 5000 Blackwell laptop GPU (24 GB), driver
595.84, Python 3.11.18, torch 2.12.0+cu130, `mamba_ssm 2.3.2.post1`,
`causal_conv1d 1.6.2.post1`, numpy 2.4.6. The fused-kernel column was first
measured with CUDA 12.8 builds of the two extensions, then re-run with CUDA
13.0 builds compiled from the same pinned PyPI sdists that Tier 1 compiles
from source (tests, gate per-cell table, system and per-target recall):
identical on every field. The shipped `.safetensors` weights hold the same
tensors bit for bit as the PyTorch files these numbers were first measured
with, and loaded from the `.safetensors` files the gate per-cell table,
system and per-target recall are again identical on every field.

| Check | Original | x86, fused kernels | x86, pure-PyTorch reference |
|---|---|---|---|
| Gate per-cell table, all k | as section 2 | identical | identical |
| System, every field of section 1 for gate A | as section 1 | identical | (not run) |
| Classifier confusion counts (canonical data) | 1494 / 1498 / 2 / 6 | identical | n/a (no Mamba) |
| Segmenter epoch-19 val loss / Dice / IoU | 0.0339 / 0.3909 / 0.3183 | identical | n/a (no Mamba) |
| 512² data, batch 0 regenerated | aarch64 bytes | 100 / 100 files identical | n/a |
| Test suite | | 246 passed, 2 skipped (the 1024² tests) | |

Voxel level: the two Mamba backends differ slightly (max |logit diff| 0.18
against logits up to about 1795; 216 of 15.7 M voxels flip at 0.5 over 30
scenes), with no change to any cell, chip or group decision. Mamba reference
vs fused kernel on random inputs: max abs difference 1.2e-7 (fp32).

**Tolerances.** On this platform expect exact equality. Elsewhere, treat
anything outside the following as a regression or a data-identity problem
(check dataset hashes first):

| Quantity | Tolerance (absolute) |
|---|---|
| Gate per-cell recall (any k), caught rate | 0.005 |
| Gate absent-cell flag rate (mean) | 0.0125 (one absent scene flipping between 0 and 64 cells) |
| Per-target(group) recall, chip system recall | 0.005 |
| False detections per scene | 0.05 |
| Classifier TP / TN / FP / FN on canonical data | 3 each |
| Segmenter val Dice / IoU | 0.005 |
| Mamba reference vs fused kernel, unit test | 1e-4 |

<!-- Sources: originating repo documents/notes/validation_2026_11_05_export.md
sections 0-8 (environment, "241 passed, 2 skipped", per_k equality on both
backends, gate_logit_diff "overall max_abs 0.1844 total flips 216", parity
"1.192e-07", "identical=100 differ=0 missing=0", tolerance table section 8);
the reproduced_on_x86 blocks of the three checkpoint manifests; the re-run
with the rebuilt wheels: validation record section 9b and the gate manifest's
reproduced_on_x86.recheck_rebuilt_wheels. The system
evaluation was run on the fused-kernel backend only. -->

## 8. What has not been measured

- Per-mover recall (only per group).
- Anything end to end at 1024², including gate memory and wall time for
  inference at 1024².
- The segmenter in the pipeline (any end-to-end segmentation metric).
- The classifier on deployment-style crops after a retrain.
- Gate training on the shipped Tier 1 stack beyond one timed 512² epoch;
  multi-seed variance of the shipped recipe.
- Any real (non-synthetic) imagery.
