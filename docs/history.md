# History: how the architecture got here

**What this is.** A two-page orientation for why the shipped system looks
the way it does, so that retired ideas are not re-tried blind. **State:**
gate -> classifier is measured end to end on 512² synthetic data; the
segmenter is validated standalone only (shipped weights: epoch-19
checkpoint, val Dice 0.391), and its integration is open work.

## The starting point (March 2026)

The original design (`docs/concept/stage1_concept.pdf`) was a two-stage
pipeline: a bidirectional-Mamba region-proposal network over the full
1024 x 1024 x 64 volume, followed by an existing 3D U-Net + Swin segmenter
on 128 x 128 x 64 patches. The segmenter (today's Stage 3) and a 3D CNN +
attention mover classifier (today's Stage 2) were recovered from earlier
work and validated on whole 128² swarm scenes. Everything below is about
building the first stage.

## Stage 1, in order

| Round | What was tried | Verdict |
|---|---|---|
| Phase A (April) | Mamba state-space backbone over projected 256² frames | **Retired.** Could not localize: single-scene overfit F1 0.545 against a bar of 0.80; multi-scene val F1 0.024. A skip path carried most of the gradient. |
| Day 0 - 0.6 (May) | Hand-built motion features, then optical flow, then a per-pixel signal check, all at 256² | **Negative.** Best was 0.170 of empty area discarded at 0.964 recall (needed 0.75 at 0.95); per-pixel signal weak. A learned 3D model is needed. |
| Day 1 | Learned 3D stem, then temporal collapse to a 2D ConvNeXt | **Retired.** Held-out AUC peaked at 0.652 at epoch 2 and fell to 0.349 by epoch 20; judged unlikely to work on this task. |
| Day 2 | 3D CNN + Swin vs the Day 1 network at 128² | Null result; not merged. |
| v0 | Reproduce the recovered pure-convolution 3D segmenter as a gate, four bottleneck variants, 128² | **Learns** (trajectory Dice up to 0.906), but val-loss checkpoint selection saved weak epochs. Checkpoint criterion changed to Dice from here on. |
| v0.5 | Temporal-axis variants (wide kernels, delayed pooling, dilation), 128² | Isotropic 3x3x3 baseline won (Dice 0.849). |
| v0.6 | CNN + Mamba hybrid vs pure Mamba, full-res vs low-res head, 128² | **Family settled: CNN + Mamba hybrid** (low-res arm Dice 0.903). Pure Mamba unstable (gradient explosions). |
| v0.7 - v0.8 | Pure-Mamba rescue attempts; the hybrid at 1024² (stride-2 stem added to fit memory: 111 GiB -> 31.8 GiB) | Pure Mamba failed to learn spatial discrimination at 1024². The 1024² hybrid was stable but weak; absent-scene restraint identified as the missing property. |
| v0.9 | Controlled isolation on a new multi-target 128² set | Hybrid passes (Dice 0.868); pure Mamba fails both ways. Pure Mamba deferred as a research line. |
| Collapse diagnostic | Which knob causes absent-side collapse at scale | Scale-intrinsic, not a config knob; present-side loss traced to the stride-2 stem. |
| 512² gate diagnostic | Move to 512² as the working scale | Restraint is solvable at 512² (pos_weight about 180). 512² becomes the measurement scale. |
| v0.10 | Arm A (no stabilizer), B1 (+EMA), B2 (+EMA + LR floor) at 512² | Initially framed as "stabilize the absent side; EMA passes". **Framing retired** by the system verdict below. |
| System verdict | Score gates per cell, then run gate -> classifier end to end | **Arm A is the gate.** Per-group recall 0.949 at about 0 false detections per scene; the classifier absorbs the gate's flood; 87% of the remaining loss is the classifier. EMA cut coverage to fix a flood the system already absorbs. No more gate stabilization; no architecture pivot. |
| Precision ladder (closed end of June) | Cheap gate-precision levers at 512²: replicate padding, stride-1 stem, cell-focal loss | None reached both goals. Cell-focal loss (L3) is the best-balanced lever and the recipe named for the next decoder round; stride-1 collapses after about epoch 3; same-config seeds land in opposite basins. |

<!-- Sources (originating repository, documents/history/results-handoffs/):
stage1_phase_a_retired_handoff.md ("0.5454 (ep 200)", "F1 > 0.80",
"val_f1 = 0.0241", skip path "65-76% of the gradient");
stage1_day0_bank_baseline_handoff.md, stage1_day0_5_diagnostic_handoff.md
("empty-discarded 0.170 at recall 0.964"), stage1_day0_6_perpixel_diag_handoff.md;
stage1_day1_3d_learning_handoff.md ("0.652 at epoch 2" ... "0.349 by epoch 20");
Day 2: recon_2026_05_16_reorg.md (null result, not merged);
stage1_v0_partial_results_handoff.md ("A at ep 69 ... 0.906");
stage1_v0_5_handoff.md ("0.8493 (ep 115)"); stage1_v0_6_handoff.md
("0.9034 at ep 81", "gradient explosions"); stage1_v0_7_handoff.md and
documents/notes/v0_7_step11_measurement.md (111.61 / 31.82 GiB);
stage1_v0_8_handoff.md; stage1_v0_7_v0_8_hybrid_rescore_handoff.md;
stage1_v0_9_handoff.md ("best_unsplit_dice = 0.8678 @ ep 40");
stage1_hybrid_collapse_isolation_diagnostic_handoff.md;
stage1_gatediag_512_phase2_handoff.md; stage1_gate_viability_system_handoff.md;
stage1_gate_precision_ladder_512_handoff.md. Dice values in v0-v0.9 are the
training-time unsplit Dice of those rounds, not comparable with the 512²
per-cell numbers. -->

## What the history means for you

- **Do not reopen pure Mamba, early temporal collapse, or hand-built motion
  features** without new evidence; each failed for a recorded reason.
- **Do not judge the gate in isolation.** Gate-only metrics looked alarming
  (40% of empty cells flagged) until the system measurement showed the
  classifier absorbs them. Measure changes end to end.
- **Checkpoint selection has bitten twice**: val-loss selection saved weak
  epochs in v0 and is why the shipped segmenter scores 0.391 instead of
  0.8985. Select on the metric you care about, and save the trajectory.
- **Single seeds mislead** for hard-binary gate training (the same config
  landed in two basins).
- **Names are historical.** The gate factory is `build_segmenter_v0_7`
  because Stage 1 began as a segmentation network; the older factories stay
  in `gate/model.py` for provenance. The segmenter was once called the
  "Stage 2 segmenter"; Stage 2 is now the classifier and the segmenter is
  Stage 3. Packages use role names (`gate`, `classifier`, `segmenter`).
- Scales: 256² (Phase A to Day 1), 128² (v0 to v0.9), 1024² (v0.7 - v0.8),
  512² (from the gate diagnostic on; every shipped number).

The full experimental record lives in the originating repository.
