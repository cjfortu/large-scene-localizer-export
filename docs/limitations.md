# Limitations and open work, ordered by leverage

**What this is.** Everything known to be wrong, missing or unmeasured, with
the most valuable fix first. Each item says what is wrong, the evidence, and
the indicated next step.

**State.** Gate -> classifier is measured end to end on 512² synthetic data;
the segmenter is validated standalone only, its shipped weights are the
epoch-19 checkpoint (val Dice 0.391), and integration is open work. Nothing
on this list has been started.

**This package ships no trained weights** (only the manifests in
`checkpoints/`), so every item below starts from a retrain of the three
models with the shipped recipes (`README.md`, "No weights in this package:
day one"). Items 1 and 2 change those recipes; do the plain retrain first to
confirm you hit the manifests' targets.

## 1. Retrain the segmenter with Dice-based checkpoint selection

**Wrong.** The shipped segmenter weights are the epoch-19 best-val-loss
checkpoint, val Dice 0.391. The same run reached val Dice 0.8985 at epoch
150, but val loss rose while Dice climbed, the trainer only saved on a new
val-loss minimum, and the good weights were never written.

**Next.** Retrain `configs/segmenter/default.yaml` selecting on val Dice
(better: present-scene Dice, with absent-scene false firing reported
separately, since the current per-sample Dice scores an empty prediction on
an empty scene as 1). Save per-epoch history and val predictions. Ideally
train on deployment-style crops (item 2) so the segmenter sees what the
pipeline will feed it. This unblocks item 3.

<!-- Source: checkpoints/segmenter_volseg_ep19.json val_metrics
(at_checkpoint_epoch_19 val_dice 0.3909; at_final_epoch_150_not_saved
val_dice 0.8985, val_loss 0.0892 vs 0.0339) and notes. -->

## 2. Retrain the classifier on deployment-style crops

**Wrong.** The classifier was trained on whole 128² scenes and runs in the
pipeline on 128² crops of 512² scenes (off-centre, edge-cut, sparse or
clustered targets). 186 of the 213 target groups the system loses (87%) are
classifier rejects; only 27 are gate misses. Most lost groups (168 of 213)
span a single chip.

**Next.** Retrain or fine-tune on crops cut exactly as the pipeline cuts them
(the gate's 4 x 4 grid of 128 px chips over 512² scenes, including
gate-proposed non-target chips as negatives). Save the checkpoint *and*
per-sample val predictions. Expect recall to rise and the near-zero false
detection rate to move; re-measure both with `lsl-per-target-recall` and
`lsl-system-eval`. Recovering half of the classifier rejects would put
per-group recall near 0.97.

<!-- Source: originating run artifact analysis/stage1_v0_10_percell/
per_target_recall.json results.A (stage2_reject 186, gate_miss 27, lost 213,
footprint_lost.1chip 168); (3930 + 93) / 4143 = 0.971. -->

## 3. Integrate the segmenter after the classifier

**Missing.** No code passes classifier-accepted chips to the segmenter, and
there is no end-to-end segmentation metric on 512² scenes. Do this after
item 1 (integrating the epoch-19 weights would measure a known-weak model).
Define the end-to-end metric first (per-voxel Dice over accepted chips, and
per-target mask coverage), then extend `localizer.pipeline`.

## 4. Measure per-mover recall

**Unmeasured.** Every system number counts targets as 3D connected-component
*groups* of the truth mask; crossing or touching tracks merge, so a group of
several movers counts as caught if one chip of it is detected. Per-mover
recall could be lower than 0.949. Decide whether the deployment unit is the
mover or the group, then measure it (individual mover tracks have to be
recovered from the generator, which knows them at generation time but does
not save them).

## 5. 1024² end to end

**Unmeasured.** All system numbers are 512². The gate architecture runs at
1024² (output `(32, 256, 256)`) and was trained there in earlier rounds, but
the shipped gate has never been evaluated end to end at 1024², gate
inference memory and wall time at 1024² are unmeasured, the pipeline code
hardcodes the 512² chip grid (`NCHIP = 4`), and training the shipped
architecture at 1024² needs 31.8 GiB, more than a 24 GB GPU. Treat 1024² as a
system question: end-to-end recall with a distribution-matched classifier,
selecting any gate on system recall.

<!-- Source: originating repo documents/notes/v0_7_step11_measurement.md
("Peak alloc (real data) 31.82 GiB"); src/localizer/pipeline/system_eval.py
NCHIP = 4. -->

## 6. Next gate decoder round (gate miss rate < 1%)

**Not reached.** Cheap levers at 512² (replicate padding, stride-1 stem,
cell-focal loss) did not reach per-cell miss < 1% with absent-cell flagging
< 25%; the best miss rate was 2.2%, at 41.5% flagging. The indicated lever is
a full-resolution U-Net decoder carrying the cell-focal max-pool loss and
head bias-init of `configs/gate/ladder_L3.yaml`. Two methodology fixes go
with it: the stride-1 stem collapses after about epoch 3, so use early
stopping and a low `gate_settle_min_epoch` (about 3-5); and the cause of the
collapse is not understood. Lower priority than items 1-5 while the
classifier absorbs gate false firing.

<!-- Source: analysis/stage1_gate_ladder_512/rescore_L2_L3_early.json
results.L2.per_k["1"] (per_cell_recall_strict 0.97768 -> 2.2% miss;
absent rate 0.41502) and the ladder results record sections 4 and 6. -->

## 7. Seed bistability of hard-binary gate training

**Wrong.** Two runs with the same config, node and software environment
landed in opposite basins: the shipped gate (per-cell miss 5.2%,
absent-cell flagging 40.2%) and its twin (28.4%, 15.3%). The trigger is
seed or GPU (Mamba kernel) non-determinism and is unidentified. Any
single-seed comparison of hard-binary gate arms carries regime-level noise.
Run at least 3 seeds per arm; the cell-focal loss (L3) is less exposed
because its absent side is pinned by construction.

<!-- Source: analysis/stage1_v0_10_percell/rescore.json results.A and
analysis/stage1_gate_ladder_512/rescore_L0_L1_border.json results.L0
(per_cell_recall_strict 0.71617, absent rate 0.15282); ladder record section 4. -->

## 8. Border-ring flood in the gate (optional fix)

**Wrong.** About 45% of the shipped gate's false cell flags on empty scenes
are the outer ring of the 8 x 8 grid, an architectural boundary effect (zero
padding and the stride-2 stem); the rest is interior. Replicate padding made
the gate fire more, not less. Fixing it (a different padding scheme, or
cropping the border at evaluation) would improve gate precision; optional
while the classifier absorbs the flood.

<!-- Source: analysis/stage1_v0_10_percell/border_attribution.json results.A
interior_absent_flagged_mean 14.134 of 25.756 (rescore.json) -> ring 45.1%;
rescore_L0_L1_border.json L1 vs L0 absent rate 0.41159 vs 0.15282. -->

## 9. Two historical gate arms ran without `causal_conv1d`

**Caveat on history, not on shipped weights.** Arms B1 (EMA) and L2
(stride-1 stem) trained on a node without `causal_conv1d`, so `mamba_ssm`
used its non-fused convolution path. Comparisons of B1 or L2 against the
other arms carry that numerical-path difference. The shipped gate (A) used
the fused kernel.

## 10. Checkpoints carry no metadata

The weight files are bare tensor maps (`state_dict`s in safetensors format):
no architecture, config or epoch inside. Their identity is
`safetensors_sha256` in `checkpoints/<name>.json`; three other classifier
runs produced files of the same size with different weights. Always verify
by hash (runbook step 3). The trainers still write PyTorch `.pth` / `.pt`
files; `scripts/checkpoints/to_safetensors.py` converts one and proves the
round trip, and every loader accepts both formats (preferring
`.safetensors`). The gate's stride-2 stem keys (`down1.down.*`) are not compatible
with stride-1 builds.

## 11. Selection metrics in the gate trainer are loose

The gate trainer's `cluster_recall` / `recall@K` credit a proposal landing
near a target among the top K (near-trivial once the gate fires), and
`abs_pos_frac` is a voxel fraction rather than a proposal count. The shipped
gate was selected on them. Select future gates on system recall
(`lsl-per-target-recall`).

## 12. Fixed shapes in the pipeline

The classifier and segmenter accept only 128 x 128 x 64 chips (the
classifier head is sized for it); the pipeline evaluation assumes 512²
scenes and 64 frames.

## 13. Environment gaps

- Gate training on the shipped Tier 1 stack has been timed for one full
  512² epoch, not run to convergence; full training reproducibility on
  Tier 1 is unmeasured.
- The fused Mamba kernels do not build on Windows; Windows runs Tier 0 (CPU
  or GPU, reference Mamba) or Tier 1 inside WSL2.
- Tier 1 compiles the kernels locally (CUDA 13.0 toolkit, network access to
  the package indexes or a mirror); the build covers GPU architectures
  sm_75 to sm_121 (no pre-Turing) and the hash pins cover CPython 3.11
  only. If the kernels fail to import, the gate silently falls back to the
  slow reference; check `MAMBA_BACKEND`.
- Training the gate with the reference Mamba (Tier 0 on GPU) needs 33.4 GiB
  at 512² (out of memory on a 24 GB card at about 21.4 GiB) and was timed
  only over 300 batches on a GB10 (about 3.4x slower per batch than the
  `mamba_ssm` path there); its epoch time is extrapolated. Its training
  memory could be cut with a checkpointed or chunked selective scan
  (recompute the scan states in the backward pass instead of storing them);
  not done.
- Tier 0 on CPU has not been timed at 512².

## 14. Synthetic data only

All data is abstract synthetic swarm video: 2 x 2 px movers in a uniform
swarm, no sensor model, clouds, parallax or jitter. Nothing has been
measured on real imagery, and the generator's bit-exactness is measured for
one numpy version on two CPU architectures.
