# Architecture: the shipped system as built

**What this is.** A three-stage pipeline for finding small movers in a large
video volume. Stage numbering lives here and in the docs only; packages carry
role names: Stage 1 = `localizer.gate`, Stage 2 = `localizer.classifier`,
Stage 3 = `localizer.segmenter`.

**State.** Gate -> classifier is measured end to end on 512² synthetic data.
The segmenter is validated standalone only; its shipped weights are the
epoch-19 checkpoint (val Dice 0.391), and integrating it after the classifier
is open work.

The original design (`docs/concept/stage1_concept.pdf`, a bidirectional-Mamba
region-proposal network over a 1024² volume with motion-direction heads and
connected-component patch extraction) is the *original design, since
diverged*. Where it and this page disagree, this page describes the code.

## Data contract

Every dataset is a directory `<dataset>/{data_all,label_msk}/` produced by
`localizer.synth.swarm`:

| File | Content |
|---|---|
| `data_all/vid{idx}_{label}.npy` | `(T, H, W)` uint8 video, T = 64 |
| `label_msk/vid{idx}_{label}_mask.npy` | `(T, H, W)` uint8 truth mask (0 or 255), all zeros when `label` = 0 |

`label` is 1 (at least one target present) or 0 (target absent).
`localizer.common.data.MoverDataset` loads a pair as float32 scaled to
`[0, 1]` with a leading channel axis: `volume` `(1, T, H, W)`, `mask`
`(1, T, H, W)`, `label`. With `output_resolution: low_res` (the gate) it also
returns the truth mask max-pooled by `low_res_pool` = `(2, 4, 4)` over
`(T, H, W)`, plus the full-resolution mask as `mask_full`.

## Stage 1: gate (`localizer.gate`)

Factory `localizer.gate.model.build_segmenter_v0_7(arm=2, nfeats=32,
fusion="sum", stem_stride=2)`, class `Stage1V07Arm2CnnMambaLowRes`. (The
`segmenter` in the factory name is historical: the gate began as a
segmentation network. The other factories in `gate/model.py` are kept for
provenance and are not used by the shipped system.)

Shapes for one 512² scene (B = 1):

| Layer | Output `(B, C, T, H, W)` |
|---|---|
| input | `(1, 1, 64, 512, 512)` |
| `down1` stride-(1,2,2) stem, two 3x3x3 conv + BN + ReLU | `(1, 32, 64, 256, 256)` |
| `pool1` MaxPool3d(2) | `(1, 32, 32, 128, 128)` |
| `down2` DoubleConv3D | `(1, 64, 32, 128, 128)` |
| `mamba2` TemporalMambaBlock: forward + backward Mamba over T at every (h, w), summed, residual | `(1, 64, 32, 128, 128)` |
| `pool2`, `down3` | `(1, 128, 16, 64, 64)` |
| `pool3`, `down4` | `(1, 256, 8, 32, 32)` |
| `head` 1x1x1 conv -> 1 logit | `(1, 1, 8, 32, 32)` |
| trilinear resize to `(T/2, H/4, W/4)` | **`(1, 1, 32, 128, 128)` logits** |

So the output grid is `(2, 4, 4)` coarser than the input (one logit per
2 frames x 4 x 4 px), but the head's native resolution is 8 frames x
16 x 16 px; the last step only interpolates. The same code accepts any H, W
divisible by 32 (at 1024² the output is `(32, 256, 256)`). 3,580,705
trainable parameters.

Training target: the truth mask max-pooled to the same `(32, 128, 128)` grid,
weighted BCE with `pos_weight` set from `target_positive_pixels_per_frame`
(91 -> about 180 at 512²).

Mamba backend: `localizer.gate.blocks` imports the fused CUDA kernels from
`mamba_ssm` when installed, otherwise a pure-PyTorch reference
(`localizer.gate.mamba_ref`) with identical parameter names, so the same
`state_dict` loads either way. `LSL_FORCE_MAMBA_REF=1` forces the reference.
`localizer.gate.blocks.MAMBA_BACKEND` says which one is active.

<!-- Sources: shapes traced by forward hooks on build_segmenter_v0_7(arm=2,
stem_stride=2) at 128² input (down1 (1,32,64,64,64) ... head (1,1,8,8,8),
out (1,1,32,32,32)), scaled by the code's fixed strides; 512² output
(1,1,32,128,128) also stated in src/localizer/pipeline/system_eval.py
gate_chip_flags; 3,580,705 trainable params: originating repo
documents/notes/v0_7_step11_measurement.md ("Trainable params | 3,580,705")
and the same hook run; pos_weight 16384/91 = 180: configs/gate/A_stem2_bs1.yaml
comment. -->

## From gate logits to chips (the measured pipeline)

`localizer.pipeline.system_eval` and `localizer.pipeline.per_target_recall`
do this, per 512² scene:

1. Threshold `sigmoid(logits) > 0.5` on the `(32, 128, 128)` grid.
2. Partition the scene into a fixed **4 x 4 grid of 128 x 128 px chips**
   (32 x 32 cells of the low-res grid each). A chip is a **proposal** if any
   voxel in it, at any frame, is above threshold.
3. Crop each proposal chip from the input at full resolution:
   `(1, 1, 64, 128, 128)`, and run the classifier on it.
4. A chip is a **detection** if the classifier's probability is > 0.5.

This fixed-grid rule is what every system number in `docs/results.md` uses.

`localizer.gate.metrics.extract_proposals` is a different, finer-grained
proposal mechanism: 26-connected components of the thresholded logits, each
turned into a patch centred on the component centroid and clamped in bounds,
ranked by mean probability. It feeds the gate's training-time detection
metrics (`cluster_recall`, `recall@K` in `DetectionEpochAccumulator`); it is
**not** what the measured pipeline uses. Do not add post-processing on top of
gate output until the integration work calls for it.

## Stage 2: classifier (`localizer.classifier`)

`Mover3DCNNTrans(t_dim=64, attn_type="full", head_type="flatten",
return_logits=False)`: three stages of `Conv3d 3x3x3 + BN + ReLU +
MaxPool3d(2)` (1 -> 32 -> 64 -> 128 channels), a global relative-position
self-attention block over the `8 x 16 x 16 = 2048` bottleneck tokens, then an
MLP head with dropout. Input `(B, 1, 64, 128, 128)`, output `(B,)`
probabilities (sigmoid applied inside `forward`).

**The 128² chip contract.** The classifier only accepts 128 x 128 x 64 input:
the head's linear layer is sized for a 16 x 16 spatial bottleneck. It was
trained on whole 128² *scenes* (each holding at most one anomalous mover in a
background swarm) and is evaluated in the pipeline on 128² *crops of 512²
scenes*, where targets can sit off-centre, be cut by the chip border, or come
in clusters. That mismatch drives most of the system's residual loss; see
`docs/classifier.md`.

## Stage 3: segmenter (`localizer.segmenter`, not integrated)

`VolumetricUNet(nfeats=32)`: four-stage 3D U-Net encoder (DoubleConv3D +
MaxPool3d(2)), a bottleneck of a 1x1x1 patch-embed (256 -> 512 channels) and
two 3D Swin blocks (window `(2, 4, 4)`, 16 heads, the second shifted), and a
four-stage transposed-conv decoder with skip connections. Input
`(B, 1, 64, 128, 128)`, output `(B, 1, 64, 128, 128)` logits (no activation).
No code in the pipeline calls it yet; the intended position is after the
classifier, on chips the classifier accepts.

## What "region" and "target" mean

| Term | Meaning in this code and in every reported number |
|---|---|
| **chip** / region | One cell of the fixed 4 x 4 grid of 128 x 128 px tiles over a 512² scene. The pipeline code hardcodes this grid (`NCHIP = 4`), so it runs on 512² scenes only; 1024² would need an 8 x 8 grid. |
| **cell** | One of the 8 x 8 = 64 tiles of 64 x 64 px (16 x 16 low-res voxels, time-collapsed) used by the gate's per-cell scoring (`scripts/gate/rescore_percell.py`). |
| **target** | A 3D 6-connected component of the truth mask over `(T, H, W)`: a *group* of movers. Crossing or touching tracks merge into one group, so the group count is a lower bound on the mover count. |
| **caught** (per target) | At least one chip touched by the group is both proposed by the gate and accepted by the classifier ("any-chip"). The stricter "centroid" variant requires the chip under the group's centroid. |
| **false detection** | A classifier-accepted proposal chip that holds no truth voxel. |

<!-- Sources: chip grid, thresholds and target definition:
src/localizer/pipeline/system_eval.py (NCHIP 4, CHIP_IN 128, CHIP_LR 32, THR 0.5)
and src/localizer/pipeline/per_target_recall.py (target_components: 3D label of
mask_full, centroid chip); originating run artifact
analysis/stage1_v0_10_percell/per_target_recall.json meta.target_def
"3D ... (6-conn) of mask = connected target GROUP"; cell grid:
scripts/gate/rescore_percell.py docstring (8x8 = 64 cells, 16x16 prediction-voxel
blocks). -->

## Known to be wrong or unfinished here

- The segmenter is not wired in; the end-to-end pipeline stops at the
  classifier.
- The classifier's input contract is fixed at 128 x 128 x 64.
- The gate's per-cell discrimination is weak (it floods about 40% of cells on
  the average target-absent scene); the classifier absorbs it. Numbers in
  `docs/gate.md` and `docs/results.md`.
- 1024² is architecturally supported by the gate but not measured end to end.

Full list, ordered by leverage: `docs/limitations.md`.
