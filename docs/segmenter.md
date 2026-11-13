# Stage 3: the segmenter (`localizer.segmenter`)

> **This package ships no trained weights.** The weights described on this
> page (`segmenter_volseg_ep19.safetensors`) are not included; `checkpoints/segmenter_volseg_ep19.json`
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
| Weights measured (not in this package) | `segmenter_volseg_ep19.safetensors`, 73,659,692 bytes |
| sha256 (identity) | `104d3543e47684375a3463cde1874cf9dfd025323482efcb9b78aebf34034578` |
| Converted from | `segmenter_volseg_ep19.pth` (PyTorch, 73,699,819 bytes, sha256 `a039922d...21a4f`; not shipped), tensor-for-tensor identical |
| Manifest | `checkpoints/segmenter_volseg_ep19.json` |
| Config that produced it | `configs/segmenter/default.yaml` (hyperparameters identical to the training-time config at commit `609f1fe`; only data/run locations were made relative) |
| Build | `VolumetricUNet(nfeats=32)` |

## What it is

A per-voxel mover segmenter for one 128 x 128 x 64 chip: a four-stage 3D
U-Net (32 -> 256 channels) whose bottleneck is a 1x1x1 patch-embed to 512
channels followed by two 3D Swin transformer blocks (window `(2, 4, 4)`,
16 heads, the second shifted), and a transposed-conv decoder with skip
connections. Input `(B, 1, 64, 128, 128)` in `[0, 1]`; output
`(B, 1, 64, 128, 128)` logits; threshold `sigmoid > 0.5` for a mask.

Its intended place is after the classifier, on the chips the classifier
accepts. **Nothing in the pipeline calls it yet.**

## What it was trained on

- Data: the first 2,500 scenes (by index) of `swarm_128x128x64` (the same
  30k whole-128²-scene set as the classifier; 10,000 were on disk at the
  time), split 2,250 train / 250 val by `val_split 0.1`, `val_split_seed 0`.
- Recipe: AdamW, lr 1e-3, weight decay 1e-2, batch 8, 150 epochs,
  ReduceLROnPlateau on val loss (factor 0.5, patience 10), BCE-with-logits
  with `pos_weight = (128 * 128) / 13`, about 1260. Seed 0.
- Checkpoint selection: **best val loss**, which happened at epoch 19.
- Training stack: not recorded (host and torch version unknown).

<!-- Sources: checkpoints/segmenter_volseg_ep19.json (bytes, sha256,
config_commit, selection_criterion "best val_loss ... epoch 19", host null,
torch null, dataset "first 2,500 scenes by index ... val 250"); configs/segmenter/default.yaml
(subset_size 2500, val_split 0.1, val_split_seed 0, batch_size 8,
num_epochs 150, lr, weight_decay, scheduler_*, target_positive_pixels_per_frame 13);
src/localizer/segmenter/train.py (AdamW, ReduceLROnPlateau, pos_weight
formula); val N = 250 (subset_size=2500, val_split=0.1, seed=0): originating
repo documents/notes/validation_2026_11_05_export.md section 5. -->

## What it has been measured to do (and on what)

On its 250-scene val split (whole 128² scenes, target-present and
target-absent mixed):

| Weights | val loss | val Dice | val IoU | Exists on disk? |
|---|---|---|---|---|
| **epoch 19 (shipped)** | 0.0339 | **0.391** | 0.318 | yes |
| epoch 150 (end of run) | 0.0892 | 0.8985 | 0.8403 | **no**, never saved |

The often-quoted "Dice 0.8985" is the epoch-150 training-log value. The run
saved weights only when val loss improved, which last happened at epoch 19;
val loss then rose while Dice kept climbing, so the good weights were never
written. **No shipped artifact scores 0.8985.**

Metric definition: mean over samples of per-sample Dice (and IoU) at
threshold 0.5, with a 1e-6 smoothing term. A target-absent sample scores 1
if the prediction is empty and about 0 otherwise, so the number mixes mask
quality on target scenes with false firing on empty ones. Present-only and
absent-only Dice were not reported.

**Reproduction on x86.** `scripts/segmenter/evaluate.py` on Linux x86-64,
Tier 1, reproduces the epoch-19 line exactly: loss 0.0339, Dice 0.3909,
IoU 0.3183 (full precision: Dice 0.390946533203125).

<!-- Sources: checkpoints/segmenter_volseg_ep19.json val_metrics
(at_checkpoint_epoch_19: val_loss 0.0339, val_dice 0.3909, val_iou 0.3183;
at_final_epoch_150_not_saved: val_loss 0.0892, val_dice 0.8985, val_iou 0.8403)
and reproduced_on_x86; full-precision Dice: originating run artifact
runs/validation_export_2026_11_05/results/segmenter_eval.json via the
validation record section 5; metric definition:
src/localizer/common/metrics.py get_dice_batch_sum (smooth 1e-6). -->

## What is known to be wrong with it

1. **The shipped weights are a weak checkpoint.** Val-loss selection picked
   epoch 19 (Dice 0.391). The model reached Dice 0.8985 by epoch 150, but
   those weights do not exist. The fix is a retrain with Dice-based
   checkpoint selection (ideally present-scene Dice, reported separately
   from absent-scene behaviour), saving per-epoch history and val
   predictions. This is the first item in `docs/limitations.md`.
2. **Not integrated.** There is no code path from classifier-accepted chips
   to the segmenter, and no end-to-end segmentation metric on 512² scenes.
3. **Same distribution mismatch as the classifier.** Trained on whole 128²
   scenes with one anomalous mover; in the pipeline it would see 128² crops
   of larger multi-target scenes.
4. **Training stack unknown.** Host and software versions were not
   recorded for the original run.

## Run it

Evaluate the original checkpoint on its val split (needs `swarm_128x128x64`
under `$LSL_DATA_ROOT` and the weight file, not shipped; for your own
checkpoint pass its path to `--checkpoint`):

```bash
python scripts/segmenter/evaluate.py --config configs/segmenter/default.yaml \
    --checkpoint segmenter_volseg_ep19.safetensors --out segmenter_eval.json
```

### Retrain

With the shipped recipe (selects on val loss as shipped; see item 1; any
tier with a GPU, no Mamba blocks):

```bash
lsl-train-segmenter --config configs/segmenter/default.yaml 2>&1 | tee seg_train.log
```

- Data: the first 2,500 scenes of `swarm_128x128x64` (`docs/data.md`).
- Time: 443 s per epoch, **measured** in the original run (host not
  recorded); 150 epochs = 18.4 h.
- Target (`checkpoints/segmenter_volseg_ep19.json` `val_metrics`): val loss
  0.0339, Dice 0.3909, IoU 0.3183 at the val-loss-selected epoch (epoch 19 in
  the original run). The same run reached Dice 0.8985 at epoch 150 (Dice
  passed 0.8 at epoch 49, peaked at 0.904 at epoch 125), but the recipe
  saves only the val-loss best.
- Score your checkpoint:
  `python scripts/segmenter/evaluate.py --config configs/segmenter/default.yaml --checkpoint $LSL_RUNS_ROOT/segmenter/default/models/best_3d_unet_model.pth`.

Load it in Python (with the weight file in `checkpoints/`):

```python
from localizer.common.checkpoint_io import load_weights
from localizer.segmenter.model import VolumetricUNet
m = load_weights(VolumetricUNet(nfeats=32), "checkpoints/segmenter_volseg_ep19.safetensors").eval()
```
