# Runbook

> **This package ships no trained weights.** `checkpoints/` holds only the
> three `<name>.json` manifests. Day one, in order: restore the names (step
> 1), install (step 1), set the locations (step 2), run the tests (step 4),
> generate the data (step 5), retrain the three models (step 8), then
> evaluate your checkpoints (step 7) against the manifests' `val_metrics`
> with the tolerances of `docs/results.md`, section 7. Steps 3 and 6 need
> weight files: your own, or the original ones if they arrive separately
> (step 3 says how to check them; everything else then applies unchanged).

**What this is.** Copy-pasteable steps from a fresh checkout to every run the
project supports. Commands are bash unless marked PowerShell, and run from
the repository root.

**State.** Gate -> classifier is measured end to end on 512² synthetic data;
the segmenter is validated standalone only (shipped weights: epoch-19
checkpoint, val Dice 0.391), and its integration is open work. Steps 1-6 work
on Tier 0 (any OS, CPU). Steps 7-8 at full size need a GPU and the full
datasets: Tier 0 on GPU (Windows or Linux, no compiler; slower gate, and
gate training at 512² needs more than a 24 GB GPU) or Tier 1 from source
(Linux, fused kernels).

**Known problems you will hit** are listed at the step where they bite and
collected in `docs/limitations.md`.

## 1. Install

### Restoring the transferred tree (first, before any other command)

The transferred snapshot has `.txt` appended to every script and
executable-type file (`TRANSFER_RENAMES.txt` lists the pairs). Restore the
names first; the script verifies `MANIFEST.transfer.sha256`, renames back
without overwriting anything, verifies `MANIFEST.sha256`, and prints `PASS`
or `FAIL` (details: `README.md`, "Restoring the transferred tree"). Skip
this if the names already end in `.py`.

```bash
cp restore_names.sh.txt restore_names.sh && sh restore_names.sh   # Linux; expect: PASS: ...
```

```powershell
Copy-Item restore_names.ps1.txt restore_names.ps1                  # Windows
powershell -NoProfile -ExecutionPolicy Bypass -File .\restore_names.ps1   # expect: PASS: ...
```

### Install the environment

Tier 0 (any OS, CPU; details and Windows/conda variants in
`docs/environment.md`):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements/base.txt
python -m pip install -e . --no-deps
```

Tier 0 on GPU (Windows or Linux with an NVIDIA GPU, nothing compiled):
Tier 0 as above, then

```bash
python -m pip install torch==2.12.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
```

Tier 1 from source (Linux x86-64 or WSL2, NVIDIA GPU, CUDA 13.0 toolkit
with nvcc; prerequisites, build time and gotchas in `docs/environment.md`,
"Tier 1 from source"; the fused kernels do not build on Windows):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements/cuda-linux-x86_64-build.txt
MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 NVCC_THREADS=2 \
  python -m pip install --no-build-isolation -r requirements/cuda-linux-x86_64.txt
python -m pip install -e . --no-deps
```

Or build the same thing as a Docker image: `docker build -t large-scene-localizer:tier1 .`

If the fused kernels fail to import, the gate silently runs the slow
reference Mamba, so check the backend (below) before timing or training.

Check which Mamba backend you got (`reference` on Tier 0, CPU or GPU;
`mamba_ssm` on Tier 1):

```bash
python -c "import localizer.gate.blocks as b; print(b.MAMBA_BACKEND)"
```

## 2. Set the three locations

```bash
export LSL_DATA_ROOT=$PWD/work/data      # datasets: <root>/<dataset>/{data_all,label_msk}
export LSL_RUNS_ROOT=$PWD/work/runs      # training and evaluation output
export LSL_CKPT_ROOT=$PWD/checkpoints    # weight files next to their manifests (none ship; step 3)
mkdir -p "$LSL_DATA_ROOT" "$LSL_RUNS_ROOT"
```

PowerShell:

```powershell
$env:LSL_DATA_ROOT = "$PWD\work\data"
$env:LSL_RUNS_ROOT = "$PWD\work\runs"
$env:LSL_CKPT_ROOT = "$PWD\checkpoints"
New-Item -ItemType Directory -Force $env:LSL_DATA_ROOT, $env:LSL_RUNS_ROOT
```

Every script also takes `--data-root`, `--runs-root`, `--checkpoints-root`.

## 3. Verify the checkpoints (only if the weight files arrived separately)

No weight files ship with this package; on a fresh tree this check prints
`MISMATCH MISSING` for each of the three files, which is expected. If the original weights arrive
separately, put the three `<name>.safetensors` files in `$LSL_CKPT_ROOT`
under the names in the manifests and run the check: it must print `OK` three
times. The weight files are `state_dict`s in safetensors format, with no
metadata; `safetensors_sha256` in the manifest is their identity. With them
in place every later step applies unchanged, with the default checkpoint
names.

```bash
python - <<'EOF'
import hashlib, json, os, pathlib
root = pathlib.Path(os.environ["LSL_CKPT_ROOT"])
for m in sorted(pathlib.Path("checkpoints").glob("*.json")):
    man = json.loads(m.read_text())
    f = root / (man["name"] + ".safetensors")
    h = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "MISSING"
    print(f"{f.name:45s} {'OK' if h == man['safetensors_sha256'] else 'MISMATCH ' + h}")
EOF
```

## 4. Run the tests

```bash
python -m pytest tests/ -q
```

Expected as shipped (no weight files): Tier 0 (CPU): 195 passed,
53 skipped. Tier 0 on GPU: 221 passed, 27 skipped (the extra skips compare
against the fused kernels, which are absent). Tier 1 (GPU): 242 passed,
6 skipped. Four of the skips need the weight files (three manifest hash
checks and one real-checkpoint test). Two Tier 1 skips are 1024²
forward/backward shape tests, gated behind `LOCALIZER_RUN_1024_TESTS=1`;
they need more than 24 GB of GPU memory. With the original weight files in
`$LSL_CKPT_ROOT` (step 3) the four run: Tier 0 199 passed, 49 skipped;
Tier 0 on GPU 225 / 23; Tier 1 246 / 2.
To force CPU on a GPU machine: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q`.

## 5. Generate smoke data

Small sets for checking that code runs (not for measuring anything):

```bash
lsl-generate --config configs/data/multi_target_swarm_128.yaml --output_root $LSL_DATA_ROOT/smoke_128 \
    --total_scenes 40 --batch_size 20 --num_workers 2 --save_gifs false
lsl-generate --config configs/data/multi_target_swarm_512_v2.yaml --output_root $LSL_DATA_ROOT/smoke_512 \
    --total_scenes 10 --batch_size 2 --num_workers 5
```

The 128² set takes seconds; the 512² set about a minute with 5 workers.
Use at least 40 scenes at 128²: the classifier's stratified split fails with
fewer than two validation samples per class.

Full datasets, needed to retrain and for the full evaluation (details,
seeds and multi-machine slicing: `docs/data.md`):

```bash
lsl-generate --config configs/data/multi_target_swarm_512_v2.yaml     # gate + pipeline: 3,000 scenes, about 94 GiB
# swarm_128x128x64 (classifier + segmenter; about 59 GiB): six invocations, as originally generated
for s in "0 3334" "3334 3333" "6667 3333" "10000 6667" "16667 6667" "23334 6666"; do
  set -- $s
  lsl-generate --config configs/data/swarm_128.yaml --start_idx $1 --total_scenes $2
done
```

Original walls: the 512² set about 6,865 s with 8 workers (inferred); the
30k set about 88 CPU-hours (9,331 s on three machines with 20 workers each).

## 6. Load the checkpoints in a notebook

Works on every tier. Run from the repository root after step 2
and step 5. It needs weight files: the original ones (step 3), or your own
from step 8 (set the three file names below to their paths, for example
`$LSL_RUNS_ROOT/gate/A_stem2_bs1/models/best_gate_model.pth`; absolute paths
are used as given).

```python
import os, glob
import numpy as np
import torch
from localizer.common.checkpoint_io import load_weights
from localizer.gate.model import build_segmenter_v0_7
from localizer.classifier.model import Mover3DCNNTrans
from localizer.segmenter.model import VolumetricUNet

ck = os.environ["LSL_CKPT_ROOT"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
load = lambda m, f: load_weights(m, os.path.join(ck, f), map_location=dev).to(dev).eval()

GATE, CLF, SEG = ("gate_A_stem2_bs1_ep42.safetensors", "classifier_baseline_30k_ep64.safetensors",
                  "segmenter_volseg_ep19.safetensors")   # or the paths of your retrained files
gate = load(build_segmenter_v0_7(arm=2, nfeats=32, fusion="sum", stem_stride=2), GATE)
clf = load(Mover3DCNNTrans(t_dim=64, dropout_rate=0.5, attn_dropout=0.1, max_attn_len=2200,
                           head_type="flatten", return_logits=False, attn_type="full"), CLF)
seg = load(VolumetricUNet(nfeats=32), SEG)

# One 512² scene: uint8 (64, 512, 512) -> float (1, 1, 64, 512, 512) in [0, 1]
f = sorted(glob.glob(os.path.join(os.environ["LSL_DATA_ROOT"], "smoke_512", "data_all", "*.npy")))[0]
x = torch.from_numpy(np.load(f).astype(np.float32) / 255.0)[None, None].to(dev)

with torch.no_grad():
    logits = gate(x)                                         # (1, 1, 32, 128, 128)
    pos = torch.sigmoid(logits)[0, 0] > 0.5                  # (32, 128, 128)
    chips = pos.reshape(32, 4, 32, 4, 32).any(dim=(0, 2, 4)) # (4, 4) proposed 128 px chips
    for r, c in chips.nonzero().tolist():
        crop = x[..., r * 128:(r + 1) * 128, c * 128:(c + 1) * 128]   # (1, 1, 64, 128, 128)
        p = clf(crop).item()                                 # mover probability
        mask = torch.sigmoid(seg(crop)) > 0.5 if p > 0.5 else None   # segmenter: NOT part of the measured pipeline
        print(f"chip ({r},{c}) p_mover={p:.3f} mask_voxels={int(mask.sum()) if mask is not None else '-'}")
```

On CPU the gate forward pass on one 512² scene takes seconds and several GB of
RAM. The segmenter call is shown for completeness only: its weights are the
weak epoch-19 checkpoint and nothing upstream has been validated with it
attached.

## 7. Run the pipeline evaluation (gate -> classifier)

This needs weight files. With no weights shipped, train first (step 8) and
point the scripts at your checkpoints by path; the defaults name the
original files, which apply unchanged if they arrive separately (step 3):

```bash
G=$LSL_RUNS_ROOT/gate/A_stem2_bs1/models/best_gate_model.pth          # gate (written from epoch 30 on)
C=$LSL_RUNS_ROOT/classifier/baseline_30k/models/best_classifier.pt    # classifier
S=$LSL_RUNS_ROOT/segmenter/default/models/best_3d_unet_model.pth      # segmenter
```

On the smoke set (any tier; evaluates the 10% val split, here one scene).
The 1-epoch smoke run of step 8 writes no `models/best_gate_model.pth`
(the gate criterion starts at epoch 30), so use `models/last_epoch_model.pth`:

```bash
G=$LSL_RUNS_ROOT/gate/A_stem2_bs1/models/last_epoch_model.pth
lsl-system-eval --dataset smoke_512 --gate-ckpt A=$G --classifier-ckpt $C --out $LSL_RUNS_ROOT/system_eval_smoke.json
lsl-per-target-recall --dataset smoke_512 --gate-ckpt A=$G --classifier-ckpt $C --out $LSL_RUNS_ROOT/per_target_smoke.json
```

On the full 512² dataset (a GPU: Tier 1, or Tier 0 on GPU, whose gate
inference takes about twice as long). With your retrained checkpoints, add
`--gate-ckpt A=$G` and `--classifier-ckpt $C` to the first two commands,
use `--gate-ckpt A=$G` in the rescore and `--checkpoint $C` / `--checkpoint
$S` in the last two. The `# expect:` lines are the original weights'
results, which are also the retrain targets (manifests' `val_metrics`;
tolerances in `docs/results.md`, section 7). `specificity.py` exits with
code 2 whenever its counts differ from the original weights' counts, so
expect that with a retrained classifier unless it lands exactly on them:

```bash
lsl-system-eval --out $LSL_RUNS_ROOT/system_eval.json
# expect: A: system_recall=0.7147 FP/scene abs mean=0.00 ... S2spec@hardneg=0.988
lsl-per-target-recall --out $LSL_RUNS_ROOT/per_target_recall.json
# expect: A: per-target recall=0.9486 (3930/4143) lenient=0.9172 | loss: gate-miss=27 S2-reject=186
python scripts/gate/rescore_percell.py --gate-ckpt A=gate_A_stem2_bs1_ep42.safetensors --out $LSL_RUNS_ROOT/rescore.json
# expect: A: caught@any=1.000 ... cellRecall@any=0.948 absentFlag@any mean=25.76 p95=64.0
python scripts/classifier/specificity.py
# expect: TP=1494 TN=1498 FP=2 FN=6 (needs swarm_128x128x64 under $LSL_DATA_ROOT)
python scripts/segmenter/evaluate.py --config configs/segmenter/default.yaml \
    --checkpoint segmenter_volseg_ep19.safetensors --out $LSL_RUNS_ROOT/segmenter_eval.json
# expect: val: loss=0.0339 dice=0.3909 iou=0.3183 (needs swarm_128x128x64)
```

The defaults are the shipped checkpoints and `multi_target_swarm_512x512x64`;
override with `--gate-ckpt LABEL=FILE` (repeatable), `--classifier-ckpt FILE`,
`--dataset NAME`. The pipeline code assumes 512² scenes (a fixed 4 x 4 chip
grid). Tolerances for "reproduced" are in `docs/results.md`, section 7.
The last two checks need `swarm_128x128x64`; regenerate it with
`configs/data/swarm_128.yaml` exactly as described in `docs/data.md` (six
invocations, about 59 GiB; the segmenter check reads only indices 0-2499,
the classifier check a 3,000-scene val split drawn from all 30,000).

<!-- Expected lines: originating repo documents/notes/validation_2026_11_05_export.md
sections 2-5 (raw log lines). -->

## 8. Train

With no weights shipped, this is how you get checkpoints. One-epoch smoke
runs (any tier; prove the training path, not quality):

```bash
lsl-train-gate --config configs/gate/A_stem2_bs1.yaml --data_root $LSL_DATA_ROOT/smoke_128 \
    --subset_size 40 --num_epochs 1
lsl-train-classifier --config configs/classifier/baseline_30k.yaml --data_root $LSL_DATA_ROOT/smoke_128 \
    --num_epochs 1
lsl-train-segmenter --config configs/segmenter/default.yaml --data_root $LSL_DATA_ROOT/smoke_128 \
    --subset_size 40 --num_epochs 1
```

Full training with the shipped recipes (Tier 1; datasets from `docs/data.md`).
Tier 0 on GPU trains the classifier and segmenter at full speed (no Mamba
blocks). Its gate training at 512² needs 33.4 GiB of GPU memory (more than a
24 GB card) and runs about 3.4x slower per batch; see `docs/environment.md`:

```bash
lsl-train-gate --config configs/gate/A_stem2_bs1.yaml 2>&1 | tee gate_train.log            # 512², ~9 GiB GPU
lsl-train-classifier --config configs/classifier/baseline_30k.yaml 2>&1 | tee cls_train.log  # swarm_128x128x64
lsl-train-segmenter --config configs/segmenter/default.yaml 2>&1 | tee seg_train.log         # swarm_128x128x64
```

Times and targets (targets: the manifests' `val_metrics`; tolerances:
`docs/results.md`, section 7):

| Model | Time | Target |
|---|---|---|
| Gate | 50 epochs x about 1,478 s = about 20.5 h on a 24 GB GPU, Tier 1 (**estimate**: one epoch measured on an RTX PRO 5000 laptop GPU, times 50) | per-cell recall (k = 1) 0.9485, absent-cell flag rate 0.4024 (rescore); per-target(group) recall 0.9486 (3930 / 4143) with the classifier |
| Classifier | 1,052 s per epoch, **measured** in the original run on an NVIDIA GB10 (selected epoch 64 at about 18.7 h; all 100 planned epochs about 29 h, **estimate**); not timed on a 24 GB GPU | val accuracy 0.9973; TP / TN / FP / FN 1494 / 1498 / 2 / 6 (`scripts/classifier/specificity.py`) |
| Segmenter | 443 s per epoch, **measured** in the original run (host not recorded); 150 epochs = 18.4 h | val Dice 0.3909, IoU 0.3183 at the val-loss-selected epoch (`scripts/segmenter/evaluate.py`) |

Before you train, know that:

- **Gate:** the shipped recipe is seed-bistable; a retrain can land in the
  cautious basin (high miss rate). Run at least 3 seeds (`--seed`) and
  select on system recall, not the logged `cluster_recall`.
- **Classifier:** in the original run validation accuracy sat at chance
  (0.50-0.52) for the first 12 epochs before rising; a flat start is
  expected.
- **Gate, L3 recipe** (`configs/gate/ladder_L3.yaml`): needs about 35 GiB and
  collapses after about epoch 3; read it at its early peak
  (`best_stage1_seg_model_by_recall_at_15.pth`), not at the gate checkpoint.
- **Classifier and segmenter:** their dataset `swarm_128x128x64` regenerates
  from `configs/data/swarm_128.yaml` (six invocations, `docs/data.md`). The
  segmenter trainer selects on val loss, which is what produced the weak
  shipped checkpoint (in the original run Dice passed 0.8 at epoch 49 and
  peaked at 0.904 at epoch 125); see `docs/limitations.md` item 1.

Trainers log to stdout/stderr (hence `tee`); they write no log file.

## 9. Where outputs land

| Producer | Location |
|---|---|
| `lsl-generate` | `$LSL_DATA_ROOT/<dataset>/{data_all,label_msk}/` (config `dataset:`), or `--output_root` |
| `lsl-train-gate` | `$LSL_RUNS_ROOT/<run_name>/` (config `run_name:`, e.g. `gate/A_stem2_bs1`): `models/best_gate_model.pth` (gate criterion, when a qualifying epoch exists), `models/best_stage1_seg_model_by_recall_at_15.pth`, `models/last_epoch_model.pth`, `training_history.json`, `stats_*.csv`, `learning-curve_*.png`, `generated_files/` |
| `lsl-train-classifier` | `$LSL_RUNS_ROOT/classifier/<name>/`: `models/best_classifier.pt`, history JSON |
| `lsl-train-segmenter` | `$LSL_RUNS_ROOT/segmenter/<name>/`: `models/best_3d_unet_model.pth`, `training_history.json`, `stats_*.csv`, plots |
| evaluation scripts | the `--out` JSON you pass; summary lines on stdout |

`--data_root` / `--output_root` on the trainers override the dataset
directory and run directory directly.

To promote a new checkpoint, copy it to `$LSL_CKPT_ROOT` under a new name
and write a `checkpoints/<name>.json` manifest with its sha256, config,
data and metrics, following the existing ones. Do not overwrite or edit the
shipped manifests.
