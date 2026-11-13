# Large-Scene Localizer: instructions for coding agents

**This snapshot ships no trained weights.** `checkpoints/` holds only the
three `<name>.json` manifests. Day one: restore the names, install, run the
tests (expect 195 passed / 53 skipped on Tier 0 CPU), generate the data,
retrain the three models, and compare against the manifests' `val_metrics`
(the targets) with the tolerances of `docs/results.md`, section 7. Commands,
times and targets: `README.md`, "No weights in this package: day one". Do not
invent or download weights; if the original weight files arrive separately,
they go in `checkpoints/` and must match the manifest hashes, and then
everything applies unchanged.

This file is read automatically by coding agents working in this repository.
It is written for this snapshot of the project, not for any particular
compute environment. Humans: start with `README.md` and `docs/`.

## What this is and where it stands

A staged learned pipeline that finds small (one to a few pixel) moving
targets in large overhead video volumes (frames x H x W; 64 x 512 x 512
measured, 64 x 1024 x 1024 the design target). All data is synthetic.

| Stage | Package | Role | Weights the numbers were measured with (manifest only) |
|---|---|---|---|
| 1 | `localizer.gate` | CNN + Mamba region proposal: which 128 px chips may hold movers | `gate_A_stem2_bs1_ep42` |
| 2 | `localizer.classifier` | 3D CNN + attention: mover / no mover per 128 x 128 x 64 chip | `classifier_baseline_30k_ep64` |
| 3 | `localizer.segmenter` | Volumetric U-Net + 3D Swin: per-voxel mask per chip | `segmenter_volseg_ep19` |

**Status, stated the same way everywhere:** gate -> classifier is measured
end to end on 512² synthetic data; the segmenter is validated standalone
only, its shipped weights are the epoch-19 checkpoint (val Dice 0.391), and
integration is open work.

Headline: 0.949 per-target(group) recall (3930 / 4143) at about 0 false
detections per scene on the 512² val split; 87% of the remaining loss is the
classifier rejecting off-distribution crops. "Target" means a 3D
connected-component group, not an individual mover. 1024² end to end is
unmeasured. The authoritative numbers, caveats and sources are
`docs/results.md`; open work in priority order is `docs/limitations.md`.
Never quote a number that is not in those pages or in a result file you just
produced.

## Layout

```
src/localizer/
  common/      data.py (MoverDataset, make_dataloaders), blocks.py, swin3d.py,
               metrics.py, plotting.py, paths.py (root resolution)
  gate/        model.py (all factories; build_segmenter_v0_7(arm=2, stem_stride=2) is shipped),
               blocks.py (Mamba blocks: mamba_ssm or the mamba_ref fallback), mamba_ref.py,
               train.py, metrics.py (DetectionEpochAccumulator, extract_proposals), losses.py
  classifier/  model.py, relative_attention.py, data.py, train.py
  segmenter/   model.py (VolumetricUNet), train.py
  synth/       swarm.py (generator; no torch, imports nothing else from localizer)
  pipeline/    system_eval.py, per_target_recall.py (gate -> classifier)
scripts/       thin CLI wrappers: gate/ classifier/ segmenter/ data/ pipeline/
configs/       gate/ classifier/ segmenter/ data/
checkpoints/   <name>.json manifests of the weights (the .safetensors files are not in this snapshot)
tests/         mirrors the package; tests/data/ holds small fixtures
docs/          the documentation set
requirements/, environment-tier{0,1}.yml, Dockerfile   pinned environments
```

Package dependency rule, enforced by `tests/test_layering.py`:
`pipeline -> {gate, classifier, common}`, `gate -> common`,
`classifier -> common`, `segmenter -> common`, `synth` isolated, `common`
imports nothing else from `localizer`, and no `__init__.py` imports anything.
Keep it that way.

## Locations

No tracked file may contain an absolute path, host name or IP address.
Configs name datasets and runs relatively (`dataset:
multi_target_swarm_512x512x64`, `run_name: gate/A_stem2_bs1`). Scripts
resolve them against three roots, from `--data-root` / `--runs-root` /
`--checkpoints-root` or else the environment:

```bash
export LSL_DATA_ROOT=/path/to/datasets      # <root>/<dataset>/{data_all,label_msk}
export LSL_RUNS_ROOT=/path/to/runs
export LSL_CKPT_ROOT=/path/to/checkpoints   # weight files named as in checkpoints/*.json
```

Never write generated data, run output or weights into the repository tree
outside `checkpoints/`.

## Environments

- **Tier 0** (any OS, CPU): `requirements/base.txt`; the gate uses the
  pure-PyTorch Mamba reference. Tests, inference, notebooks, small
  pipeline runs.
- **Tier 0 on GPU** (Windows, or Linux without a compiler): Tier 0 plus
  `torch==2.12.0+cu130`; reference Mamba on the GPU. Full-size inference
  and evaluation on 24 GB (gate about 1.9x slower). Gate training at 512²
  needs 33.4 GiB (more than a 24 GB card) and is about 3.4x slower per
  batch.
- **Tier 1 from source** (Linux x86-64 or WSL2 + CUDA 13.0 toolkit):
  `requirements/cuda-linux-x86_64-build.txt`, then
  `requirements/cuda-linux-x86_64.txt` with `--no-build-isolation`, which
  compiles `mamba-ssm` and `causal-conv1d` from their pinned sdists
  (`docs/environment.md`, "Tier 1 from source"; or the `Dockerfile`).
  Python 3.11 only. The fused kernels do not build on Windows.
- Tier 1: check that `MAMBA_BACKEND` is `mamba_ssm` before any timing or
  training run, since a failed kernel import silently falls back to the
  reference.
- The repository holds no binary wheels and no pickles; keep it that way.
  Weights are `.safetensors`; load them with
  `localizer.common.checkpoint_io.load_weights`, which also reads the `.pth`
  files the trainers write. `scripts/checkpoints/to_safetensors.py`
  converts a `.pth` with a round-trip check.
- Both pin torch 2.12.0. `localizer.gate.blocks.MAMBA_BACKEND` says which
  Mamba backend is active; `LSL_FORCE_MAMBA_REF=1` forces the reference.
- GPU memory: everything at 512² with the shipped models fits 24 GB; 1024²
  gate training needs about 32 GiB or more. Details: `docs/environment.md`.

## Commands

If the scripts here end in `.py.txt` / `.sh.txt`, this is the transferred
form: restore the names first (`cp restore_names.sh.txt restore_names.sh &&
sh restore_names.sh`, or the PowerShell variant in `README.md`, "Restoring
the transferred tree"), and expect `PASS`.

```bash
python -m pip install -r requirements/base.txt && python -m pip install -e . --no-deps   # Tier 0
python -m pytest tests/ -q                       # as shipped (no weights): Tier 0 195 passed / 53 skipped, Tier 0 on GPU 221 / 27, Tier 1 242 / 6 (with the original weights: 199 / 49, 225 / 23, 246 / 2)
lsl-generate --config configs/data/multi_target_swarm_128.yaml --output_root $LSL_DATA_ROOT/smoke_128 \
    --total_scenes 40 --batch_size 20 --num_workers 2 --save_gifs false
lsl-train-gate --config configs/gate/A_stem2_bs1.yaml               # Tier 1, ~20.5 h on a 24 GB GPU (estimate)
lsl-train-classifier --config configs/classifier/baseline_30k.yaml  # 1,052 s/epoch measured on a GB10
lsl-train-segmenter --config configs/segmenter/default.yaml         # 443 s/epoch measured, 150 epochs
# the evaluation commands below default to the original weight files; for
# retrained ones pass --gate-ckpt A=<path>, --classifier-ckpt <path>, --checkpoint <path>
lsl-system-eval --out system_eval.json           # gate -> classifier, 512² val split
lsl-per-target-recall --out per_target_recall.json
python scripts/gate/rescore_percell.py --gate-ckpt A=gate_A_stem2_bs1_ep42.safetensors --out rescore.json
python scripts/classifier/specificity.py
python scripts/segmenter/evaluate.py --config configs/segmenter/default.yaml --checkpoint segmenter_volseg_ep19.safetensors
```

`docs/runbook.md` has every step with expected output lines; the tolerances
that define "reproduced" are in `docs/results.md`, section 7. Run the test
suite before and after any change.

## Do not touch

- **`checkpoints/*.json` and the weight files they describe.** The manifests
  are the retrain targets and the identity of the original weights (not in
  this snapshot): `state_dict`s in safetensors format whose
  `safetensors_sha256` is in the manifest. To add a model (a retrained one
  included), convert it with `scripts/checkpoints/to_safetensors.py` and write
  a new manifest under a new name; never overwrite or edit these three.
- **The shipped configs** `configs/gate/A_stem2_bs1.yaml`,
  `configs/classifier/baseline_30k.yaml`, `configs/segmenter/default.yaml`
  and `configs/data/*.yaml`. They are the provenance of the shipped weights
  and data. Copy one to a new name to change a recipe.
- **The data configs' `batch_size`**: the generator seeds each batch with
  its start index, so changing it changes every scene.
- **The historical factories in `gate/model.py`**: unused by the shipped
  system but kept deliberately; do not trim them.
- **Gate output post-processing** beyond `extract_proposals` and the fixed
  chip grid, until segmenter integration calls for it.

## Known traps

- The shipped segmenter is the weak epoch-19 checkpoint (Dice 0.391); the
  often-quoted 0.8985 is an unsaved epoch. Retrain with Dice-based selection
  before integrating it.
- `swarm_128x128x64` (classifier and segmenter training data) regenerates
  bit-identically only with the six-invocation layout in `docs/data.md`; a
  single run from index 0 gives different scenes at most indices and
  slightly different classifier counts. Check dataset hashes before calling
  a mismatch a regression.
- Gate training with the shipped recipe is seed-bistable; use several seeds
  and select on system recall, not the trainer's `cluster_recall`.
- The pipeline code assumes 512² scenes (4 x 4 chip grid) and 128 x 128 x 64
  chips.

## Conventions

- Python >= 3.10, type hints on signatures, full docstrings (purpose, args
  with shapes, returns).
- NumPy vectorized; `numpy.random.Generator` with per-scene
  `default_rng(seed)`; no global RNG state.
- Configs are YAML loaded into dataclasses; no magic numbers in code.
- `pytest` on small volumes; tests mirror the package under `tests/`; mark
  GPU-only tests and skip them without CUDA.
- ASCII only in code; no decorative characters.
- Every reported number cites the file it came from.
