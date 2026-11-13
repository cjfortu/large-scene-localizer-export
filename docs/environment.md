# Environment: install tiers, operating systems, GPUs

> **This package ships no trained weights.** `checkpoints/` holds only the
> three `<name>.json` manifests. Day one: restore the names, install, run the
> tests, generate the data, retrain (`README.md`, "No weights in this
> package: day one"); the manifests' `val_metrics` are the targets, with the
> tolerances of `docs/results.md`, section 7. If the weight files arrive
> separately, put them in `checkpoints/`, check them against the manifest
> hashes (`docs/runbook.md`, step 3), and everything else applies unchanged.

**What this is.** How to get a working environment, what each environment can
do, how fast it is, and what the original training stack was.

**State.** Three ways to run, all pinned to torch 2.12.0 and all verified:
Tier 0 on CPU (any OS), Tier 0 on a GPU (Windows or Linux, CUDA torch, no
compiler needed), and Tier 1 from source (Linux x86-64, compiles the fused
Mamba kernels locally). Every published number reproduced exactly on Tier 1
with the original weights (`docs/results.md`, section 7). This snapshot
contains no weight files, no binary wheels and no pickles: the weights are
described by their manifests (they were `.safetensors` files), and the two
compiled packages are source pins. To train, use Tier 1: it is the only tier
that trains the gate at 512² on a 24 GB card.

**The fused Mamba kernels do not build on Windows.** `mamba_ssm` and
`causal_conv1d` are Linux-only upstream. On Windows, use Tier 0 (CPU or GPU)
natively, or run Tier 1 inside WSL2, which counts as Linux.

Pipeline status, for context: gate -> classifier is measured end to end on
512² synthetic data; the segmenter is validated standalone only (shipped
weights: epoch-19 checkpoint, val Dice 0.391), and its integration is open
work.

## The tiers

| | Tier 0 (CPU) | Tier 0 on GPU | Tier 1 from source |
|---|---|---|---|
| OS | Windows (incl. JupyterLab), Linux, macOS | Windows or Linux | Linux x86-64 (Windows: WSL2) |
| Python | 3.11 pinned set; 3.12 and 3.13 also tested (below) | 3.11 | 3.11 (the `--hash` pins cover CPython 3.11 only) |
| Hardware | CPU | NVIDIA GPU, driver supporting CUDA 13.0; more than 40 GB to train the gate at 512² | NVIDIA GPU, compute capability 7.5 or newer, driver supporting CUDA 13.0 |
| torch | 2.12.0+cpu | 2.12.0+cu130 | 2.12.0+cu130 |
| Mamba blocks | pure-PyTorch reference (`localizer.gate.mamba_ref`) | pure-PyTorch reference, on the GPU | fused `mamba_ssm 2.3.2.post1` + `causal_conv1d 1.6.2.post1`, compiled locally |
| Compiler needed | no | no | CUDA 13.0 toolkit (nvcc) and a C++ compiler |
| Pins | `requirements/base.txt`, `environment-tier0.yml` | `requirements/base.txt` + the cu130 torch | `requirements/cuda-linux-x86_64-build.txt`, `requirements/cuda-linux-x86_64.txt`, `environment-tier1.yml`, `Dockerfile` |
| Use for | tests, loading checkpoints, inference, notebooks, small pipeline runs | full-size inference and evaluation (24 GB is enough); gate training at 512² only on a GPU with more than 40 GB, at about 1/3.4 the speed | training, full-speed evaluation |
| Verified | CPU suite 195 passed, 53 skipped as shipped (no weight files; GPU tests skip); 199 / 49 with the original weights under `LSL_CKPT_ROOT`; same on Python 3.11, 3.12, 3.13 | GPU suite 221 passed, 27 skipped as shipped (225 / 23 with the weights): the skips are the 21 tests comparing against the `mamba_ssm` kernels, the two 1024² tests and the four weight-file tests | GPU suite 242 passed, 6 skipped as shipped (246 / 2 with the weights; the two 1024² tests always skip); all published numbers reproduced with the original weights; Docker image built from scratch, fused backend confirmed |

### Speed and memory: reference Mamba vs the kernels

The reference Mamba has the same parameter names and shapes as `mamba_ssm`'s
`Mamba`, so every checkpoint loads in every tier. On the shipped gate it
gives the same cell, chip and group decisions as the fused kernels (voxel
logits differ by at most 0.18 on logits up to about 1795). It is slower,
and for training it needs far more GPU memory.

**Training the gate at 512² with the reference Mamba (Tier 0 on GPU) needs
33.4 GiB of GPU memory allocated** (shipped recipe, batch 1), against 8-9 GiB
with the kernels. It therefore needs a GPU with more than 40 GB of memory
(a 48 GB card or larger). On a 24 GB card it runs out of memory on the first
batch at about 21.4 GiB allocated; bf16 and gradient checkpointing of the
CNN blocks do not change that. Inference and evaluation with the reference
fit a 24 GB card.

Inference, measured on a 24 GB laptop GPU (NVIDIA RTX PRO 5000 Blackwell):

| Workload (512², shipped gate) | Fused kernels (Tier 1) | Reference (Tier 0 on GPU) | Ratio |
|---|---|---|---|
| Per-cell rescore of 300 val scenes | 43 s | 81 s | 1.9x |

Training, shipped gate recipe at 512², batch 1. Seconds per batch are
measured over 300 batches after 20 warm-up batches; they cover data
loading, forward, loss, backward, gradient clipping and the optimizer step.

| GPU | Mamba path | s / batch, mean (p95) | Peak allocated | One epoch (2,700 train + 300 val scenes) |
|---|---|---|---|---|
| RTX PRO 5000 laptop, 24 GB (x86-64) | fused kernels (Tier 1) | 0.393 (0.408) | 8.05 GiB | 1,478 s (24.6 min), measured |
| RTX PRO 5000 laptop, 24 GB | reference (Tier 0 on GPU) | out of memory on batch 1 | > 21.4 GiB | does not run |
| NVIDIA GB10, 128 GB unified (aarch64) | `mamba_ssm` kernels, non-fused convolution (no `causal_conv1d` on that node) | 0.901 (0.907) | 9.00 GiB | about 2,780 s (46 min), extrapolated |
| NVIDIA GB10, 128 GB unified | reference | 3.053 (3.061) | 33.35 GiB | about 8,900 s (2.5 h), extrapolated |

**Same hardware (GB10), extrapolated: the reference trains about 3.4x slower
per batch** than the `mamba_ssm` path (3.053 / 0.901 s). That path lacked the
fused convolution kernel, so the ratio against the fully fused kernels is
probably somewhat higher (not measured). The GB10 epoch times are
**extrapolated**: 2,700 x the mean time per batch, plus validation. For
validation, the `mamba_ssm` row takes 348 s from the original training run on
a GB10, and the reference row takes that 348 s scaled by the inference ratio
81 / 43. The method checks out on the `mamba_ssm` row: about 2,780 s
extrapolated against 2,731-2,780 s measured per epoch in the original GB10
training run. On CPU the reference is much slower still (not measured at
512²); do not train the gate at 512² on CPU.

`localizer.gate.blocks.MAMBA_BACKEND` reports the active backend
(`mamba_ssm` or `reference`); a warning is logged when the reference is used.
`LSL_FORCE_MAMBA_REF=1` forces the reference even when the kernels are
installed.

<!-- Sources: originating repo validation record
documents/notes/validation_2026_11_05_export.md sections 1-2 and 9 (43 s /
81 s rescore walls, gate_logit_diff max_abs 0.1844); the export v3 record
documents/notes/validation_2026_11_06_snapshot.md section 6 (training
timings, test counts, from-source build). -->

## Install: Tier 0 (CPU)

Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/base.txt      # macOS: edit the torch line to torch==2.12.0 first
python -m pip install -e . --no-deps
python -c "import localizer.gate.blocks as b; print(b.MAMBA_BACKEND)"   # prints: reference
```

Windows (PowerShell), including a JupyterLab kernel:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements\base.txt
python -m pip install -e . --no-deps
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name lsl-tier0
```

Conda (any OS):

```bash
conda env create -f environment-tier0.yml
conda activate lsl-tier0
```

## Install: Tier 0 on GPU (Windows, or Linux without a compiler)

Tier 0, then the CUDA build of the same torch release in place of the CPU
build. Nothing is compiled. Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements\base.txt
python -m pip install torch==2.12.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
python -m pip install -e . --no-deps
python -c "import torch, localizer.gate.blocks as b; print(torch.__version__, torch.cuda.is_available(), b.MAMBA_BACKEND)"
# expected: 2.12.0+cu130 True reference
```

Linux: the same commands with `python3.11 -m venv .venv`,
`source .venv/bin/activate` and forward slashes. The GPU needs an NVIDIA
driver supporting CUDA 13.0; the torch cu130 packages bring their own CUDA
runtime. Every script then uses the GPU with the reference Mamba. Inference
and evaluation at full size work on a 24 GB card. Gate training at 512²
needs 33.4 GiB (a GPU with more than 40 GB) and runs about 3.4x slower per
batch than with the kernels. The classifier and segmenter have no Mamba
blocks and train at full speed.

## Install: Tier 1 from source (Linux x86-64, incl. WSL2)

The two kernel packages are compiled on your machine from the PyPI sdists
pinned by sha256 in `requirements/cuda-linux-x86_64.txt`. Every other pin
in both Tier 1 files also carries its `--hash` values, so pip verifies every
download (pip's hash-checking mode applies to the whole file once any pin has
a hash).

### Prerequisites

- **Linux x86-64**, or WSL2 (Ubuntu) on Windows with the NVIDIA Windows
  driver; install the "WSL-Ubuntu" CUDA toolkit inside WSL2, never a Linux
  display driver.
- **CUDA 13.0 toolkit with `nvcc`** on `PATH` (or `CUDA_HOME` set). The
  major version must match torch's CUDA (13): torch's extension builder
  refuses a mismatched toolkit. Validated: nvcc 13.0 from
  `nvidia/cuda:13.0.2-devel-ubuntu24.04` (gcc 13.3) and from
  `nvidia/cuda:13.0.2-devel-ubuntu22.04` (gcc 11.4).
- **A C++ compiler nvcc 13.0 accepts** (gcc 11 to 13 validated) and the
  **Python 3.11 headers** (`python3.11-dev` on Debian/Ubuntu).
- **Python 3.11**, in a venv or conda env.
- **torch installed first**, with its CUDA 13 pip packages (`nvidia-*`,
  `cuda-toolkit`), plus the build tools `setuptools`, `wheel`, `packaging`
  and **`ninja`** (without ninja the build is serial and much slower). All
  of these come from `requirements/cuda-linux-x86_64-build.txt` in step 1.
- Network access to PyPI and `download.pytorch.org` (or a mirror of both).
- No GPU is needed to build; the NVIDIA driver is needed to run.

### Commands

From the repository root:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
nvcc --version                                   # must say release 13.0

# 1. torch 2.12.0+cu130, its CUDA 13 packages and the build tools (hash-checked)
python -m pip install -r requirements/cuda-linux-x86_64-build.txt

# 2. everything else; compiles mamba-ssm and causal-conv1d against that torch
MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 NVCC_THREADS=2 \
  python -m pip install --no-build-isolation -r requirements/cuda-linux-x86_64.txt

# 3. this package
python -m pip install -e . --no-deps
```

### Build time

Measured on an x86-64 laptop workstation, in the `Dockerfile` (the same
three steps), from scratch, with the build pinned to 8 CPU threads (the
performance cores) and `MAX_JOBS=4 NVCC_THREADS=2`:

| Stage | Wall time |
|---|---|
| `causal-conv1d` compile | 5 min 32 s |
| `mamba-ssm` compile | 6 min 29 s |
| Both kernels (step 2's compile) | **12 min 01 s** |
| Whole image from scratch (apt, Python, downloads of about 3 GB, compile, install) | 16 min 07 s |

The compile is CPU-bound; more cores with a higher `MAX_JOBS` shorten it.
An earlier build of the same sdists in `nvidia/cuda:13.0.2-devel-ubuntu22.04`
(gcc 11.4) with the same settings took about 13 minutes.

### Confirm the backend

```bash
python -c "import torch, localizer.gate.blocks as b; print(torch.__version__, torch.cuda.is_available(), b.MAMBA_BACKEND)"
# expected: 2.12.0+cu130 True mamba_ssm
```

If `mamba_ssm` fails to import for any reason, the gate **silently falls
back to the slow pure-PyTorch reference** (a warning is logged). Check that
this prints `mamba_ssm` before timing or training anything.

### Gotchas

- **`TORCH_CUDA_ARCH_LIST` is ignored.** Both upstream `setup.py` files
  hard-code their `-gencode` list (sm_75, 80, 87, 90, 100, 103, 110, 120,
  121), and torch's extension builder drops `TORCH_CUDA_ARCH_LIST` when
  arch flags are already present. You always compile all nine
  architectures; that is why the build takes as long as it does.
- **Set `MAMBA_FORCE_BUILD=TRUE` and `CAUSAL_CONV1D_FORCE_BUILD=TRUE`.**
  Without them, each `setup.py` first tries to download a prebuilt binary
  from the upstream project's release page, built for some other torch and
  CUDA combination.
- **`--no-build-isolation` is required**, and torch must already be
  installed (step 1). Under build isolation pip would compile against a
  freshly downloaded torch in a temporary environment, not the pinned one.
- **Linux only.** The kernels do not build natively on Windows or macOS.
  WSL2 counts as Linux.
- **Memory:** `MAX_JOBS` parallel nvcc jobs, each with `NVCC_THREADS`
  threads. 4 x 2 built without trouble on a 125 GB host; lower `MAX_JOBS`
  on a small machine if the compiler is killed.
- **Hash-checking mode** applies to everything in both files: a mirror must
  serve byte-identical files. The PyTorch index serves its own build of
  `triton` under the same file name as PyPI; both hashes are listed. If the
  pins themselves are unavailable, see "If the pins are unavailable" below.
- After step 2, `import selective_scan_cuda` only works after
  `import torch` (the extension links against torch's libraries); the
  project's own imports already do this.

### Docker (the same build, in an image)

`Dockerfile` performs exactly the three steps above in
`nvidia/cuda:13.0.2-devel-ubuntu24.04` with deadsnakes Python 3.11, and
fails the build unless the backend is `mamba_ssm`:

```bash
docker build -t large-scene-localizer:tier1 .
docker run --rm -it --gpus all --shm-size=8g \
    -v /path/to/datasets:/data -v /path/to/runs:/runs large-scene-localizer:tier1
```

Running needs the NVIDIA Container Toolkit on the host (building does not).
Inside the image the three roots are preset: `LSL_DATA_ROOT=/data`,
`LSL_RUNS_ROOT=/runs`, `LSL_CKPT_ROOT` = the image's `checkpoints/`.

### Conda

`environment-tier1.yml` creates Python 3.11 with step 1; then run steps 2
and 3 inside it (conda's pip pass cannot do the `--no-build-isolation`
build in the same step):

```bash
conda env create -f environment-tier1.yml && conda activate lsl-tier1
MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 NVCC_THREADS=2 \
  python -m pip install --no-build-isolation -r requirements/cuda-linux-x86_64.txt
python -m pip install -e . --no-deps
```

## If the pins are unavailable

A mirror (or an Anaconda distribution) that lags behind may not have every
pinned version. Then:

1. Install the nearest available versions instead: the same torch minor
   release if possible (2.12.x), otherwise the nearest one, with the
   matching CUDA build for Tier 1; for every other package, the nearest
   version your mirror has. Drop the `--hash` lines (they only match the
   pinned files): for example `pip install torch==2.12.* numpy scipy ...`
   from the names in the pin files, or edit a copy of the file. For Tier 1,
   keep `mamba-ssm==2.3.2.post1` and `causal-conv1d==1.6.2.post1` from
   source if your mirror has those sdists; they must be compiled against the
   torch you installed.
2. If you have the weight files, check them: runbook step 3 (the manifest
   check). This package ships none.
3. Run the test suite: runbook step 4. Compare the counts with the
   "Verified" row above; any failure is a real difference.
4. Compare results with the tolerance table in `docs/results.md`, section 7:
   on the full 512² dataset, the per-cell rescore, system evaluation and
   per-target recall of runbook step 7. Inside the tolerances, the
   environment is good.

Evidence for how far the pins stretch: the Tier 0 CPU suite gives the same
result under Python 3.11 (pinned), 3.12 and 3.13 with the same pins, in
fresh environments (199 passed, 49 skipped with the weights; 195 / 53 without; Python 3.12.16 and 3.13.18, all pins installed as-is), and an earlier pin, torch 2.15.0+cpu (Python
3.11), gave the same passes and skips as torch 2.12.0+cpu (measured before the five checkpoint-loader tests were added).

## Checkpoints (safetensors)

**No weight files ship with this package**; `checkpoints/` holds only the
three manifests. If the weights arrive separately, they are
`checkpoints/<name>.safetensors`: plain tensor
files (no executable content), with the same tensor names and dtypes as the
PyTorch `state_dict`s they were converted from. Their identity is
`safetensors_sha256` in `checkpoints/<name>.json` (the manifest also records
the original PyTorch file's sha256, which is not shipped). Every loader in
the project accepts either format and prefers `.safetensors`
(`localizer.common.checkpoint_io`); the trainers still write `.pth` files,
which `scripts/checkpoints/to_safetensors.py` converts with a tensor-by-
tensor round-trip check.

## GPU memory: what fits where

| Workload | Needs (peak allocated) | 24 GB card | 128 GB class |
|---|---|---|---|
| Inference, all three models, 512² scenes | under the training peaks below | yes (measured) | yes |
| Train the shipped gate recipe at 512² (stride-2 stem, batch 1) | 8.05 GiB | yes | yes |
| Train classifier (batch 16) / segmenter (batch 8) at 128² | not recorded; 1-epoch smoke runs at these batch sizes fit | yes (measured) | yes |
| Train the L3 gate recipe at 512² (stride-1 stem) | 34.6 GiB | **no** | yes |
| Train the shipped gate architecture at 1024² (stride-2 stem, batch 1) | 31.8 GiB | **no** (the 1024² shape tests run out of memory) | yes |
| Train the gate with a stride-1 stem at 1024² | about 111.6 GiB; ran out of memory on a 128 GB unified-memory GPU | **no** | **no** (needs bf16 / gradient checkpointing) |

So: **512² work with the shipped models fits a 24 GB GPU; 1024² gate
training needs more than 24 GB (32+ GiB allocated), and all 1024² work in
the originating project was done on 128 GB unified-memory GPUs.** Gate
inference memory at 1024² has not been measured. Peaks are PyTorch
`max_memory_allocated`; leave headroom for the CUDA context and allocator
cache. Training with the reference Mamba (Tier 0 on GPU) needs 33.4 GiB at
512² (section "Speed" above): more than a 24 GB card.

<!-- Sources: 8.05 GiB: checkpoints/gate_A_stem2_bs1_ep42.json
peak_vram_bytes 8648129536; 34.6 GiB: originating run artifact
analysis/stage1_gate_ladder_512/preflight_L3.json vram_peak_gib 34.574;
31.8 and ~111.6 GiB: originating repo documents/notes/v0_7_step11_measurement.md;
1024² shape-test OOM ("Tried to allocate 8.00 GiB ... total capacity of
23.42 GiB") and the smoke runs: validation record sections 1 and 7; bf16 +
gradient checkpointing for stride-1 1024²: src/localizer/gate/model.py v0.8
block comment ("vanilla 1024² Arm 2 fits with bf16 + grad-ckpt"); trainer
options --use_bf16 / --gradient_checkpoint. SM mapping: sm_80 SASS also
runs on sm_86/sm_89 (same major version). -->

## GPU architectures and driver (Tier 1)

The from-source build produces compiled kernels (SASS) for
`sm_75, sm_80, sm_87, sm_90, sm_100, sm_103, sm_110, sm_120, sm_121` and no
PTX (the list is fixed in the upstream `setup.py` files): Turing (T4),
Ampere (A100, A10, A30; sm_80 code also runs on sm_86), Ada (L4, L40S; runs
the sm_80 code), Hopper (H100, H200) and Blackwell (B200, B300, RTX PRO /
RTX 50xx). Pre-Turing GPUs (V100, P100) are not supported by this build,
nor by CUDA 13.0 itself; use Tier 0 there (CPU, or the GPU with a torch
build that still supports it).

The driver must support CUDA 13.0 (check NVIDIA's CUDA 13.0 release notes
for the minimum version for your platform). Validated on driver 595.84. The
torch cu130 packages bring their own CUDA runtime libraries (the compiled
kernels link `libcudart.so.13` from them); the system CUDA toolkit is only
needed to compile.

<!-- Sources: cuobjdump --list-elf / --list-ptx on the kernels built from
these sdists (elf: sm_75, 80, 87, 90, 100, 103, 110, 120, 121; ptx: none);
readelf -d: NEEDED libcudart.so.13; originating repo
documents/notes/validation_2026_11_05_export.md section 9b; driver:
checkpoint manifests reproduced_on_x86.platform. -->

## The three locations

Nothing in configs or scripts is an absolute path. Every script resolves
datasets, run output and checkpoints against three roots, from a flag or,
if absent, an environment variable, and fails with a message naming both if
neither is set:

| Variable | Flag | Holds |
|---|---|---|
| `LSL_DATA_ROOT` | `--data-root` | `<root>/<dataset>/{data_all,label_msk}/` |
| `LSL_RUNS_ROOT` | `--runs-root` | `<root>/<run_name>/` training and evaluation output |
| `LSL_CKPT_ROOT` | `--checkpoints-root` | the weight files, named as in `checkpoints/<name>.json` |

## The original training stack (not shipped, and why)

The shipped gate was trained on an aarch64 NVIDIA GB10 node (128 GB unified
memory) with a dated torch nightly, `2.12.0.dev20260322+cu128`,
`mamba_ssm 2.3.1` and `causal_conv1d 1.6.1` compiled locally for aarch64.
The classifier was trained on the same kind of node, inferred to be the same
torch; the segmenter's stack was not recorded. That stack cannot be
reinstalled on x86 (the nightly is gone, the extensions were built for
aarch64), so it is not shipped. Instead Tier 1 pins the nearest stable
release (torch 2.12.0) with x86-64 builds of the next patch versions of the
two extensions, and the reproduction in `docs/results.md` (section 7)
measures how far Tier 1 sits from the original: zero delta on every
binarized metric.

Two of the historical gate runs (internally "B1" and "L2", neither shipped)
ran on a node without `causal_conv1d`, so `mamba_ssm` took its non-fused
convolution path there. The shipped gate used the fused kernel.

<!-- Sources: checkpoints/*.json host, torch, mamba_ssm, causal_conv1d,
inferred_fields; B1/L2 without causal_conv1d: originating repo
documents/notes/recon_2026_11_05_stage1_export_report.md section 3.2 (as
cited in the gate-viability and ladder records). -->

## Known to be wrong or missing

- No Windows-native Tier 1 (the upstream kernels are Linux-only); use WSL2.
- No pre-Turing GPU support in the Tier 1 build.
- Tier 1 needs network access to the package indexes (or a full mirror) and
  a CUDA 13.0 toolkit; nothing prebuilt ships.
- Tier 0 on CPU has not been timed at 512²; expect it to be slow.
- Training with the reference Mamba was timed over 300 batches on a GB10,
  not on an x86 GPU, and its epoch time is extrapolated; it does not fit a
  24 GB GPU at 512².
- Gate training on Tier 1 has been timed for one epoch, not run to
  convergence.
