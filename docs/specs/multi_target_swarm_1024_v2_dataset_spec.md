> **Scrubbed copy.** Source: the originating repository's dataset spec at commit `42fe7b3`. Host names, storage paths and process notes from the originating compute environment were replaced with generic terms ("node 1-3", "shared storage", `$LSL_DATA_ROOT`); file paths were mapped to the current layout (`src/localizer/synth/swarm.py`, `configs/data/`, `scripts/data/generate.py`), but line-number references still point at the pre-reorganization files. The generation recipe, parameters, seeds and rationale are unchanged. Shell commands below are records of how the set was originally produced; for current, copy-pasteable commands see `docs/data.md`, which is authoritative where the two differ.

# Multi-target swarm 1024×1024×64 dataset spec — v2 (GRF-clustered)

**Status:** draft, awaiting user acceptance.

**Scope:** Specification for a 1024×1024×64 synthetic dataset that
**supersedes** the v1 `multi_target_swarm_1024x1024x64` dataset
(spec `multi_target_swarm_1024x1024x64_dataset_spec.md (originating repository; not shipped)`,
commit `db104ee`; handoff `multi_target_swarm_1024_dataset_handoff.md (originating repository; not shipped)`,
commit `7a854ff`). The v1 dataset (3000 scenes, 50% target-absent,
chip-aware uniform-without-replacement chip selection,
1–10 targets/chip uniform) is to be **discarded** on regeneration
under this v2 spec; the v1 spec, code (the `chip_aware` sampler),
handoff narrative, and tests remain in the repo for historical
record. See §1.1 for the supersession rationale.

The v2 dataset preserves v1's structural commitments (1024² spatial
scale, 64-frame temporal extent, 2×2 square targets and swarm members,
1,920,000-circle background swarm, v0 speeds `[0.5, 1.0, 2.0]`, NPY
layout and filename conventions, `MoverDataset` consumer compatibility)
and changes only the **target-presence and spatial-distribution
sampler**:

- **75% target-present, 25% target-absent** (was 50/50).
- **Per-scene non-empty-chip fraction `p ~ Beta(2, 5)` scaled to
  `[1/10, 2/3]`** (was Uniform on `[24/64, 40/64]`). Right-skewed
  toward sparse scenes; rare-but-present dense scenes; the modal
  scene is sparse-to-moderate.
- **Spatial clustering of non-empty chips via a Gaussian random
  field (GRF) over the 8×8 chip grid** (was uniform-without-replacement
  scatter). Direct-covariance method with an anisotropic-capable
  kernel; production setting `[ℓ_y, ℓ_x] = [1.5, 1.5]` chips
  (isotropic). The GRF + threshold mechanism gives contiguous
  "urban" patches whose shape varies scene-to-scene.
- **Per-chip target count derived from the GRF field value**
  (linear-by-field-value mapping into `[1, 10]` plus Gaussian
  noise σ=1, rounded and clipped), so cluster cores carry more
  targets than cluster edges. Per-chip count is no longer uniform
  across non-empty chips.

**Why:** the v1 sampler produced scenes that, at the per-scene chip-
level, looked like a uniform salt-shaker pattern (k non-empty chips
out of 64 scattered without replacement, with `k` tightly concentrated
around 32). This does not reflect the operational regime of overhead
persistent surveillance, where target density varies sharply across a
single scene — a dense urban core surrounded by sparser suburbs and
empty rural / undeveloped tracts. A Stage 1 region-proposal model
trained on uniformly-scattered chips never has to learn to produce
contrast between populated and unpopulated regions of the *same*
scene, which is exactly the operational signal a heatmap-based
region proposer needs to deliver. v2 reshapes the per-scene chip-
density distribution to mirror this structure: most scenes are
sparse with one or two clusters of activity; some are dense; some
are entirely empty. The GRF mechanism makes the spatial clusters
contiguous rather than scattered. The Beta(2,5) shape on per-scene
density makes the sparse-scene tail dominant while preserving a
small dense tail. The 25% target-absent fraction retains the
false-positive control signal that the v1 50/50 parity provided.

**Related artifacts (same as v1; reproduced here for the v2 spec's
self-containment):**

- v0.5/v0.6 segmenter trainer (eventual consumer):
  `src/localizer/training/stage1_seg_train.py`.
- v0.5/v0.6 segmenter dataloader (eventual consumer, format-defining):
  `src/localizer/common/data.py` (`MoverDataset`,
  `make_dataloaders`). **The v2 dataset's per-file shapes and naming
  remain byte-compatible with this loader**; the consumer's API
  contract is unchanged from v1.
- Generator (the codebase to extend further):
  `src/localizer/synth/swarm.py`. v1's `chip_aware`
  sampler path remains in place; v2 adds a third
  `target_sampler="grf_cluster"` branch (additive, backward-compatible
  for both `single` and `chip_aware`).
- Generator CLI: `scripts/data/generate.py`.
  v1 already threaded sampler-specific YAML keys; v2 adds the new
  GRF-specific keys (§4.1 Change 8).
- v1 spec (the immediate precedent for style):
  `multi_target_swarm_1024x1024x64_dataset_spec.md (originating repository; not shipped)`.
  v2 inherits §3.3 (per-target velocity sampling), §3.4 (velocity
  scaling), §3.5 (background swarm), §3.6 (target-absent scenes),
  most of §4.1 (generator design principle), §7 (operations), and
  §8 (out-of-scope) substantively unchanged from v1.

---

## 1. Dataset identity

- **Name:** `multi_target_swarm_1024x1024x64` (unchanged; the v2
  dataset reuses the same name and canonical shared copy path — see §1.1
  on the supersession decision).
- **canonical shared copy path:**
  `$LSL_DATA_ROOT/multi_target_swarm_1024x1024x64/`.
  The v1 dataset is overwritten at this path during the v2
  regeneration (per §5 step 1 and §7.3).
- **Per-node local path** (post-replication):
  `<node-local data>/multi_target_swarm_1024x1024x64/`.
  Likewise overwritten per-node during the v2 regeneration.
- **Layout** (matches v0.5/v0.6 / v1 swarm dataset conventions
  exactly for the `.npy` files; sibling `_gif/` subdirectories for
  human inspection only):

  ```
  multi_target_swarm_1024x1024x64/
    data_all/
      vid0_1.npy        # idx % 4 ∈ {0,1,2} → targets present (label=1)
      vid1_1.npy
      vid2_1.npy
      vid3_0.npy        # idx % 4 == 3       → targets absent  (label=0)
      vid4_1.npy
      ...
      vid2999_0.npy
    label_msk/
      vid0_1_mask.npy
      vid1_1_mask.npy
      ...
      vid2999_0_mask.npy
    data_all_gif/                  # human-inspection GIFs (NOT used by training)
    label_msk_gif/                 # human-inspection mask GIFs (NOT used by training)
    dataset_metadata.json          # generator config + partitioning + provenance
  ```

  **Critical separation:** `MoverDataset.make_dataloaders`
  (`src/localizer/common/data.py` line 111) globs `vid*.npy`
  from `data_all/` only — the sibling `data_all_gif/` and
  `label_msk_gif/` directories contain `.gif` files which will never
  be picked up by the `*.npy` glob. Identical to v0.5/v0.6 / v1.

  **Filename format:** `vid{idx}_{label}.npy` with no zero-padding,
  matching the v0/v0.5/v0.6 / v1 generator's exact convention
  (`src/localizer/synth/swarm.py` line 269:
  `f"vid{idx}_{label}.npy"`).

- **Per-file shapes and dtypes** (unchanged from v1):
  - `data_all/vid{idx}_{label}.npy` — **`(T, H, W) = (64, 1024, 1024)` uint8**.
    Pixels under any object (target or swarm member) at value 255;
    all other pixels at value 0. Identical convention to v0/v0.5/v0.6 / v1
    (`circval=255` per `src/localizer/synth/swarm.py` line 139).
    Loaded by `MoverDataset` as `np.load(...)` then
    `np.expand_dims(..., axis=0).astype(np.float32) / 255.0` →
    `(1, T, H, W)` float32 in `{0.0, 1.0}`. Per-file size:
    64 × 1024 × 1024 × 1 byte = 67,108,864 bytes ≈ 64 MiB raw.
  - `label_msk/vid{idx}_{label}_mask.npy` — **`(T, H, W) = (64, 1024, 1024)` uint8**,
    per-frame anomalous-target presence (NOT swarm presence): pixel
    value 255 if any of the N anomalous targets' 2×2 footprint
    covers pixel `(r, c)` at frame `t`, 0 otherwise. Union mask
    convention (per-frame binary OR across all targets), identical
    to v1. Per-file size: same as the data file.

- **Scene index in filename:** bare integer (1–4 digits depending on
  value), no zero-padding. Matches all prior conventions.

- **Label encoding in filename (v2-specific parity rule):** `1` for
  targets-present, `0` for targets-absent. **The v2 parity rule is
  `has_mover = 0 if (idx % 4 == 3) else 1`**, giving 25% absent and
  75% present. This is a semantic change from v1's `idx % 2 == 0`
  parity (which gave 50/50). Of the 3000 scenes, exactly 750 are
  target-absent (indices 3, 7, 11, …, 2999) and 2250 are target-present
  (indices 0, 1, 2, 4, 5, 6, 8, …, 2998).

- **Total scenes:** 3000 → 2250 target-present + 750 target-absent
  (75/25 by construction).

- **Train/val split:** 90/10. The split is performed at dataloader
  time by `MoverDataset.make_dataloaders` using sklearn's
  `train_test_split` with `val_split=0.1` and a configurable
  `val_split_seed` (default 0). Split is random across the
  3000-scene index range — NOT contiguous, NOT stratified on
  present/absent. With the 75/25 parity rule across the full
  dataset, random non-stratified split gives ~75/25 present/absent
  in both train and val in expectation. The 300-scene val set
  contains roughly 225 present and 75 absent (binomial draw; ±2σ
  is roughly ±15 on each count). The dataset spec does not
  prescribe a specific `val_split_seed`; the consuming training
  spec sets it.

**Per-scene metadata.** No per-scene JSON sidecar. Per-scene target
count and target placements are recoverable from the mask via
connected-component labeling on frame 0 (8-connectivity), exactly
as the v1 handoff (commit `7a854ff`) demonstrated. Dataset-level
provenance is captured in a single `dataset_metadata.json` at the
dataset root (per §7.5).

### 1.1 Supersession of v1

The v1 dataset and the v2 dataset share the same name, canonical shared copy
path, and per-node local path. v2 **overwrites** v1 at all three
locations during regeneration (§5 step 1 explicitly deletes the v1
on-disk dataset before v2 generation starts).

Rationale for overwriting in place rather than versioning the on-disk
location:

1. The two datasets serve the same downstream purpose (1024² scale-up
   substrate for the CNN+Mamba hybrid). There is no anticipated need
   to consume both simultaneously.
2. The training-spec consumer (`MoverDataset` + `--data_root` override)
   identifies the dataset by directory path, not by content. Keeping
   v1 around at a separate path would invite accidental cross-version
   training runs, with no benefit.
3. Per the user's framing, the v1 dataset's on-disk artifacts have no
   research value once v2 is generated. The v1 specifics relevant to
   the project history are captured in commits `db104ee` (v1 spec),
   `7a854ff` (v1 handoff), `0f8fee1` (v1 generator code), and the
   intermediate tests / launchers, all of which remain in `main`.

What is preserved from v1:

- The v1 spec markdown
  (`multi_target_swarm_1024x1024x64_dataset_spec.md (originating repository; not shipped)`)
  remains in the repo, unedited, as the documentation of the v1
  design. It is flagged in §1 of itself as draft-but-implemented;
  v2 supersedes it via this v2 spec doc, and a cross-reference is
  added to v1's §1 in the same commit that lands v2 (per §5 step 0).
- The v1 handoff narrative
  (`multi_target_swarm_1024_dataset_handoff.md (originating repository; not shipped)`)
  remains in the repo as the record of the v1 dataset's generation.
- The v1 generator code (`src/localizer/synth/swarm.py`
  with `target_sampler="chip_aware"`) remains in place and remains
  reachable by configuration (`chip_aware` is still a valid
  `target_sampler` value post-v2 extension). The v2 extension is
  additive (`grf_cluster` is a third sampler added alongside
  `single` and `chip_aware`); regenerating the v1 dataset from
  post-v2 code is mechanically possible should it ever be needed,
  by pointing at the v1 YAML config.
- The v1 YAML config
  (`configs/data/multi_target_swarm_1024.yaml`)
  remains in the repo. The v2 config is a new file
  (`configs/data/multi_target_swarm_1024_v2.yaml`)
  per §4.2; the v1 YAML is untouched.
- The v1 backward-compat fixture
  (`tests/data/swarm_compat_ref/`) remains in place; v2 also passes
  this regression (per §6.6).

What is discarded:

- The 6000 NPY files (3000 data + 3000 mask) on canonical shared copy and
  on each node's node-local copy.
- The per-node smoke GIFs from the v1 generation (the 20-scene
  node-1 smoke set under
  `<node-local data>/multi_target_swarm_1024_smoke/`
  per v1 handoff §3.1; no bulk-time GIFs were generated).
- The v1 `dataset_metadata.json` at canonical shared copy (overwritten by
  v2's version per §7.5).

This is the same "discard data, keep code and docs" pattern
formalized in the project conventions path conventions (Rule 1: source of truth
lives in git; Rule 2: run artifacts live on shared storage under
`data/`/`runs/`/`checkpoints/`/`analysis/`/`archive/`). v1's on-disk
data falls under Rule 2 and is disposable; v1's spec, code, tests,
and handoff fall under Rule 1 and are preserved in git.

---

## 2. Storage and node budget

**Per-scene size:** ~64 MiB data + ~64 MiB mask = ~128 MiB on disk
per scene (raw; identical to v1, since the per-scene tensor shape
is unchanged and storage cost is dominated by the volume tensor
and background swarm rather than the targets).

**Full 3000-scene dataset size (raw NPYs only):**
3000 × 128 MiB = **~375 GiB** (same as v1).

**per-node slice (1000 scenes):** ~125 GiB. Comfortably fits each
node's local NVMe (~1 TiB usable post-v1-cleanup; see §5 step 1
for the v1-deletion step) without threatening capacity.

**canonical shared copy:** ~375 GiB. Per the project conventions, the shared storage
volume is multi-TB and currently holds the 30k swarm (60 GiB),
the 5k slow-swarm (~10 GiB), v0/v0.5/v0.6/v1 run artifacts, and
the v1 multi_target_swarm_1024 dataset (~375 GiB). The v2
generation **deletes the v1 dataset first** (§5 step 1) before
writing v2, so the net incremental shared storage storage during v2 generation
is bounded: at any point in time either v1 or v2 occupies the
canonical path, not both.

**Workstation local NVMe:** the workstation does not host a copy of the v1
dataset (per v1 §7.3, workstation-side replication was deferred to
training-spec time). v2 follows the same policy: workstation-side
replication, if needed, is the training-spec's call.

**GIF directories:** identical recommendation to v1. Generate
GIFs only for a smoke subset and the first ~50 scenes of the
full run for inspection; skip GIFs for the bulk of the dataset
to save generation wall and disk.

**Storage preflight (mandatory before the bulk run, per §5 step 7):**
whoever launches the bulk run must verify each node's
node-local storage has ≥150 GiB free (post-v1-deletion) AND the shared storage
has ≥400 GiB free (post-v1-deletion) before launching. Abort the
launch and surface to the user if preflight fails — there's no
graceful in-flight recovery from a disk-full failure on a
1000-scene worker.

---

## 3. Dataset content: multi-target swarm at 1024² with GRF clustering

### 3.1 Per-scene structure

Each scene is a 1024×1024×64 binary uint8 volume containing:

- **A background swarm of 1,920,000 identical 2×2-square circles**
  drifting coherently in one direction at one speed (the
  "background velocity" — single direction and single speed,
  shared across all 1,920,000 swarm members), unchanged from v1.
- **In target-present scenes (idx % 4 ∈ {0, 1, 2})**: an additional
  set of N anomalous-mover targets (also 2×2 squares, also
  rendered at value 255) placed and moved per §3.2 (GRF-clustered
  placement) and §3.3 (velocity sampling, inherited from v1). N
  varies per scene; typically 20–120 by the §3.2 sampler (the
  per-scene expected count drops from v1's ~176 to v2's ~68; see
  §3.2.2 for the realized-N analysis).

Targets are visually indistinguishable from swarm members on any
single frame — the only discriminative signal is **velocity
anomaly**. Each target either moves in a direction different from
the swarm, or shares the swarm's direction but moves at a
different speed (per the existing `samedir=true` rule, inherited
unchanged from v1).

This is the same per-target detection task as v1; the structural
change is the *spatial distribution* of the N targets across the
scene and the *per-scene variability* in target density.

### 3.2 Per-scene target count and GRF-clustered placement

The dataset is designed so that the per-scene chip-level structure,
across a tiling of the 1024² scene into 64 non-overlapping 128²
chips, **varies sharply scene-to-scene**:

- **Per-scene non-empty-chip fraction `p` is drawn from a right-
  skewed distribution on `[1/10, 2/3]`.** Most scenes are sparse
  (p typically 15–30%); a long-tail of dense scenes exists
  (p approaching 2/3); the absolute floor is 1/10 (6 non-empty
  chips out of 64).
- **The non-empty chips form contiguous spatial clusters**
  ("urban patches") via a Gaussian random field over the 8×8 chip
  grid, rather than the v1 uniform-without-replacement scatter.
- **Per-chip target count varies with the GRF field value**:
  chips at cluster cores carry more targets (up to 10); chips at
  cluster edges carry fewer (down to 1). The range `[1, 10]` is
  preserved from v1 but the *distribution* within that range is
  no longer uniform.

**Sampler (`grf_cluster`, four-stage).** This is the algorithm the
spec mandates. Run once per target-present scene. RNG draw order
is specified explicitly in §3.2.1 to preserve cross-node
reproducibility (the same property v1 had, extended to the new
draws).

#### Stage A — draw per-scene non-empty-chip fraction `p`.

Draw `u ~ Beta(α=2, β=5)`, then rescale to `[p_lo, p_hi] = [1/10, 2/3]`:

```
p = p_lo + u · (p_hi - p_lo)
```

`Beta(2, 5)` has mode `(α-1)/(α+β-2) = 1/5 = 0.2`, mean
`α/(α+β) ≈ 0.286`, and right tail extending to 1. Rescaled to
`[1/10, 2/3]`, this gives a mode at scaled fraction ≈ 0.213
(k ≈ 14 non-empty chips out of 64), mean ≈ 0.262 (k ≈ 17), and
the upper tail reaches 0.667 (k = 43). The lower bound 1/10
guarantees every target-present scene has at least ~6 non-empty
chips.

Compute the realized integer non-empty chip count:

```
k = max(1, round(p · 64))
```

(banker's rounding / round-half-to-even is fine; `max(1, ...)` is
a safety lower bound — `round(0.1 · 64) = 6` already, so the max
clause is inert in practice given `p_lo = 1/10`.)

#### Stage B — draw the Gaussian random field `f` over the 8×8 chip grid.

The GRF mechanism is direct-covariance with an anisotropic-capable
squared-exponential kernel. Let `n_chips_per_side = H // chip_size`
(8 for the production config); the chip grid has `T_chips = 64`
nodes indexed by `(i, j)` for `i, j ∈ {0, ..., 7}` (row, col).

The covariance kernel between chip `(i, j)` and chip `(i', j')` is

```
k((i,j), (i',j')) = exp( -0.5 · ( ((i-i') / ℓ_y)² + ((j-j') / ℓ_x)² ) )
```

with production setting `[ℓ_y, ℓ_x] = [1.5, 1.5]` (isotropic).
Anisotropic settings (e.g., `[ℓ_y, ℓ_x] = [1.0, 4.0]` for
elongated east-west urban corridors) are admissible without
code changes — only the YAML value changes. For numerical
stability the kernel matrix has a small diagonal jitter
`ε_jitter = 1e-6` added before Cholesky decomposition.

The 64×64 covariance matrix `Σ = K + ε_jitter · I` and its
Cholesky factor `L` such that `L Lᵀ = Σ` are precomputed **once
at generator initialization** (constructor time, post-config-load,
pre-scene-loop). The cost is dominated by the Cholesky
(64×64; trivial), and the `L` matrix is reused across all 2250
target-present scenes generated by this `DataGenerator` instance.

Per-scene, draw `z ~ N(0, I_64)` (64 independent standard normals)
and form

```
f = L @ z       # shape (64,), one field value per chip
```

The realized `f` is a sample from `N(0, Σ)`, a smooth random field
over the 8×8 grid with the chosen correlation length. The mean of
`f` is exactly zero by construction; the marginal variance at any
chip is exactly 1 (since the diagonal of `K` is 1).

#### Stage C — threshold the field to obtain the chip mask, sized exactly to `k`.

Sort the 64 field values in descending order; pick the top `k` as
the non-empty chips:

```
chip_order = np.argsort(f)[::-1]       # indices of chips, highest-f first
non_empty_chip_indices = chip_order[:k]
```

This guarantees the realized non-empty-chip count is exactly `k`,
matching the Stage-A draw. The mechanism decouples the GRF
correlation length from the fraction: the *field* controls *where*
the populated chips cluster (via the kernel's correlation length);
the *threshold* controls *how many*.

**Why threshold-to-exact-count rather than threshold-at-a-fixed-quantile-
of-the-standard-normal:** the fixed-quantile alternative would have
realized `k` vary scene-to-scene around its expectation under the
field (and the expectation depends weakly on the correlation length).
Threshold-to-exact-count eliminates that incidental variation —
the per-scene `k` matches the per-scene `p` draw exactly, so the
two RNG mechanisms (Beta for fraction, GRF for spatial structure)
remain conceptually orthogonal.

#### Stage D — assign per-chip target counts and place targets within each chip.

For each non-empty chip, compute a field-value-driven mean count
`μ_c`, then draw a noisy realized count `m_c`.

Let `f_min = min(f[non_empty_chip_indices])` and `f_max =
max(f[non_empty_chip_indices])` (the smallest and largest field
values *among the k non-empty chips* — these are the field values
at the threshold and at the cluster core, respectively).

For each non-empty chip with field value `f_c`:

```
# Linear-by-field-value mapping into [1, 10]:
if f_max > f_min:
    μ_c = 1 + 9 · (f_c - f_min) / (f_max - f_min)   # in [1, 10]
else:
    μ_c = 5.5                                       # degenerate: all f equal
ε_c ~ Normal(0, σ_count=1)
m_c = clip(round(μ_c + ε_c), 1, 10)                 # integer in [1, 10]
```

Per-chip target placement within the chip is identical to v1 §3.2
Stage C: uniform within the chip's pixel extent with a `square_edge`-
pixel inset from the high-edge boundaries so the 2×2 target
footprint stays inside the chip at placement time. The chip-to-
pixel mapping is

- `chip_row = chip_idx // n_chips_per_side` (chip row index in 0..7)
- `chip_col = chip_idx % n_chips_per_side` (chip column index in 0..7)
- `cx_lo = chip_col · chip_size`, `cx_hi = (chip_col + 1) · chip_size - square_edge`
- `cy_lo = chip_row · chip_size`, `cy_hi = (chip_row + 1) · chip_size - square_edge`
- `cx = rng.integers(cx_lo, cx_hi)`, `cy = rng.integers(cy_lo, cy_hi)`

(`cx` is the column coordinate, `cy` is the row coordinate, matching
the generator's existing `(cx, cy)` convention. The 2×2 square
rendered by `_make_square` at `(cx, cy)` occupies rows `cy..cy+square_edge`
and cols `cx..cx+square_edge`.) Targets may exit their starting
chips in later frames as they move; this is intended (the v1 design
note carries over unchanged).

#### 3.2.1 RNG draw order (cross-node reproducibility contract)

Per-scene draws happen in **exactly** this order. This contract is
what enables bit-reproducibility across the three training nodes for the
same scene index (a scene index always takes the same has_mover
branch, since `has_mover` is determined deterministically from
`idx`).

Per target-present scene (`has_mover == 1`):

1. **bg_dir** ← `rng.choice(self.dirs)` (8 options)
2. **bg_spd** ← `rng.choice(self.speeds)` (3 options)
3. **u** ← `rng.beta(2, 5)` for the per-scene Beta-distributed
   fraction draw. `p = 1/10 + u · (2/3 - 1/10)`. `k = max(1, round(p · 64))`.
4. **z** ← `rng.standard_normal(size=64)` for the GRF draw.
   Compute `f = L @ z`. Determine `non_empty_chip_indices` by
   `argsort(f)[::-1][:k]`.
5. For each of the `k` non-empty chips, in the order they appear
   in `non_empty_chip_indices` (i.e., descending field value):
   a. **ε_c** ← `rng.normal(0, σ_count=1)`. Compute `μ_c` from the
      linear-by-field-value mapping, then `m_c = clip(round(μ_c + ε_c), 1, 10)`.
   b. For each of the `m_c` targets in this chip, in placement order:
      - **cx** ← `rng.integers(cx_lo, cx_hi)`
      - **cy** ← `rng.integers(cy_lo, cy_hi)`
6. For each of the realized `N = Σ m_c` targets, in the order they
   were placed in step 5b, draw the per-target velocity:
   a. **tgt_dir** ← per the v1 §3.3 anomalous-mover constraint
      (`rng.choice(self.dirs)`, then conditional speed draw)
   b. **tgt_spd** ← per the v1 §3.3 constraint
7. Background swarm starts: `bg_centers_arr[:, 0] = rng.integers(-W, 2W, size=n_circles)`
   followed by `bg_centers_arr[:, 1] = rng.integers(-H, 2H, size=n_circles)`,
   identical to v1.

Per target-absent scene (`has_mover == 0`, `idx % 4 == 3`):

1. **bg_dir** ← `rng.choice(self.dirs)`
2. **bg_spd** ← `rng.choice(self.speeds)`
3. *Skip* Beta and GRF draws (no targets to place).
4. *Skip* per-target velocity draws (no targets).
5. **bg_centers_arr** sampling identical to step 7 above.

This matches the v1 chip_aware pattern: target-absent scenes draw
fewer per-scene RNG draws than target-present scenes. The two
branches produce **different** RNG-stream-state after the scene
completes — meaning later scenes in the same batch see a different
state depending on whether earlier scenes were present or absent.

This does **not** break cross-node reproducibility, because (a)
the RNG is seeded once per batch (per `worker(...)` in
`src/localizer/synth/swarm.py`: `rng = np.random.default_rng(seed)`
with `seed = start_idx` by default) and threaded through all scenes
in that batch in order, (b) `has_mover` is determined deterministically
from `idx`, so on every machine the same `(start_idx, batch_size)`
batch will see the same sequence of present/absent scenes and the
same RNG state at each scene. Same `(start_idx, batch_size)` →
same outputs, byte-identical.

**Cross-machine reproducibility requirement.** Same scene index
produces same output **only when the batching configuration matches**.
Running the same `start_idx` with different `batch_size` (so the
scene starts from a different RNG state) produces different bytes
even though `idx` is the same. This is identical to v1's behavior;
the v2 spec does not change this property. Whoever implements the
chip-aware backward-compat fixture (§4.3 Step 0) and the §6.6(b)
regression test must use the same `(start_idx, batch_size,
num_workers)` configuration in both — the §4.3 fixture-prep
command's `--batch_size 2 --num_workers 1` overrides exist to
pin this.

The number of RNG draws per scene differs by branch (target-present:
2 + 1 + 64 + variable + variable + 2·n_circles; target-absent:
2 + 2·n_circles); whoever implements v2 should follow the branch
structure literally and not attempt to "balance" draws between
branches.

#### 3.2.2 Per-scene realized target count `N` — expected statistics

With `p ~ Beta(2,5)` scaled to `[1/10, 2/3]` and per-chip count
distributed by the GRF-driven linear-by-field-value mapping with
small Gaussian noise, the expected per-scene target count over
target-present scenes is approximately:

```
E[k]              ≈ E[p] · 64 = 0.286 · 64 ≈ 17 non-empty chips per present scene
E[m_c | chip]     ≈ 4 (the mean of a near-symmetric distribution centered roughly
                       at the median field-value position in [1,10] = 5.5,
                       but skewed lower because most non-empty chips are
                       near the threshold — i.e., at the cluster edges —
                       so f_c is closer to f_min than f_max)
E[N | present]    ≈ E[k] · E[m_c] ≈ 17 · 4 ≈ 68 targets per present scene
```

This is a **~2.6× drop from v1's E[N | present] ≈ 176**. The σ on
`N` derives from the law of total variance for a sum of `k` per-chip
counts: `Var[N] = E[k]·Var[m_c] + Var[k]·E[m_c]²`. With σ[k] ≈ 5.5,
σ[m_c | chip] ≈ 1.5, E[m_c] ≈ 4, this gives σ[N | present] ≈ 24,
i.e., a typical N range of roughly `[20, 120]` (±2σ) on target-present
scenes, with a long tail to ~170 at the dense extreme and a floor
near 6–7 at the sparse extreme.

These figures were validated by a 1000-trial Monte Carlo of the
sampler under iid-Gaussian field assumption (correlation length
does not affect per-scene N expectations, only spatial layout):
empirical E[k]=16.9, σ[k]=5.6, E[N]=70.4, σ[N]=24.5, E[m_c]=4.18.
The deterministic-implementation results may differ slightly under
the true correlated-GRF distribution; §6.1 sets the test tolerance
to absorb this.

Empirical verification of these expectations is part of the test
plan (§6.1).

**Downstream implication.** Per-scene background-to-target voxel
ratio rises by ~2.6× relative to v1 (same background, ~2.6× fewer
target voxels). This shifts class balance in any segmentation loss.
The training-spec author consuming this dataset should re-tune
`pos_weight` (or equivalent class-balance hyperparameter) relative
to v1's tuning. This is flagged in §9 Risks.

### 3.3 Per-target velocity sampling

**Inherited unchanged from v1 §3.3.** Per scene, the background
velocity `(bg_dir, bg_spd)` is drawn once from the 8 directions ×
3 speeds pool. Then, for each of the N targets independently, draw
`(tgt_dir_i, tgt_spd_i)` subject to the anomalous-mover constraint:

- Direction pool: `{left, right, up, down, upleft, upright,
  downleft, downright}`.
- Speed pool: `{0.5, 1.0, 2.0}`.
- Constraint per target: `(tgt_dir_i, tgt_spd_i) ≠ (bg_dir, bg_spd)`.
  The `samedir=true` convention is preserved: if a target's
  direction equals `bg_dir`, its speed is forced to differ; when
  direction differs from `bg_dir`, speed is drawn freely from the
  speed pool.

This produces N independent anomalous movers against a single
coherent background swarm. The implementation is identical to v1
`_sample_multi_target_velocities` (§4.1 Change 4 in the v1 spec);
the new `grf_cluster` sampler reuses this helper. No new
implementation needed for §3.3.

### 3.4 Velocity scaling

**Inherited unchanged from v1 §3.4.** Target speeds remain
`[0.5, 1.0, 2.0]` per-frame pixels — v0 speeds restored at the
1024² scale. The chip-locality property (most trajectories stay
within their starting chip; some cross at most one boundary) is
preserved.

### 3.5 Background swarm

**Inherited unchanged from v1 §3.5.** 1,920,000 circles per scene,
2×2 squares at value 255, single shared `(bg_dir, bg_speed)` for
the entire swarm, starting positions uniform on
`[-W, 2W] × [-H, 2H] = [-1024, 2048] × [-1024, 2048]`, no
per-swarm-member anomaly.

### 3.6 Target-absent (label=0) scenes

Same as v1 §3.6 in physics (full swarm, no targets, mask is all
zeros) but with the new parity rule per §1: target-absent iff
`idx % 4 == 3`. Per §3.2.1, target-absent scenes skip the Beta
and GRF draws entirely; the only per-scene RNG draws are bg_dir,
bg_spd, and bg_centers (matching the v1 chip_aware target-absent
path).

---

## 4. Generator extension

### 4.1 Code changes to `src/localizer/synth/swarm.py`

The v2 extension is **additive to the v1 extension**: when
`target_sampler="single"` or `target_sampler="chip_aware"`, every
existing code path runs unchanged (single-target preserved
bit-identical to pre-v1 reference; chip-aware preserved
bit-identical to v1 reference). The new `target_sampler="grf_cluster"`
branch is dispatched from `_generate_video` based on
`self.target_sampler` alongside the existing dispatch.

**Design principle (carries forward from v1).** Do not refactor
the existing `single` or `chip_aware` paths to "share code" with
the new `grf_cluster` path. The three sampler bodies are parallel
branches dispatched from `_generate_video`; each consumes a
distinct RNG draw sequence and must remain independently
reproducible.

**Change 1: Constructor adds new params (defaults preserve v1
behavior — `target_sampler="single"` still gives a single-target
generator).**

```python
def __init__(
    self,
    n_frames: int,
    n_circles: int,
    image_size: Tuple[int, int],
    radius: int,
    speeds: list,
    dat_lab_paths: list,
    samedir: bool,
    shape: str = "disk",
    square_edge: int = 2,
    # v1 params (carried forward):
    target_sampler: str = "single",
    chip_size: int = 128,
    chip_nonempty_range: Tuple[int, int] = (24, 40),
    chip_target_count_range: Tuple[int, int] = (1, 10),
    # v2 params (new):
    grf_fraction_range: Tuple[float, float] = (0.1, 2/3),
    grf_fraction_beta: Tuple[float, float] = (2.0, 5.0),
    grf_corr_length: Tuple[float, float] = (1.5, 1.5),   # [ℓ_y, ℓ_x]
    grf_count_range: Tuple[int, int] = (1, 10),
    grf_count_noise_sigma: float = 1.0,
    grf_jitter: float = 1e-6,
    grf_absent_modulo: int = 4,                           # idx % 4 == 3 → absent
) -> None:
    # (v0/v0.5/v0.6 / v1 constructor assertions preceding this point —
    #  positivity checks on n_frames/n_circles, etc — are unchanged.)
    assert target_sampler in ("single", "chip_aware", "grf_cluster"), \
        f"target_sampler must be 'single', 'chip_aware', or 'grf_cluster'; got {target_sampler!r}"

    # v1 chip_aware validation block — inline verbatim from v1 §4.1 Change 1.
    if target_sampler == "chip_aware":
        assert chip_size > 0
        assert image_size[0] % chip_size == 0 and image_size[1] % chip_size == 0, \
            f"image_size {image_size} must be divisible by chip_size {chip_size}"
        assert 0 < chip_nonempty_range[0] <= chip_nonempty_range[1]
        n_chips = (image_size[0] // chip_size) * (image_size[1] // chip_size)
        assert chip_nonempty_range[1] <= n_chips, \
            f"chip_nonempty_range upper {chip_nonempty_range[1]} exceeds total chips {n_chips}"
        assert 1 <= chip_target_count_range[0] <= chip_target_count_range[1]

    # v2 grf_cluster validation block — new.
    if target_sampler == "grf_cluster":
        assert chip_size > 0
        assert image_size[0] == image_size[1], \
            f"grf_cluster requires square scenes; got image_size {image_size}"
        assert image_size[0] % chip_size == 0, \
            f"image_size {image_size} must be divisible by chip_size {chip_size}"
        assert 0.0 < grf_fraction_range[0] < grf_fraction_range[1] <= 1.0
        assert grf_fraction_beta[0] > 0 and grf_fraction_beta[1] > 0
        assert grf_corr_length[0] > 0 and grf_corr_length[1] > 0
        assert 1 <= grf_count_range[0] <= grf_count_range[1]
        assert grf_count_noise_sigma >= 0.0
        assert grf_jitter > 0
        assert grf_absent_modulo >= 2

    # (v0/v0.5/v0.6 / v1 field assignments — self.n_frames, self.n_circles,
    #  self.H, self.W, self.speeds, self.dirs, self.samedir, etc — are
    #  unchanged. Add the v1 and v2 assignments below.)
    self.target_sampler = target_sampler
    # v1:
    self.chip_size = chip_size
    self.chip_nonempty_range = chip_nonempty_range
    self.chip_target_count_range = chip_target_count_range
    # v2:
    self.grf_fraction_range = grf_fraction_range
    self.grf_fraction_beta = grf_fraction_beta
    self.grf_corr_length = grf_corr_length
    self.grf_count_range = grf_count_range
    self.grf_count_noise_sigma = grf_count_noise_sigma
    self.grf_jitter = grf_jitter
    self.grf_absent_modulo = grf_absent_modulo

    # Precompute Cholesky factor L of the GRF covariance Σ
    # only when target_sampler == "grf_cluster"; cheap (64×64) but
    # avoids the work for other sampler modes.
    if target_sampler == "grf_cluster":
        self._grf_chol_L = self._build_grf_cholesky()
    else:
        self._grf_chol_L = None
```

**Change 2: New method `_build_grf_cholesky`** — builds the 64×64
covariance matrix from `grf_corr_length` and returns its Cholesky
factor.

```python
def _build_grf_cholesky(self) -> np.ndarray:
    """Build covariance matrix Σ from the squared-exponential kernel
    with anisotropic length scales [ℓ_y, ℓ_x], add diagonal jitter,
    and return L such that L Lᵀ = Σ.

    Returns:
        L : (T_chips, T_chips) float64 lower-triangular Cholesky factor.
            T_chips = (H // chip_size) ** 2 = 64 for production.
    """
    n_chips_per_side = self.H // self.chip_size  # H == W asserted in Change 1
    T_chips = n_chips_per_side * n_chips_per_side
    ly, lx = self.grf_corr_length

    # Build coordinate arrays for each chip in (row, col) order.
    chip_rows = np.arange(T_chips) // n_chips_per_side  # 0..n-1
    chip_cols = np.arange(T_chips) % n_chips_per_side   # 0..n-1

    # Pairwise differences.
    di = chip_rows[:, None] - chip_rows[None, :]  # (T_chips, T_chips)
    dj = chip_cols[:, None] - chip_cols[None, :]

    # Squared-exponential kernel with anisotropy.
    K = np.exp(-0.5 * ((di / ly) ** 2 + (dj / lx) ** 2))

    # Diagonal jitter for numerical stability.
    Sigma = K + self.grf_jitter * np.eye(T_chips)

    # Cholesky decomposition.
    L = np.linalg.cholesky(Sigma)  # raises if Σ is not PSD (would indicate jitter too small)
    return L
```

**Change 3: `_generate_video` dispatches on `self.target_sampler`.**
Add the new `grf_cluster` branch alongside the v1 `chip_aware`
branch. The single-target body (the post-fallthrough remainder)
is preserved **verbatim** from the v1 generator — no code changes,
no RNG reordering. The implementer must not alter any line of the single-target
body below.

```python
def _generate_video(
    self, idx: int, rng: np.random.Generator,
) -> Tuple[Path, Path]:
    """Generate one scene + mask NPY pair. Returns the two file paths."""
    if self.target_sampler == "grf_cluster":
        return self._generate_video_grf_cluster(idx, rng)
    if self.target_sampler == "chip_aware":
        return self._generate_video_chip_aware(idx, rng)

    # ----- existing single-target path, UNCHANGED from v0/v0.5/v0.6 -----
    assert idx >= 0
    self.has_mover = 1 if idx % 2 == 0 else 0
    label = self.has_mover

    bg_vel, tgt_vel, tgt_dir = self._set_vels(rng)
    a_centers_arr, m_centers_arr = self._create_centers(tgt_dir, rng)
    a_video, m_video = self._assemble_frames(
        bg_vel, tgt_vel, a_centers_arr, m_centers_arr, label,
    )

    a_filename = self.data_all_pth / f"vid{idx}_{label}.npy"
    np.save(a_filename, a_video)
    m_filename = self.label_msk_pth / f"vid{idx}_{label}_mask.npy"
    np.save(m_filename, m_video)
    logger.debug("Saved scene %d: %s, %s", idx, a_filename, m_filename)
    return a_filename, m_filename
```

**Change 4: New method `_sample_grf_cluster_targets`** — implements
the four-stage sampler from §3.2. Pure-position sampling, no
rendering. Called **only** for target-present scenes (has_mover=1);
target-absent scenes skip this method entirely per §3.2.1 step 3.
Draws happen in the exact order specified in §3.2.1.

```python
def _sample_grf_cluster_targets(
    self, rng: np.random.Generator,
) -> np.ndarray:
    """Sample target starting positions under the §3.2 GRF-clustered scheme.

    Caller responsibility: invoke only for target-present scenes.
    Target-absent scenes do not call this method (matches the v1
    chip_aware pattern, where `_sample_chip_aware_targets` is skipped
    when `has_mover == 0`).

    Args:
        rng: numpy Generator.

    Returns:
        target_positions: (N, 2) float64 array of target starts
            (cx=col, cy=row). N is determined by the sampler (Stages
            A-D of §3.2), not fixed. Empty (0, 2) only in the
            pathological case where every clip+round somehow produces
            m_c = 0 — which cannot happen, since the clip bound is
            1; so N ≥ k ≥ 1 in practice.
    """
    cs = self.chip_size
    n_chips_per_side = self.H // cs
    T_chips = n_chips_per_side * n_chips_per_side

    # Stage A — Beta fraction draw. Note: `beta_param` (not `beta`)
    # to avoid shadowing the scipy.stats.beta name should anyone
    # later import it into this module.
    alpha, beta_param = self.grf_fraction_beta
    u = float(rng.beta(alpha, beta_param))
    p_lo, p_hi = self.grf_fraction_range
    p = p_lo + u * (p_hi - p_lo)
    k = max(1, int(round(p * T_chips)))

    # Stage B — GRF field draw.
    z = rng.standard_normal(size=T_chips)
    f = self._grf_chol_L @ z

    # Stage C — threshold to exact k. non_empty_chip_indices is in
    # descending-field-value order, so subsequent iteration is also
    # in descending-field-value order (matches §3.2.1 step 5).
    chip_order = np.argsort(f)[::-1]
    non_empty_chip_indices = chip_order[:k]

    # Stage D — per-chip count and per-target placement.
    f_at_non_empty = f[non_empty_chip_indices]
    f_min = float(f_at_non_empty.min())
    f_max = float(f_at_non_empty.max())
    range_f = f_max - f_min
    lo_count, hi_count = self.grf_count_range
    count_span = hi_count - lo_count  # = 9 for the production [1, 10]

    target_positions: List[Tuple[int, int]] = []
    for chip_idx in non_empty_chip_indices:
        f_c = f[chip_idx]
        if range_f > 0:
            mu_c = lo_count + count_span * (f_c - f_min) / range_f
        else:
            mu_c = 0.5 * (lo_count + hi_count)  # degenerate case (k=1)

        # Stage D noise draw (one per non-empty chip, in field-value-descending order).
        eps_c = float(rng.normal(0.0, self.grf_count_noise_sigma))
        m_c = int(np.clip(int(round(mu_c + eps_c)), lo_count, hi_count))

        # Per-target placement within the chip.
        chip_row = int(chip_idx) // n_chips_per_side
        chip_col = int(chip_idx) % n_chips_per_side
        cx_lo = chip_col * cs
        cx_hi = (chip_col + 1) * cs - self.square_edge
        cy_lo = chip_row * cs
        cy_hi = (chip_row + 1) * cs - self.square_edge
        for _ in range(m_c):
            cx = int(rng.integers(cx_lo, cx_hi))
            cy = int(rng.integers(cy_lo, cy_hi))
            target_positions.append((cx, cy))

    return (np.array(target_positions, dtype=np.float64)
            if target_positions
            else np.empty((0, 2), dtype=np.float64))
```

**Change 5: `_sample_multi_target_velocities` is refactored to
return bg metadata.** v1's signature returns
`(bg_vel, tgt_vels, tgt_dirs)`. v2's `grf_cluster` path needs to
draw the bg velocity *before* the target count `N` is known (per
the §3.2.1 RNG-draw order), then draw the per-target velocities
*after* sampling positions. The cleanest way to support this is
to refactor `_sample_multi_target_velocities` into two methods:

```python
def _sample_bg_velocity(
    self, rng: np.random.Generator,
) -> Tuple[Tuple[float, float], str, float]:
    """Sample one bg velocity. Returns (bg_vel, bg_dir, bg_spd).

    Two RNG draws: bg_dir (choice over self.dirs), bg_spd (choice
    over self.speeds). Order: bg_dir first, bg_spd second.
    """
    bg_dir = str(rng.choice(self.dirs))
    bg_spd = float(rng.choice(self.speeds))
    bg_vel = self._velswitch(bg_dir, bg_spd)
    return bg_vel, bg_dir, bg_spd

def _sample_target_velocities(
    self, n_targets: int, bg_dir: str, bg_spd: float,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """Sample N independent target velocities under the anomalous-mover
    constraint, given the bg already chosen. Returns (tgt_vels, tgt_dirs).

    Per-target draws (in order): tgt_dir, then tgt_spd (with conditional
    speed pool when tgt_dir == bg_dir and self.samedir is True).
    """
    tgt_vels: List[Tuple[float, float]] = []
    tgt_dirs: List[str] = []
    for _ in range(n_targets):
        if self.samedir:
            tgt_dir = str(rng.choice(self.dirs))
            if tgt_dir == bg_dir:
                alt_speeds = [s for s in self.speeds if s != bg_spd]
                tgt_spd = float(rng.choice(alt_speeds))
            else:
                tgt_spd = float(rng.choice(self.speeds))
        else:
            alt_dirs = [d for d in self.dirs if d != bg_dir]
            tgt_dir = str(rng.choice(alt_dirs))
            tgt_spd = float(rng.choice(self.speeds))
        tgt_vels.append(self._velswitch(tgt_dir, tgt_spd))
        tgt_dirs.append(tgt_dir)
    return tgt_vels, tgt_dirs

@staticmethod
def _velswitch(direction: str, speed: float) -> Tuple[float, float]:
    """Convert (direction, speed) → (Vx, Vy). Extracted to a static
    method from the v1 inline helper for reuse across sampler paths."""
    return {
        "left": (-speed, 0.0), "right": (speed, 0.0),
        "up": (0.0, -speed), "down": (0.0, speed),
        "upleft": (-speed, -speed), "upright": (speed, -speed),
        "downleft": (-speed, speed), "downright": (speed, speed),
    }[direction]
```

The v1 `_sample_multi_target_velocities` is replaced by a thin
wrapper that calls both new methods in sequence, preserving its
existing signature so v1's `_generate_video_chip_aware` continues
to work without change:

```python
def _sample_multi_target_velocities(
    self, n_targets: int, rng: np.random.Generator,
) -> Tuple[Tuple[float, float], List[Tuple[float, float]], List[str]]:
    """v1 wrapper: combined bg + N-target velocity sampling.
    Used by _generate_video_chip_aware (v1) and any other path that
    wants the combined-draw convenience.

    The RNG draw order (bg_dir, bg_spd, then per-target tgt_dir/tgt_spd
    pairs) is identical to v1's pre-refactor implementation, so v1
    chip-aware reproducibility is preserved bit-identical.
    """
    bg_vel, bg_dir, bg_spd = self._sample_bg_velocity(rng)
    tgt_vels, tgt_dirs = self._sample_target_velocities(
        n_targets, bg_dir, bg_spd, rng,
    )
    return bg_vel, tgt_vels, tgt_dirs
```

**Critical RNG-equivalence note.** This refactor preserves v1
chip-aware bit-identity *only* because the new helpers draw RNG
in the same order as v1's monolithic `_sample_multi_target_velocities`:
bg_dir → bg_spd → per-target (tgt_dir → tgt_spd). The §6.6(b)
regression test in §4.3 / §6.6 verifies this. If an implementation
of the refactor inadvertently changes the draw order (e.g., draws
bg_spd before bg_dir), v1 chip-aware bit-identity breaks. The implementer must
preserve the v1 order verbatim.

**Change 6: `_assemble_frames_multi` — REUSED FROM v1, NO CHANGES.**
The v1 helper (Change 5 in the v1 spec) handles multi-target frame
assembly with arbitrary `N` targets; the new `grf_cluster` sampler
emits the same `(N, 2)` target-position array shape that v1 emitted,
so the same downstream method applies unchanged.

**Change 7: New method `_generate_video_grf_cluster`** — the
v2 orchestration entry point. The RNG draws happen in the order
specified in §3.2.1 by construction of the code below. The
has_mover=0 branch skips the GRF sampler entirely, matching the
v1 chip_aware pattern.

```python
def _generate_video_grf_cluster(
    self, idx: int, rng: np.random.Generator,
) -> Tuple[Path, Path]:
    """GRF-clustered multi-target scene generation (v2)."""
    assert idx >= 0
    # v2 parity rule: target-absent iff (idx % grf_absent_modulo == grf_absent_modulo - 1).
    # Production grf_absent_modulo = 4 ⇒ target-absent iff idx % 4 == 3 ⇒ 25% absent.
    self.has_mover = 0 if (idx % self.grf_absent_modulo == self.grf_absent_modulo - 1) else 1
    label = self.has_mover

    # RNG draw order per §3.2.1.
    # Step 1-2: bg_dir, bg_spd (always, both branches).
    bg_vel, bg_dir, bg_spd = self._sample_bg_velocity(rng)

    if self.has_mover:
        # Steps 3-5: Beta + GRF + per-chip count noise + per-target placement.
        m_centers_arr = self._sample_grf_cluster_targets(rng)
        N = int(m_centers_arr.shape[0])
        # Step 6: per-target velocity draws (N times).
        tgt_vels, _tgt_dirs = self._sample_target_velocities(
            N, bg_dir, bg_spd, rng,
        )
    else:
        # Target-absent: skip steps 3-6 entirely (matches v1 chip_aware).
        m_centers_arr = np.empty((0, 2), dtype=np.float64)
        tgt_vels: List[Tuple[float, float]] = []

    # Step 7: bg starting positions (always, both branches).
    bg_centers_arr = np.empty((self.n_circles, 2), dtype=np.float64)
    bg_centers_arr[:, 0] = rng.integers(-self.W, 2 * self.W, size=self.n_circles)
    bg_centers_arr[:, 1] = rng.integers(-self.H, 2 * self.H, size=self.n_circles)

    a_video, m_video = self._assemble_frames_multi(
        bg_vel, tgt_vels, bg_centers_arr, m_centers_arr, label,
    )

    a_filename = self.data_all_pth / f"vid{idx}_{label}.npy"
    np.save(a_filename, a_video)
    m_filename = self.label_msk_pth / f"vid{idx}_{label}_mask.npy"
    np.save(m_filename, m_video)
    logger.debug(
        "Saved grf_cluster scene %d (label=%d, N=%d): %s, %s",
        idx, label, int(m_centers_arr.shape[0]), a_filename, m_filename,
    )
    return a_filename, m_filename
```

Note: the v1 `_sample_chip_aware_targets` helper for the v1
`chip_aware` sampler is unchanged. The v1 `_generate_video_chip_aware`
orchestration method is also unchanged — it still calls
`_sample_multi_target_velocities` (now a thin wrapper) and receives
the same `(bg_vel, tgt_vels, tgt_dirs)` triple as before.

**Change 8: `scripts/data/generate.py` threads new
YAML keys.** Extend the YAML-to-kwargs threading from v1 (which
already handles `target_sampler`, `chip_size`, `chip_nonempty_range`,
`chip_target_count_range`) to additionally thread the v2 keys:
`grf_fraction_range`, `grf_fraction_beta`, `grf_corr_length`,
`grf_count_range`, `grf_count_noise_sigma`, `grf_jitter`,
`grf_absent_modulo`. Unrecognized YAML keys must raise an explicit
error (carries forward from v1).

### 4.2 New YAML config: `configs/data/multi_target_swarm_1024_v2.yaml`

```yaml
# multi_target_swarm_1024_v2 — 1024×1024×64 GRF-clustered multi-target dataset.
#
# v2 supersedes the v1 multi_target_swarm_1024 config; see
# docs/specs/multi_target_swarm_1024_v2_dataset_spec.md for full rationale.
#
# Structural changes vs v1:
#   - target_sampler: grf_cluster (was chip_aware)
#   - per-scene non-empty chip fraction: Beta(2,5) scaled to [1/10, 2/3]
#     (was uniform on [24/64, 40/64])
#   - spatial clustering of non-empty chips via 2D GRF (was uniform-without-replacement)
#   - per-chip target count derived from GRF field value (was uniform[1,10])
#   - parity rule: idx % 4 == 3 → absent (was idx % 2 ≠ 0)
#
# Unchanged from v1: image_size, n_frames, n_circles, speeds, samedir,
# 2×2 square geometry, NPY layout, filename convention.

n_frames: 64
image_size: [1024, 1024]
shape: square
square_edge: 2
radius: 2          # unused when shape="square", kept for backward-compat

# Velocity pool — v0 values at 1024² scale, inherited from v1 §3.4.
speeds: [0.5, 1.0, 2.0]
samedir: true

# Background swarm density — 30k per 128² chip × 64 chips, inherited from v1 §3.5.
n_circles: 1920000

# Multi-target sampler — grf_cluster per spec §3.2.
target_sampler: grf_cluster
chip_size: 128

# Per-scene non-empty-chip fraction p ~ Beta(α, β) scaled to [p_lo, p_hi].
grf_fraction_range: [0.1, 0.6667]      # [1/10, 2/3]
grf_fraction_beta: [2.0, 5.0]          # Beta(2, 5) — right-skewed toward sparse

# GRF kernel length scales [ℓ_y, ℓ_x] in units of chips on the 8×8 chip grid.
# Production: isotropic 1.5. Anisotropic alternative (e.g., [1.0, 4.0] for
# east-west urban corridors) is a one-line config change, no code change needed.
grf_corr_length: [1.5, 1.5]
grf_jitter: 0.000001                   # diagonal jitter for Cholesky stability

# Per-chip target count: linear-by-field-value mapping into [1, 10], plus
# Gaussian noise σ_count, clipped and rounded.
grf_count_range: [1, 10]
grf_count_noise_sigma: 1.0

# Parity rule modulus — idx % grf_absent_modulo == grf_absent_modulo - 1 ⇒ absent.
# Production: 4 ⇒ 25% absent (750 scenes of 3000).
grf_absent_modulo: 4

total_scenes: 3000
save_gifs: false       # see spec §5 step 5 — smoke run only
fps: 10

# Parallelism — carry forward v1's locked values pending re-tuning.
batch_size: 50
num_workers: 8

# Output — default to node-local storage per node, overridable via --output_root per §7.2.
output_root: <node-local data>/multi_target_swarm_1024x1024x64
```

### 4.3 Backward-compatibility requirements

The v2 extension preserves two backward-compatibility guarantees:

**(a) Single-target path bit-identical to pre-v1 reference.**
Running the v0.5 `slow_swarm.yaml` config (which sets
`target_sampler` to the default `"single"`) through the post-v2
generator on a fixed seed produces bit-identical `.npy` output to
the pre-v1 reference fixture at `tests/data/swarm_compat_ref/`.
This is the same regression test v1 introduced; v2 must also pass
it (per §6.6).

**(b) Chip-aware path bit-identical to v1 reference.** Running
the v1 `multi_target_swarm_1024.yaml` config (with
`target_sampler="chip_aware"`) through the post-v2 generator on
a fixed seed produces bit-identical `.npy` output to a v1-era
chip-aware reference fixture. The v1 spec did not introduce a
chip-aware backward-compat fixture (the v1 spec was its own
implementation point); the v2 spec introduces one.

**v2 fixture creation procedure.** Before any of the §4.1 Changes
land, the implementer of v2 performs Step 0:

- *Pre-v2 single-target fixture (already exists in repo):*
  `tests/data/swarm_compat_ref/` was committed during v1
  fixture-prep. No new action; the existing fixture serves the
  single-target regression.
- *Pre-v2 chip-aware fixture (new):* on the pre-v2 `HEAD` (i.e.,
  current `main` at v2-start time), generate 2 chip-aware scenes
  via:
  ```bash
  python scripts/data/generate.py \
      --config configs/data/multi_target_swarm_1024.yaml \
      --total_scenes 2 \
      --batch_size 2 \
      --num_workers 1 \
      --output_root /tmp/swarm_chip_aware_compat_ref_genN
  ```
  The `--batch_size 2 --num_workers 1` overrides force a single
  worker generating both scenes from one RNG (`default_rng(0)`
  per the v1 worker convention), which is the deterministic
  configuration that §6.6(b)'s regression test will reproduce.
  If the YAML's default batch_size and num_workers were used
  instead, scene 1 would get its own worker with `default_rng(1)`
  (per the seed = start_idx convention), and the fixture would
  still be deterministic but coupled to whatever batch arithmetic
  the YAML implies — fragile across YAML edits. The CLI overrides
  pin the fixture to a single canonical configuration.
  Note: the v1 YAML uses `image_size: [1024, 1024]` and
  `n_circles: 1920000`, which produces two ~64 MiB NPYs per scene.
  Total fixture size ~256 MiB on disk. **This is too large for
  a git fixture.** Instead, compute and store **SHA256 hashes of
  the four file contents** in `tests/data/swarm_chip_aware_compat_ref_hashes.json`:

  ```json
  {
    "vid0_1.npy":      "<sha256>",
    "vid1_0.npy":      "<sha256>",
    "vid0_1_mask.npy": "<sha256>",
    "vid1_0_mask.npy": "<sha256>"
  }
  ```

  Commit only the hashes (a tiny JSON file). The v2 chip-aware
  regression test (§6.6(b)) regenerates the same 2 scenes with the
  post-v2 generator on the same seed and compares SHA256 hashes
  to the committed JSON; no large fixture data lives in the repo.

The same hash-only approach could in principle be used for the
single-target fixture; v1 already committed the four ~10 KiB
single-target NPYs into `tests/data/swarm_compat_ref/` (small
enough at 128²) and that fixture is retained as-is for backward-
compat continuity. The chip-aware case is large enough that
hashes-only is the only practical option.

**Acceptance criterion (must-have).** Both regression tests pass
on the post-v2 generator. If either fails, treat as a regression
and investigate — the §4.1 design preserves the v1 RNG-draw orders
for `single` and `chip_aware` paths intact, and a bit-identity
failure indicates a real bug.

---

## 5. Execution plan

### 5.1 Order of operations

0. **Pre-change fixture creation** (per §4.3). On the pre-v2 `HEAD`,
   generate the 4-file chip-aware reference set via the v1 YAML,
   compute SHA256 hashes, commit
   `tests/data/swarm_chip_aware_compat_ref_hashes.json` as a
   fixture-prep commit. **Must precede Step 2.**
1. **Discard v1 dataset on-disk.** Delete the v1 dataset at:
   - `$LSL_DATA_ROOT/multi_target_swarm_1024x1024x64/`
     (all three of `data_all/`, `label_msk/`, `data_all_gif/`,
     `label_msk_gif/`, `dataset_metadata.json`).
   - `<node-local data>/multi_target_swarm_1024x1024x64/`
     on each of the three training nodes.
   - workstation-side `<node-local data>/...` if any v1 replica exists there (per v1
     §7.3 the workstation didn't host the v1 dataset; the implementer verifies).

   Deletion uses `rm -rf` per directory. Confirm post-deletion
   that the canonical shared copy path is empty (`ls -la` returns only
   `.` and `..`) and each node's local storage path is also empty.
   This step is the storage-recovery prerequisite for the v2
   generation (~375 GiB recovered on shared storage, ~125 GiB recovered
   per training node).

   **Irreversibility note.** Step 1 is the only irreversible step
   in this plan. Between Step 1 (v1 deletion) and Step 9 (v2
   consolidation to canonical shared copy), the canonical path is empty
   and the dataset does not exist in a usable form. If the v2
   generation fails between these steps, the recovery is: relaunch
   the v2 generation (Step 8). The v1 dataset can also be
   regenerated from scratch via the v1 YAML and the chip_aware
   sampler that remains in the post-v2 code — same regeneration
   path the v1 handoff §3 documents, with the same ~4.5 h per-node
   wall.
2. **Implement Changes 1-8 from §4.1.** Pure dev work on
   `src/localizer/synth/swarm.py` and
   `scripts/data/generate.py`.
3. **Create the new YAML config** at
   `configs/data/multi_target_swarm_1024_v2.yaml`
   per §4.2.
4. **Implement the test suite per §6** (six required tests: five
   new GRF-specific tests + one backward-compat regression set —
   two regression checks, single and chip-aware). Run with
   `pytest tests/`. All tests must pass.
5. **Smoke run on node 1: ~20 scenes** (indices 0-19), `save_gifs=true`,
   to a temp directory
   `<node-local data>/multi_target_swarm_1024_v2_smoke/`.
   Wall-clock and per-scene memory recorded. GIFs visually inspected
   for these properties (expected per Beta(2,5) sparse-skewed mass):
   (a) target-present scene density varies meaningfully scene-to-scene,
   with most scenes sparse-to-moderate (k typically in [6, 25]) and
   at least the densest 1-2 of the 15 present scenes showing visibly
   higher density than the rest — a 10000-scene Monte Carlo estimates
   p(max k over 15 ≥ 25) ≈ 83%, p(max k over 15 ≥ 30) ≈ 34%;
   (b) non-empty chips form *contiguous* clusters, not salt-shaker
   scatter — the principal structural property of v2 vs v1, expected
   on every present scene;
   (c) target counts within clusters peak at cluster cores (visible
   density gradient from cluster center outward) — expected on
   every present scene;
   (d) target-absent scenes (idx % 4 == 3, i.e., indices 3, 7, 11,
   15, 19) have all-zero masks;
   (e) bg swarm density is unchanged from v1.

   The 20-scene smoke covers 15 target-present (indices 0,1,2,4,5,6,
   8,9,10,12,13,14,16,17,18) and 5 target-absent (3,7,11,15,19).
   If property (a) is borderline (e.g., max k over 15 present scenes
   is only ~20), this is consistent with the right-skewed Beta(2,5)
   distribution and is not a failure — extend the smoke to 40 scenes
   if a richer density-range demonstration is wanted before bulk launch.
6. **Worker-count and batch-size tuning** based on smoke results.
   Per-scene memory and rendering load are identical to v1 (same
   1.92M-circle swarm, same N-target rendering math); v1's tuned
   `num_workers=8, batch_size=50` should carry forward unless the
   smoke surfaces new pressure. Lock tuned values in the YAML
   and commit if changed.
7. **Storage preflight on all 3 training nodes and the shared storage** per
   §2: confirm ≥150 GiB free on each node's local storage AND ≥400 GiB
   free on the shared storage *after* Step 1 v1-deletion has propagated.
   Abort if not met.
8. **Bulk generation on 3 training nodes** per §7.2: 1000 scenes per training node,
   parallel, `save_gifs=false`. Wall-clock per-node recorded.
   First 50 scenes per training node generated with `save_gifs=true` into
   a sibling smoke directory for documentary inspection. Bulk
   itself produces NPYs only.
9. **Consolidate to canonical shared copy** per §7.3. Each training node rsyncs
   its 1000-scene slice plus its 50-scene smoke GIF subset to the
   canonical shared copy path
   `$LSL_DATA_ROOT/multi_target_swarm_1024x1024x64/`.
   Use `rsync -rt --no-g --no-o --no-p` per the project conventions (the v1
   handoff §6 documented this requirement; v2 inherits).
10. **Verify per §7.6** — cross-node hash spot-checks, dataset
    integrity (all 6000 expected filenames present, all `.npy`
    files loadable to expected shape/dtype), realized-N distribution
    matches §3.2.2 prediction within tolerance.
11. **Generate `dataset_metadata.json`** per §7.5 and commit it
    to the canonical shared copy location.
12. **Hand off** — write a results-handoff narrative
    (`multi_target_swarm_1024_v2_dataset_handoff.md (originating repository; not shipped)`)
    summarizing per-node wall, realized per-scene target count
    distribution, smoke-run sanity checks (especially cluster-
    contiguity verification, since this is the v1→v2 structural
    delta), and any deviations from spec.

**Execution order.** fixture prep -> implementation -> discard v1 -> smoke run -> bulk
generation (about 4.5 h per 1000-scene slice) -> consolidate -> handoff record.

### 5.2 Per-node partitioning

Identical to v1 §5.2:

| Node | --start_idx | --total_scenes |
|---|---|---|
| node 1 (node 1) | 0 | 1000 |
| node 2 (node 2) | 1000 | 1000 |
| node 3 (node 3) | 2000 | 1000 |

(Three training nodes × 1000 scenes = 3000 scenes total.) With the v2
parity rule (`idx % 4 == 3 → absent`), each node's 1000-scene
slice contains exactly 250 target-absent scenes (indices ending
in 3, 7, 11, ..., 999 within each block of 1000) and 750
target-present. The overall 750/2250 absent/present ratio is
realized exactly, not just in expectation.

---

## 6. Testing strategy

Six tests gate acceptance. All must pass before the smoke run
(§5 step 5) launches. The test set covers: five new GRF-specific
tests + two regression tests (single-target and chip-aware) under
a combined §6.6.

### 6.1 Test: `test_swarm_grf_n_distribution`

Run the GRF-clustered sampler 200 times on a synthetic 1024×1024
scene shell. Setup: construct a `DataGenerator` instance with the
production v2 YAML's grf params (`target_sampler="grf_cluster"`,
`chip_size=128`, `image_size=[1024, 1024]`, plus the grf_* fields)
and `dat_lab_paths` pointing to throwaway temp dirs (the test
doesn't actually save files). For 200 fixed seeds, construct
`rng = np.random.default_rng(seed)` and call
`_sample_grf_cluster_targets(rng)`. For each call, count:

- `k_realized` = number of non-empty chips (recoverable from the
  test by mirroring the Stages A-C logic: same `rng.beta` then
  same `rng.standard_normal(64)` would re-derive k; alternatively,
  count distinct chip indices in the returned `target_positions`
  array via `chip_idx = (cy // chip_size) * 8 + (cx // chip_size)`).
  Assert `k_realized ∈ [6, 43]` for every run
  (= `[round(0.1·64), round(0.6667·64)]`).
- `N_realized` = total target count (= `target_positions.shape[0]`).
  Assert `N_realized ∈ [6, 430]` for every run.
- Per-chip target count `m_c` for each non-empty chip: assert
  `1 ≤ m_c ≤ 10` (the clip bound, inviolable).
- Empirical mean over the 200 runs: `mean(N_realized) ∈ [50, 90]`
  (target ~68 per §3.2.2, ±30% wall to absorb sampling noise from
  the 200-run pool).
- Empirical mean: `mean(k_realized) ∈ [14, 22]` (target ~17,
  ±30% wall).

### 6.2 Test: `test_swarm_grf_fraction_distribution_shape`

Same 200-run sweep, capturing the per-scene `u ~ Beta(2,5)` draw.
Verify the empirical distribution shape:

- Histogram into 10 bins on `[0, 1]`. The leftmost bin (u in
  `[0, 0.1]`) and the second-leftmost (`[0.1, 0.2]`) together
  should contain ≥30% of the mass. Beta(2,5)'s true CDF at 0.2 is
  ~34%, so the expected fraction is ~0.34; a 200-sample empirical
  draw will scatter around this with σ ≈ sqrt(0.34·0.66/200) ≈ 0.034,
  so ±2σ gives `[0.27, 0.41]` — the test threshold ≥0.30 sits at
  about -1.2σ from the mean and gives ~88% per-run pass probability.
  (If flakiness emerges, the threshold can be lowered to ≥0.27 or
  the sample size raised to 500; the test is documenting an
  empirical regularity, not a tight statistical bound.)
- The rightmost two bins (`[0.8, 1.0]`) together should contain
  ≤5% of the mass (Beta(2,5)'s survival function at 0.8 is ~0.16%,
  so 5% is many σ above expectation — generously safe).
- Empirical mean `mean(u) ∈ [0.22, 0.36]` (Beta(2,5) mean is
  2/7 ≈ 0.286; the test interval is ±0.07 ≈ ±2.5σ on a 200-sample
  draw, safe).

This test verifies the Beta-draw is implemented correctly; the
production version of the `p` scaling (Stage A of §3.2) just
remaps `u` linearly into `[0.1, 0.6667]`, which is checked
implicitly by §6.1's `k_realized` range.

### 6.3 Test: `test_swarm_grf_field_contiguity`

The defining structural property of v2 vs v1: non-empty chips
form contiguous clusters, not scattered. This test verifies that
property on a sample of generated chip masks.

Setup: same as §6.1 (DataGenerator with production grf params).
Run 50 fixed-seed scenes. For each scene, call
`generator._sample_grf_cluster_targets(rng)` to get
`target_positions` (N, 2). Derive the 8×8 binary chip mask by:

```python
n_chips_per_side = 8
chip_mask = np.zeros((n_chips_per_side, n_chips_per_side), dtype=bool)
for cx, cy in target_positions:
    chip_row = int(cy) // chip_size  # = 128 for production
    chip_col = int(cx) // chip_size
    chip_mask[chip_row, chip_col] = True
```

Compute the number of connected components in `chip_mask` using
4-connectivity (`scipy.ndimage.label` with cross-shaped structuring
element `[[0,1,0],[1,1,1],[0,1,0]]`).

A 200-run sampler simulation (iid-standard-normal field with
correlation length 1.5, the production setting) gives empirical
mean ≈ 2.5 connected components per scene, median 2, max 5, with
≥80% of scenes having ≤ 3 components.

Assertion: empirical mean number of connected components over 50
scenes ≤ 4, and at least 30 of the 50 scenes have ≤ 3 connected
components. The thresholds are calibrated against the simulation
above with a generous tolerance for sampling noise (50-scene mean
is a noisier estimator than 200-scene). They fail clearly if the
GRF mechanism is broken and producing salt-shaker output (which
would give ≈ k/2 components on average at k ≈ 17 — i.e., 8–9 per
scene at low k).

The test is most discriminative at low k (sparse scenes), where
isolated salt-shaker chips would produce many singletons. The §3.2
right-skewed Beta on `p` ensures the test sample is dominated by
low-k scenes; the contiguity assertion biases toward catching the
key failure mode of the GRF mechanism (a broken implementation
that thresholds white noise instead of the smooth field).

### 6.4 Test: `test_swarm_grf_count_gradient`

The per-chip count is supposed to correlate with the GRF field
value: cluster cores get more, edges get fewer.

Setup: same as §6.1 (DataGenerator with production grf params,
200 fixed seeds). For each seed, the test cannot directly observe
the GRF field `f` (the sampler doesn't return it). Instead, the
test **mirrors** the Stages A-C logic deterministically: given the
same `rng = np.random.default_rng(seed)`, draw `u ~ rng.beta(α, β)`,
then `z ~ rng.standard_normal(64)`, then compute `f = L @ z` using
the same precomputed Cholesky `L` from `generator._grf_chol_L`.
Then call `generator._sample_grf_cluster_targets(rng_copy)` on a
*fresh* copy of the same seed to get the actual targets. The
per-chip counts `m_c` are recovered by binning the returned
`target_positions` into chip indices and counting.

For each scene with k ≥ 4 non-empty chips, compute the rank
correlation (Spearman's ρ) between `f_c` (the mirrored field
value at chip c) and `m_c` (the empirical per-chip count).
Assertion: **median rank correlation across the 200 scenes ≥ 0.5**
(positive monotonic — high field values give high counts — but
loose tolerance for the σ=1 noise term). 1000-trial Monte Carlo
estimates median ρ ≈ 0.91, so 0.5 is a generously safe lower
bound.

Additional structural check: bin chips by their normalized field
rank (`(f_c - f_min) / (f_max - f_min)` ∈ [0, 1] within each scene)
into quartiles. **The top quartile's mean `m_c` should be ≥ 1.5×
the bottom quartile's mean `m_c`** (cluster-core chips average
meaningfully more targets than edge chips). Monte Carlo estimates
this ratio ≈ 4.5 in expectation, so 1.5 is a safe lower bound.

**Practical implementation note:** the "mirror the draws then call
the sampler" pattern requires two RNG instances with the same seed
because the sampler consumes its rng. The implementer can use
`rng1 = np.random.default_rng(seed); rng2 = np.random.default_rng(seed)`
and use rng1 for the mirror (Stages A-C only — draw `u`, `z`,
compute `f`) and rng2 for the actual sampler call. The mirror's
state diverges from the sampler's after Stage C (sampler continues
with Stage D draws; mirror stops). Both produce the same `f`.

### 6.5 Test: `test_swarm_grf_mask_and_absent_parity`

Reduced-size test (fast). Use config overrides:
- `image_size=[256, 256]`, `chip_size=64` → 4×4 = 16 chips total.
- `grf_fraction_range=[0.1, 0.6667]`, `grf_fraction_beta=[2.0, 5.0]`,
  `grf_corr_length=[1.5, 1.5]`, `grf_count_range=[1, 5]` (kept low
  for test speed), `grf_count_noise_sigma=1.0`, `grf_absent_modulo=4`.
- `n_circles=2000` (proportional to production scaled by area).
- `target_sampler="grf_cluster"`, `shape="square"`, `square_edge=2`.

Generate scenes 0, 1, 2, 3 (which under the parity rule are
present, present, present, absent).

Verify:
- Output mask shape `(64, 256, 256)` uint8 for all four.
- Scenes 0, 1, 2 have non-zero mask voxels on frame 0; scene 3
  has all-zero mask across all 64 frames.
- For scenes 0, 1, 2, mask positive-pixel count on frame 0 is
  between `N_realized · 4 · 0.9` and `N_realized · 4` (each
  target is a 2×2 square = 4 pixels; up to ~10% overlap reduction
  is acceptable).
- For scene 3 (target-absent), verify per-scene RNG-state
  determinism: re-run the same 4-scene generation a second time
  with the same seed, and assert the resulting `vid3_0.npy` file
  hashes byte-identically. This is a regression test on the v2
  generator's per-scene determinism (the same property v1
  chip_aware has, where same idx → same output).

### 6.6 Test: `test_swarm_backward_compat_v0_5_v0_6_and_v1`

**Two regression checks bundled in one test (or two parameterized
sub-tests; the implementer's call on structure).**

*(a) Single-target regression.* Run the v0.5/v0.6 `slow_swarm.yaml`
config through the post-v2 generator with `--total_scenes 2` and
the same RNG seed used in the v1-era fixture-prep. Assert the
four resulting `.npy` files are byte-identical to
`tests/data/swarm_compat_ref/` (the v1-era committed fixture).
This is the v1 §6.5 regression, carried forward unchanged.

*(b) Chip-aware regression.* Run the v1 `multi_target_swarm_1024.yaml`
config through the post-v2 generator with `--total_scenes 2` and
the same RNG seed used in the v2-era fixture-prep (§4.3 Step 0).
Compute SHA256 hashes of the four resulting `.npy` files and assert
they match the hashes committed in
`tests/data/swarm_chip_aware_compat_ref_hashes.json`. This is the
v2-new regression for chip-aware-path invariance.

If either sub-check fails, do not silently fall back; investigate
the regression. The §4.1 Change 1-8 design is structured to keep
both prior sampler RNG paths untouched.

---

## 7. Operations and dataset assembly

### 7.1 Per-node generation

Each training node runs the generator independently per §5.2 partitioning.
Per-node invocation:

```bash
# On node 1 (start_idx=0, count=1000)
python scripts/data/generate.py \
    --config configs/data/multi_target_swarm_1024_v2.yaml \
    --start_idx 0 \
    --total_scenes 1000 \
    --output_root <node-local data>/multi_target_swarm_1024x1024x64
```

Wrap in tmux for ssh-disconnect survival (generation is CPU-only;
the v0.6 / v1 launcher pattern carries forward — `set -o pipefail`
without `set -e`, conda env activation, sentinel start/exit lines,
trap-outside-tee for SIGINT). Launchers at
`scripts/launch_multi_target_swarm_1024_v2_{1,2,3}.sh` (parallel
to v1's per-node launchers but pointing to the v2 YAML).

### 7.2 Three-node parallel execution

Identical to v1 §7.2. The three training nodes generate their 1000-scene
slices simultaneously into their own node-local storage paths, disjoint
scene-index ranges, no inter-training node coordination during generation.

**Wall-clock estimate from v1 measurement.** Per the v1 handoff
(commit `7a854ff`), per-node wall was 4.44–4.48 h for 1000 scenes
at 8 workers / batch_size=50. v2 per-scene generation work is
**identical** in dominant cost terms (same swarm rendering, same
`N`-target rendering math) — the new sampler stages (Beta, GRF
Cholesky multiply, threshold, per-chip noisy count) add O(64)
work per scene, negligible against the 1.92M-circle render. v2
wall should be within ±10% of v1's 4.5 h per training node.

The v1 §7.2 fallback to v0 disk-mode density (`n_circles=630000`,
3× faster) remains authorized if smoke surfaces unexpected
slowdown, but is not expected to be needed.

### 7.3 Consolidation to canonical shared copy

Identical to v1 §7.3 in mechanism. After §5 step 1's v1-deletion,
the canonical shared copy path is empty; each training node rsyncs its 1000-scene
slice plus the 50-scene smoke GIF subset to the canonical path.

`rsync -rt --no-g --no-o --no-p` per the project conventions ("rsync to shared storage:
do NOT use `-a`"). The v1 handoff's rsync recovery
documentation (commit `b1e7f85` to the project conventions) is now stable
guidance; v2 follows the same flags from the start.

### 7.4 Replication strategy

Per-node replication is the training-spec's concern, not this
dataset spec's. Each training node already has its local NVMe copy from
the §5 step 8 bulk generation (no separate replication step
needed). workstation-side replication, if a training spec wants it,
happens at training-launch time via `rsync` from canonical shared copy
to workstation node-local storage.

### 7.5 `dataset_metadata.json` schema

Single file at canonical shared copy root, generated by the consolidation
step. v2 schema:

```json
{
  "dataset_name": "multi_target_swarm_1024x1024x64",
  "dataset_version": "v2",
  "supersedes": "v1 (commits db104ee, 0f8fee1, 7a854ff)",
  "spec_doc": "docs/specs/multi_target_swarm_1024_v2_dataset_spec.md",
  "spec_commit": "<git SHA of the v2 spec when generation was launched>",
  "generator_commit": "<git SHA of the generator code at gen time>",
  "yaml_config": "configs/data/multi_target_swarm_1024_v2.yaml",
  "yaml_config_resolved": {
    "n_frames": 64,
    "image_size": [1024, 1024],
    "n_circles": 1920000,
    "speeds": [0.5, 1.0, 2.0],
    "target_sampler": "grf_cluster",
    "chip_size": 128,
    "grf_fraction_range": [0.1, 0.6667],
    "grf_fraction_beta": [2.0, 5.0],
    "grf_corr_length": [1.5, 1.5],
    "grf_count_range": [1, 10],
    "grf_count_noise_sigma": 1.0,
    "grf_jitter": 1e-6,
    "grf_absent_modulo": 4
  },
  "total_scenes": 3000,
  "present_absent_split": {
    "present_count": 2250,
    "absent_count": 750,
    "parity_rule": "idx % 4 == 3 → absent"
  },
  "scene_index_partitioning": {
    "node_1": {"start_idx": 0, "count": 1000},
    "node_2": {"start_idx": 1000, "count": 1000},
    "node_3": {"start_idx": 2000, "count": 1000}
  },
  "train_val_split": {
    "scheme": "MoverDataset.make_dataloaders with sklearn train_test_split",
    "val_split": 0.1,
    "val_split_seed_default": 0,
    "stratified_by_label": false,
    "note": "Random non-stratified across 3000-scene index range."
  },
  "per_scene_realized_target_counts": {
    "min": "<filled at gen time>",
    "max": "<filled at gen time>",
    "mean": "<filled at gen time>",
    "median": "<filled at gen time>",
    "p25": "<filled at gen time>",
    "p75": "<filled at gen time>",
    "histogram_bins": "<filled at gen time>"
  },
  "per_scene_realized_k": {
    "min": "<filled at gen time>",
    "max": "<filled at gen time>",
    "mean": "<filled at gen time>",
    "median": "<filled at gen time>",
    "histogram_bins": "<filled at gen time>"
  },
  "per_node_wall_seconds": {
    "node_1": "<filled>",
    "node_2": "<filled>",
    "node_3": "<filled>"
  },
  "generated_at": "<ISO 8601 timestamp>"
}
```

New fields vs v1: `dataset_version`, `supersedes`,
`present_absent_split`, `per_scene_realized_k` (separate from
`per_scene_realized_target_counts` so post-hoc analysis can recover
both the per-scene chip-fraction realization and the per-scene N
realization). The `train_val_split.stratified_by_label` field is
new and explicitly set to `false` to document the design choice
from §1.

### 7.6 Verification

After §7.3 consolidation, run on at least one training node and on the
canonical shared copy:

- **Filename completeness:** all 6000 expected `.npy` files exist
  (3000 data + 3000 mask). Verify the parity rule by checking the
  label suffix on a random sample: for each of 30 random indices,
  confirm `vid{idx}_{label}.npy` exists with `label = 1` iff
  `idx % 4 != 3`, else `label = 0`.
- **Spot-check shape/dtype:** load 10 random scenes via `np.load`,
  assert shape `(64, 1024, 1024)` and dtype `uint8`.
- **Cross-node hash spot-check:** for 10 randomly-chosen scene
  indices, hash both `data_all/vid{idx}_{label}.npy` and
  `label_msk/vid{idx}_{label}_mask.npy` on the generating training node
  and on the canonical shared copy. Assert hashes match.
- **Per-scene size sanity:** assert each loaded array has
  `arr.nbytes == 67_108_864`.
- **Realized-N distribution sanity:** on a 300-scene random sample
  (target-present only — filter `idx % 4 != 3`), compute
  `N_realized` per scene by counting 8-connectivity components in
  frame 0 of the mask. Assert `mean(N) ∈ [45, 90]` (the §3.2.2
  prediction is ~68, with wide tolerance for sampling noise on a
  300-scene sample). Persist the histogram into
  `dataset_metadata.json`.
- **Realized-k distribution sanity:** same 300-scene sample,
  compute per-scene `k` (number of distinct 128² chips that
  contain at least one mask voxel on frame 0). Assert
  `mean(k) ∈ [12, 22]`, all individual `k` in `[6, 43]`.
- **Cluster contiguity check:** on a 50-scene random sample,
  build the per-scene 8×8 binary chip-mask from frame 0 mask
  voxels, compute connected-component count (4-connectivity);
  assert median across 50 scenes ≤ 3. This is the v2 structural
  invariant most worth verifying post-generation.

---

## 8. Things deliberately out of scope

All v1 §8 items carry forward unchanged. Repeating here for the
v2 spec's self-containment:

- **Per-target instance masks.** Union mask only; per-target
  instance segmentation can be recovered post-hoc via
  connected-component labeling.
- **PSF rendering, subpixel target rendering, target intensity
  variation.** Deferred to the eventual full-fidelity generator.
- **Textured background, cloud layers, sensor noise.** Deferred;
  pixels still binary {0, 255}.
- **Grouped / clustered target structures *within* a chip.**
  Per-chip targets are placed independently within the chip's
  pixel extent. Inter-target clustering at sub-chip scale is
  a separate axis.
- **Inter-target collision physics.** Targets and swarm members
  motion-independent; co-located positions render as a single 255
  pixel (binary OR).
- **Continuous velocity sampling.** Discrete 3-speed × 8-direction
  preserved.
- **Train/val pre-split into separate directories.** One bucket
  of 3000 scenes; split at dataloader time.
- **A dedicated v0.7 (or higher) training spec.** This dataset
  spec deliberately does not bind to any training spec. Reusable
  across future 1024² training experiments.

v2-specific out-of-scope items:

- **Per-chip target count dependence on cluster-center distance
  via a separate field.** v2 uses the same GRF that determines
  chip presence to also determine per-chip counts (via the
  field-value rank); a separate field for counts is rejected as
  unnecessary complexity.
- **Anisotropic correlation in production.** Production uses
  isotropic `[1.5, 1.5]`. Anisotropic length scales are
  config-only and may be explored in a future ablation; not part
  of this spec's deliverable.
- **Multiple cluster types within a scene** (e.g., "downtown"
  + "industrial park"). The single-GRF mechanism produces a
  single contiguity structure; multi-modal clustering would
  require a different point-process construction (e.g., Matérn
  cluster process). Not in scope.

---

## 9. Risks

- **Per-scene class balance shifts ~2.6× toward background.**
  E[N | present] drops from v1's ~176 to v2's ~68 (per §3.2.2),
  and 25% of scenes have zero targets (vs v1's 50%). Total
  target-voxel volume across the dataset drops accordingly.
  Any segmentation training spec consuming this dataset should
  re-tune `pos_weight` (or equivalent class-balance hyperparameter)
  upward relative to v1's tuning. This is a known consequence
  of the design, not a defect.
- **Realized k-distribution shape is sensitive to the Beta-parameter
  choice.** Beta(2,5) gives the right-skewed shape the spec
  intends. If the smoke run (§5 step 5) reveals the realized
  k-distribution feels "too sparse" (the model never sees a dense
  scene during training) or "too uniform" (the right-skew is
  insufficient), the parameters `(α, β)` can be retuned via YAML
  change without code changes. The spec does not lock retuning;
  the smoke run's purpose is to validate the production setting
  is acceptable.
- **GRF correlation length sensitivity.** Production uses 1.5
  chips. At correlation length 1.0, clusters become more compact
  (often single chips at low k); at 2.5, clusters become large
  but rare (one big cluster per scene). The 1.5 setting is the
  spec's prediction of "right"; the smoke run is what validates
  it. If GIFs show clusters that don't feel "urban-like enough"
  (e.g., too many isolated single-chip clusters at low p, or one
  giant cluster at moderate p), tune via YAML.
- **GRF Cholesky stability at large grids.** Not an issue for
  the production 8×8=64-node grid (Cholesky on 64×64 is trivially
  stable). If a future config wants finer-grain chip layout
  (e.g., 16×16 grid with 64×64 chips at 1024² resolution), the
  Cholesky may need larger jitter. Production setting is
  comfortably stable.
- **Bit-identity regression risk in v1 chip_aware path.** The
  v2 implementation extends `_generate_video`'s dispatch logic
  and refactors `_sample_multi_target_velocities` into two helpers
  (`_sample_bg_velocity` and `_sample_target_velocities`) with a
  thin wrapper preserving the v1 signature (per §4.1 Change 5).
  If the refactor inadvertently reorders any RNG draw on the
  chip_aware path, the §6.6(b) chip-aware regression fails.
  Mitigation: §4.3's fixture-prep step 0 captures the v1
  chip_aware hashes before any v2 code lands, so regression is
  detected at test time, not at generation time.

---

## 10. Open questions for user decision

(None known at draft time. The spec is internally consistent and
ready for acceptance. The user explicitly settled the four design
choices that produced the §3.2 sampler:
- Target-absent fraction = 25% (parity rule `idx % 4 == 3`).
- Per-scene fraction distribution = Beta(2,5) scaled to [1/10, 2/3].
- Spatial clustering = GRF via direct covariance, anisotropic-capable
  kernel, isotropic [1.5, 1.5] in production.
- Per-chip count = linear-by-field-value mapping into [1, 10] with
  σ=1 additive Gaussian noise, clipped.
- Sampler name = `grf_cluster`.
- RNG draw order = per §3.2.1; target-absent scenes skip Beta/GRF
  draws entirely (matches v1 chip_aware skip pattern).)
