> **Scrubbed copy.** Source: the originating repository's dataset spec at commit `a5c9c07`. Host names, storage paths and process notes from the originating compute environment were replaced with generic terms ("node 1-3", "shared storage", `$LSL_DATA_ROOT`); file paths were mapped to the current layout (`src/localizer/synth/swarm.py`, `configs/data/`, `scripts/data/generate.py`), but line-number references still point at the pre-reorganization files. The generation recipe, parameters, seeds and rationale are unchanged. Shell commands below are records of how the set was originally produced; for current, copy-pasteable commands see `docs/data.md`, which is authoritative where the two differ.

# Multi-target swarm 128×128×64 dataset spec — controlled diagnostic at the v0.6-success scale

## 0. Purpose and what this dataset is for

`multi_target_swarm_128` is a **controlled-isolation instrument**, not a
production dataset. Its purpose is to attribute the performance collapse
between v0.6 (working hybrid at 128² slow_swarm) and v0.8 (pure-Mamba ×
1024² × multi-target failed to learn spatial discrimination per the
v0.8 handoff §4.4) to one variable at a time, by holding **resolution,
object geometry, motion regime, and background swarm density at their
v0.6-success values** and changing **only the target distribution**
from single-target to GRF-clustered multi-target.

This dataset is designed to be paired with a four-arm training spec
(1 hybrid control + 3 pure-Mamba variants) run afterwards.
The dataset itself is hyperparameter-free with respect to architecture
or supervision; it produces `.npy` frames and masks like every other
dataset in the lineage, and the training spec decides what to do with
them.

### 0.1 What "controlled" means here

Three variables moved together between v0.6 and v0.8:

- Resolution: 128² → 1024²
- Scene complexity: single-target → multi-target swarm with GRF clustering
- Architecture: hybrid (working) → pure-Mamba × 1024² (failed)

`multi_target_swarm_128` removes the resolution variable (holds at 128²,
the v0.6-success value) and the architecture variable (the paired
training spec runs both hybrid and pure-Mamba on identical data), so
the only remaining axis for attribution is **scene complexity** — the
single change from `slow_swarm` (single-target) to this dataset
(GRF-clustered multi-target at 128²).

### 0.2 Anchor on v0.6-success conditions

Every parameter not changed by §0.1 is **inherited unchanged from
`slow_swarm.yaml`** (the v0.6 Arm 2 working dataset):

- Object geometry: 2×2 square (`shape: square, square_edge: 2`)
- Background swarm: 30,000 swarm members at 128² (already a saturated,
  overlapping, non-salt/pepper field — see §3.5)
- Velocities: `[0.0625, 0.125, 0.25]` (1/8-scaled from v0)
- `samedir: true` (target and background may share direction; the
  target is distinguished by *anomalous velocity*, not direction)

The only inherited-from-v2 element is the **GRF-clustered multi-target
sampler** itself (`target_sampler: grf_cluster`), with its parameters
**rescaled to a 4×4 chip grid at 128²** per §3.2.

### 0.3 Implementation: zero generator code changes

The existing `src/localizer/synth/swarm.py` already supports
every element of this dataset: the `shape="square"` path (slow_swarm /
v0.6), the `grf_cluster` sampler (v2), and a parametric `chip_size` and
GRF correlation length. **This spec is implementable as a single new
YAML config plus tests** — no changes to `data_generator.py`. §4
documents this explicitly with line references; §6.6 verifies it via
a regression test against the slow_swarm and v1 chip-aware fixtures.

### 0.4 Coupling to v0.9 training spec (declared dependency)

The eventual v0.9 training spec that uses this dataset will need the
v0.8 handoff §8 item 2 — `DetectionEpochAccumulator` multi-target
per-target hit accounting — because `multi_target_swarm_128` is
multi-target by construction. `full_recall@K` against multiple true
targets is not operationally meaningful under single-centroid hit
accounting. The dataset is fine to generate without this; the
training-spec readiness gate is the metric extension. **This is
flagged here so it doesn't surface as a surprise when training spec
drafting starts.**

---

## 1. Dataset identity

| Attribute | Value | Inherited from |
|---|---|---|
| Resolution | 128 × 128 × 64 | slow_swarm_128 |
| Scene count | 3000 (2250 present, 750 absent at 75/25) | new |
| Object geometry | 2×2 square | slow_swarm_128 |
| Background swarm | 30,000 members per scene | slow_swarm_128 |
| Velocities | `[0.0625, 0.125, 0.25]` | slow_swarm_128 |
| `samedir` | `true` | slow_swarm_128 |
| Target sampler | `grf_cluster` | v2 (rescaled) |
| Chip grid | 4 × 4 (chip_size = 32 px) | new (derived; §3.2.1) |
| GRF correlation length | ℓ = 0.75 chips | new (derived; §3.2.2) |
| GRF fraction range | `[1/10, 2/3]` | v2 (unchanged) |
| GRF fraction Beta | `(2, 5)` | v2 (unchanged) |
| Per-chip count range | `[1, 3]` | new (derived; §3.2.3) |
| Count noise σ | 1.0 | v2 (unchanged) |
| Target-present fraction | 75% (`grf_absent_modulo: 4`) | v2 (unchanged) |
| Parity rule | `idx % 4 == 3 → absent` | v2 (unchanged) |

**Identity hash (for reproducibility checks):** a `dataset_metadata.json`
will be written at generation time (§7.5) containing the exhaustive
config + git commit at generation + per-node partition map. Two
generations of this spec on different machines with the same git
commit and the same `(start_idx, batch_size, num_workers)` per training node
should produce byte-identical `.npy` outputs (§5.2.5).

### 1.1 Relationship to other datasets in the lineage

| Dataset | Resolution | Targets/scene | Object | Sampler | Velocity | Use |
|---|---|---|---|---|---|---|
| `slow_swarm` (v0.6) | 128² | 1 | 2×2 sq | `single` | slow | working anchor (v0.6 Arm 2) |
| `multi_target_swarm_1024` (v1) | 1024² | 24–80, uniform-chip | 2×2 sq | `chip_aware` | v0 (fast) | superseded |
| `multi_target_swarm_1024_v2` | 1024² | ~20–120, GRF-clustered | 2×2 sq | `grf_cluster` | v0 (fast) | production target |
| **`multi_target_swarm_128`** (this) | **128²** | **~3–12, GRF-clustered** | **2×2 sq** | **`grf_cluster`** | **slow** | **diagnostic instrument** |

Note the cross-cell: this dataset is the cross-product of
`slow_swarm`'s 128² + slow-velocity + saturated-background + 2×2-square
substrate with `multi_target_swarm_1024_v2`'s GRF-clustered multi-target
distribution. Both substrate and sampler already exist in the codebase;
their combination is new.

### 1.2 What this dataset is NOT

- Not a production dataset. The training spec that uses it should run
  cheap many-epoch experiments to settle attribution questions, then
  inform the design of a 1024²-targeted v0.9 production arm.
- Not a downsampled rendering of 1024² scenes. The 128² scenes are
  generated natively at 128² with the same generative process, not by
  rasterizing 1024² outputs at lower resolution. Downsampling 2×2
  targets through an 8× factor would make them sub-pixel and destroy
  the motion-discrimination signal that defines the problem.
- Not a "chip view" of 1024² scenes. The 128² scenes are
  **whole-scene miniatures**: same kind of internal structure (sparse
  with contiguous clusters, populated-vs-empty contrast within one
  scene), at 1/8 the linear scale. This is the design intent and
  §3.2 enforces it through grid-scale-invariant rescaling of the GRF
  parameters.

---

## 2. Storage and node budget

Each 128² × 64-frame scene at uint8 is **1 MiB per pair** (one
`a_*.npy` frames file + one `m_*.npy` mask file). 3000 scenes ≈ 3
GiB total. This fits comfortably on any one training node's local SSD and on
the workstation. No per-node partitioning is strictly required for storage,
but §5.2 still partitions across the three training nodes for parallel
generation speed, mirroring the v2 procedure.

Optional GIFs at `save_gifs: true` add ~50 KiB/scene = ~150 MiB total
across 3000 scenes. Recommended on for sample-inspection at §5.3 and
§7.6 visual-tuning gates; can be off for the canonical generation
run if storage is tight.

---

## 3. Dataset content: 128² GRF-clustered multi-target with slow velocities

### 3.1 Per-scene structure

Identical to v2 §3.1, restated:

- Frames tensor: `{output_root}/data_all/vid{idx}_{label}.npy`, shape
  `(64, 128, 128)`, uint8 in `[0, 255]`. Frame 0 is the initial state;
  frame `t > 0` advects all background-swarm members and all targets
  by their respective per-frame velocities (subpixel positions
  accumulated in float64, rendered to uint8 per frame).
- Mask tensor: `{output_root}/label_msk/vid{idx}_{label}_mask.npy`,
  shape `(64, 128, 128)`, uint8 in `{0, 1}`. Mask value 1 only at
  the 2×2 footprint of each target, per frame. Background swarm
  members are **never** marked.
- Filename `{label}` field is `0` or `1` per the v0/v1/v2 convention
  (`has_mover == 0` or `1`).
- Optional GIFs at `{output_root}/data_all_gif/vid{idx}_{label}.gif`
  and `{output_root}/label_msk_gif/vid{idx}_{label}_mask.gif` mirror
  the NPY paths when `save_gifs: true`.

### 3.2 GRF-clustered multi-target placement, rescaled for 128²

The sampler is `grf_cluster` with **three parameters rescaled** from
v2 to preserve the within-scene clustering character at the smaller
scene scale, plus one parameter (count range) reduced to keep the
per-pixel target density in a regime where targets do not saturate
the 128² scene.

#### 3.2.1 Chip grid: 4 × 4 (chip_size = 32 px)

**Choice:** `chip_size = 32`, giving `n_chips_per_side = 128/32 = 4`
and `T_chips = 16`.

**Why 4×4 and not 8×8 (which would mirror v2's grid exactly):**

- An 8×8 grid at 128² gives 16×16-pixel chips. Targets and the
  saturated 30,000-member background swarm both operate at that
  scale, and per-chip cluster structure becomes pixel-fine — the
  cluster pattern is in the data but is difficult to inspect
  visually and at the limit of what 2×2 targets can spatially
  distinguish from neighboring chips.
- A 4×4 grid at 128² gives 32×32-pixel chips: roughly 1/8 of the
  scene side per cell, **legible at this scale and large enough that
  cluster placement is visually obvious in sample GIFs**. This makes
  the §5.3 / §7.6 visual-tuning gate productive rather than ambiguous.

**Tradeoff explicitly accepted:** with only 16 chips and a typical
non-empty fraction `k/T_chips` of 3–4 chips (§3.2.2), the GRF's
spatial-contiguity mechanism has less room to express itself than at
T=64. Cluster contiguity is genuinely softer at 4×4 than 8×8; this
is an inherent property of the coarser grid, not a defect. The §6.3
contiguity test (rescaled from v2 §6.3) accepts a weaker contiguity
threshold to reflect this.

#### 3.2.2 GRF correlation length: ℓ = 0.75 chips

**Choice:** `grf_corr_length = [0.75, 0.75]` (isotropic).

**Derivation:** v2 uses ℓ = 1.5 chips on an 8×8 grid. The
correlation length **as a fraction of the grid linear extent** is
1.5/8 = 0.1875 (≈ 19% of the scene). This is the load-bearing
property: it controls cluster size as a fraction of the scene, which
in turn produces the populated-vs-empty *contrast within one scene*
that distinguishes v2 from a salt-pepper field. To preserve that
fraction at a 4×4 grid: ℓ_128 = 0.1875 × 4 = 0.75.

**Why this is non-obvious and surfaced here rather than copied from
v2:** keeping ℓ at v2's literal 1.5 on a 4×4 grid would put the
correlation length at 1.5/4 = 0.375 = 37.5% of the scene — nearly
the entire grid. The field would be smooth across most of the scene,
empty regions would barely exist, and the within-scene contrast that
defines "what a scene looks like" would be erased. This is exactly
the empirical-finding-propagation discipline the v0.8 closeout
flagged: when a load-bearing parameter is fixed by a property
external to itself (here, "cluster-as-fraction-of-scene"), every
derived parameter should be recomputed, not literal-copied.

**Sample-inspection override:** if the §5.3 visual gate finds 4×4 +
ℓ=0.75 produces clusters that are too soft (no visual contrast
between populated and empty regions) or too tight (clusters always
in adjacent cells), tune ℓ in `[0.5, 1.0]` and re-inspect. Wider
ℓ → smoother field → larger but fewer clusters; narrower ℓ → noisier
field → more scattered placement. ℓ=0.75 is the principled default;
empirical override is permitted at the visual gate but should be
recorded in `dataset_metadata.json` for reproducibility.

#### 3.2.3 Per-chip count range: [1, 3] with gradient preserved

**Choice:** `grf_count_range = (1, 3)`, `grf_count_noise_sigma = 1.0`
(unchanged from v2).

**Derivation:** v2 uses [1, 10] at 1024². On the 8×8 grid with
typical k ≈ 17 non-empty chips and mean per-chip count ≈ 4, v2 gives
~68 targets per present scene. Per-pixel density: 68 targets in
1,048,576 pixels = 6.5e-5 targets/pixel. At 128² preserving v2's
[1, 10] per cell with k ≈ 3–4 non-empty chips of 16 (§3.2.4) and a
mean per-chip count of ≈ 5 (due to gradient): ~17 targets per
present scene; per-pixel density 17 / 16384 = 1.0e-3 — **16× higher
density per pixel than v2**. That would change the *difficulty
character* of the problem (targets crowd the scene; many would
overlap or touch background-cluster cores) rather than just rescale.

Reducing to [1, 3] gives mean per-chip count ≈ 2 (with gradient
preserved by the linear-by-field-value mapping in
`_sample_grf_cluster_targets` Stage D — see §3.2.5), so typical k=3
non-empty chips × 2 targets ≈ 6 targets per present scene. Per-pixel
density 6 / 16384 = 3.7e-4 — within ~6× of v2 and consistent with
"sparse with contiguous clusters" rather than "densely packed."

**The core-to-edge gradient is preserved automatically.** The
linear mapping `μ_c = lo_count + (hi_count - lo_count) ·
(f_c - f_min) / range_f` in `_sample_grf_cluster_targets` keeps the
high-field-value chips at the upper count (3) and low-field-value
chips at the lower count (1) regardless of the count range chosen.
The gradient is intrinsic to the algorithm, not a function of the
range bounds. (§6.4 verifies this with a test.)

#### 3.2.4 Per-scene realized N — expected statistics

With Beta(2,5) on `[1/10, 2/3]` and T_chips = 16:

- Beta(2,5) mode at u_mode = 1/5 → p = 1/10 + 0.2 · 0.567 = 0.213,
  k = round(3.4) = 3 non-empty chips.
- Beta(2,5) mean at u_mean = 2/7 ≈ 0.286 → p = 0.262,
  k = round(4.19) = 4 non-empty chips.
- Lower edge p = 1/10 → k = round(1.6) = 2 non-empty chips.
- Upper edge p = 2/3 → k = round(10.67) = 11 non-empty chips.

Per-chip count distribution (gradient by field-value rank, range 1-3,
noise σ=1): mean ≈ 2, mode 2 with shoulders at 1 and 3.

**Expected realized N per present scene:** typical 3–4 chips × 2
targets ≈ 6–8; lower edge 2 × 1.5 ≈ 3; upper edge 11 × 2 ≈ 22.

**Rough N distribution (analytic; §6.1 will measure):**

| Percentile | Realized N (estimate) |
|---:|---:|
| p10 | 3 |
| p25 | 5 |
| p50 | 7 |
| p75 | 10 |
| p90 | 14 |
| p99 | 22 |

This is the "sparse with clusters" character `multi_target_swarm_128`
is designed to produce — comparable in *kind* to v2's scenes (clusters
of 1–10 in 3–17 non-empty regions of an 8×8 grid) at 1/8 the linear
scale.

#### 3.2.5 Per-chip placement within a 32×32 chip

`_sample_grf_cluster_targets` Stage D uniformly samples target start
positions in the rectangle `[cx_lo, cx_hi) × [cy_lo, cy_hi)` where
`cx_hi = chip_col · cs + cs - square_edge`. For cs=32 and
square_edge=2 this gives a 30×30 placement window per chip. Multiple
targets in the same chip are drawn independently uniform — they can
overlap in pixel coordinates at frame 0 (and will then advect under
independent per-target velocities, so by frame `t > 0` they
typically separate). This matches v2 §3.2 placement exactly.

#### 3.2.6 RNG draw order

Identical to v2 §3.2.1 — `_generate_video_grf_cluster` calls
`_sample_bg_velocity`, then either `_sample_grf_cluster_targets` and
`_sample_target_velocities` (present branch) or skips them (absent
branch). The RNG-draw contract for cross-node reproducibility holds
unchanged: same `(start_idx, batch_size, num_workers)` → byte-identical
`.npy` outputs.

### 3.3 Per-target velocity sampling

Identical to v2 §3.3 / v1 §3.3 (the anomalous-mover constraint):
`_sample_target_velocities(N, bg_dir, bg_spd, rng)` is invoked for
the realized N targets, drawing per-target `(tgt_dir, tgt_spd)` with
the velocity-anomaly constraint that each target's velocity is
distinct from the background velocity (per `samedir` and per the
v0/v0.6 anomalous-velocity logic in `_set_vels` / `_set_spds`).

**No code path changes vs v2 here.** The velocity-anomaly
implementation is in `_set_vels` and `_set_spds`, which read from
`self.speeds` and `self.dirs`; setting `speeds: [0.0625, 0.125, 0.25]`
in the YAML cascades through automatically.

### 3.4 Velocity scaling — slow (1/8 of v0)

`speeds: [0.0625, 0.125, 0.25]` (slow_swarm's values, 1/8 of v0's
`[0.5, 1.0, 2.0]`). At 128² these give spatially-compact trajectories
over 64 frames: max displacement = 0.25 × 64 = 16 pixels (1/8 of
scene side), matching slow_swarm's character. **This is the v0.6-success
velocity regime.** Velocity is inherited unchanged, not rescaled.

Note that v2 at 1024² uses fast `[0.5, 1.0, 2.0]` because over 1024
pixels the fast speeds still give spatially-compact (≈ 12.5% of scene
side) trajectories. The "scene-side fraction" displacement metric is
held invariant across the lineage; slow at 128² and fast at 1024²
are the *same* operating point in this scale-invariant sense. This
is the v0.6 motion regime carried forward, not a new design.

### 3.5 Background swarm — saturated, 30,000 members

`n_circles: 30000` (slow_swarm value, unchanged). At 128² this is a
**saturated, overlapping field**: 30,000 × 2² pixels = 120,000 covered
pixels in 16,384 total = 7.3× over-coverage, so members merge into
extended connected clusters across the frame. This is the v0.6-success
background regime and produces the **non-salt/pepper** field texture
intrinsically — no additional clustering machinery is needed for the
background.

(Note: this is *background swarm* density, distinct from §3.2's GRF
clustering of the *targets*. They are independent: targets cluster
into a few 32×32 regions by GRF; background members are uniformly
distributed across the scene but spatially overlap themselves. The
visual character of a `multi_target_swarm_128` scene is a saturated
field with 1–22 anomalous-velocity targets concentrated in 2–11
soft-clustered regions.)

### 3.6 Target-absent (label=0) scenes

`grf_absent_modulo: 4` per v2 §3.6: `has_mover = 0 iff idx % 4 == 3`.
On 3000 scenes: 750 absent (25%), 2250 present (75%). Absent scenes
skip §3.2 and §3.3 entirely (no targets to place or velocity-assign),
producing background-only frames with `m_*.npy` all zero. This is the
v2 absent-branch behavior carried forward unchanged.

---

## 4. Generator extension

### 4.1 Code changes to `src/localizer/synth/swarm.py`

**None.** The existing implementation already supports every element
of this dataset:

- `shape="square"` + `square_edge=2`: the v0.6 / slow_swarm path,
  present in `__init__` and `_make_square`.
- `target_sampler="grf_cluster"` with parametric `chip_size`,
  `grf_corr_length`, `grf_fraction_range`, `grf_fraction_beta`,
  `grf_count_range`, `grf_count_noise_sigma`, `grf_jitter`,
  `grf_absent_modulo`: present in `__init__`, `_build_grf_cholesky`,
  `_sample_grf_cluster_targets`, `_generate_video_grf_cluster`.
- `chip_size = 32` divides `image_size = 128` cleanly: the divisibility
  assertion at `__init__` line ~144 passes.
- `grf_count_range = (1, 3)`: the clip bounds in
  `_sample_grf_cluster_targets` Stage D are read from `self.grf_count_range`
  with no implicit assumption that the upper bound is 10.
- `grf_corr_length = [0.75, 0.75]`: the kernel `K = np.exp(-0.5 ·
  ((di/ly)² + (dj/lx)²))` in `_build_grf_cholesky` handles fractional
  ℓ exactly; the Cholesky succeeds with the unchanged `grf_jitter`
  (the matrix is well-conditioned at T_chips=16).

**This zero-code-change property is a load-bearing property of the
spec** (it ensures the prior `slow_swarm`, `chip_aware`, and
`grf_cluster` sampler paths are bit-unchanged), and §6.6 verifies
it via the existing two-fixture regression test.

### 4.2 New YAML config: `configs/data/multi_target_swarm_128.yaml`

```yaml
# multi_target_swarm_128 — 128×128×64 GRF-clustered multi-target dataset
# at the v0.6-success scale (slow velocities, saturated background swarm,
# 2×2-square objects), with the v2 GRF clustering rescaled to a 4×4 chip
# grid.
#
# Purpose: controlled-isolation diagnostic instrument paired with a
# four-arm training spec (1 hybrid + 3 pure-Mamba) to attribute the
# v0.6→v0.8 performance collapse to scene complexity, architecture,
# or scale. Holds resolution, object geometry, motion regime, and
# background density at v0.6-success values; changes only the target
# distribution from single to GRF-clustered multi.
#
# Spec: docs/specs/multi_target_swarm_128_dataset_spec.md
#
# Inherited from slow_swarm.yaml (v0.6-success values):
#   - image_size, shape, square_edge, n_circles, samedir, speeds
#
# Inherited from multi_target_swarm_1024_v2.yaml (clustering machinery):
#   - target_sampler: grf_cluster
#   - grf_fraction_range, grf_fraction_beta, grf_count_noise_sigma,
#     grf_jitter, grf_absent_modulo
#
# Rescaled for 128² (derived in spec §3.2):
#   - chip_size: 32 (gives 4×4 grid, was 128 on the 1024² 8×8 grid)
#   - grf_corr_length: [0.75, 0.75] (preserves cluster-as-fraction-of-scene
#     at 0.1875, was [1.5, 1.5] on the 8×8 grid)
#   - grf_count_range: [1, 3] (preserves per-pixel target density within
#     ~6× of v2, was [1, 10] at 1024²)

n_frames: 64
image_size: [128, 128]
shape: square
square_edge: 2
radius: 2          # unused when shape="square", kept for backward-compat
speeds: [0.0625, 0.125, 0.25]
n_circles: 30000
samedir: true
total_scenes: 3000
save_gifs: true
fps: 10

# Multi-target sampler — grf_cluster per spec §3.2.
target_sampler: grf_cluster

# 4×4 chip grid at 128² (32-pixel cells).
chip_size: 32

# Per-scene non-empty-chip fraction p ~ Beta(α, β) scaled to [p_lo, p_hi].
grf_fraction_range: [0.10, 0.6667]
grf_fraction_beta: [2.0, 5.0]

# GRF kernel length scales [ℓ_y, ℓ_x] in units of chips on the 4×4 grid.
# Rescaled from v2's [1.5, 1.5] on the 8×8 grid to preserve
# cluster-as-fraction-of-scene at 0.1875 (= 1.5/8 = 0.75/4).
grf_corr_length: [0.75, 0.75]

# Per-chip target count: linear-by-field-value mapping into [1, 3],
# plus Gaussian noise σ_count, clipped and rounded. Range reduced from
# v2's [1, 10] to keep per-pixel target density commensurate with v2 at
# 1024² (see spec §3.2.3). Core-to-edge gradient is preserved automatically
# by the linear mapping in _sample_grf_cluster_targets Stage D.
grf_count_range: [1, 3]
grf_count_noise_sigma: 1.0

# GRF Cholesky jitter — small diagonal added for numerical stability.
grf_jitter: 1.0e-6

# Parity rule modulus — idx % grf_absent_modulo == grf_absent_modulo - 1 ⇒ absent.
# 4 ⇒ 25% absent (750 scenes of 3000).
grf_absent_modulo: 4

# Parallelism.
batch_size: 250
num_workers: 12

# Output (default; overridable via --output_root for per-node generation).
output_root: <node-local data>/multi_target_swarm_128x128x64
```

### 4.3 Backward-compatibility requirements

Because §4.1 introduces zero code changes, the v0.6 / slow_swarm,
v1 chip_aware, and v2 grf_cluster sampler paths are guaranteed
unchanged at the source-code level. **The §6.6 regression test
verifies this empirically** by re-running the slow_swarm and v1
fixtures through the post-`multi_target_swarm_128` generator and
asserting byte-identical outputs.

If any future code change to `data_generator.py` is made
concurrently with this spec landing, that change must include §6.6
regression-test execution as part of its merge gate. The "no code
change" property of this spec is fragile against unrelated edits;
the test is the durable verification.

---

## 5. Execution plan

### 5.1 Order of operations

1. **Branch and YAML.** Create the new YAML at
   `configs/data/multi_target_swarm_128.yaml` per §4.2.
2. **Tests.** Add the six tests from §6 to
   `tests/test_swarm_grf_128.py` (new file; §6.6 may reuse the
   existing v1/v2 backward-compat fixture by parameterizing on
   config path).
3. **Smoke generation.** Run the YAML with `--total_scenes 20` on a
   workstation or training node. Verify:
   - Generator instantiates without assertion failures.
   - 20 scenes (15 present, 5 absent) are produced.
   - Per-scene realized N falls in the §3.2.4 expected range.
   - Sample GIFs at `save_gifs: true` show recognizable clustered
     placement.
4. **Visual-tuning gate.** Generate a 60-scene inspection batch with
   `save_gifs: true`. Hand-inspect 10 present scenes and 5 absent
   scenes for:
   - Cluster structure: targets visibly grouped into 2–4 regions per
     scene (not scattered uniformly, not all in one corner).
   - Background texture: saturated swarm covers the field
     (non-salt/pepper); target clusters are visible *against* this
     background, not within an empty field.
   - Velocity anomaly: targets visibly move differently from the
     background swarm flow.
   - Empty regions: typical scenes have ~12 of the 16 chips empty
     (per §3.2.4, typical non-empty k≈3-4 of 16). The visual signature
     of correct GRF behavior is that the ~3-4 non-empty chips tend to
     be **spatially adjacent** (forming a contiguous cluster region)
     rather than scattered across all four corners. If non-empty chips
     are consistently scattered (no contiguous cluster preference),
     the GRF is not doing its job and `grf_corr_length` is likely too
     small or `grf_jitter` is interfering with the field structure.
   - Conversely, if non-empty chips fill nearly all 16 cells in
     typical scenes, the Beta(2,5) fraction draw is mis-tuned or
     `grf_fraction_range` is wrong; investigate.
   - If clusters look too soft (uniform-ish) or too tight (always
     adjacent cells), tune `grf_corr_length` in `[0.5, 1.0]` per
     §3.2.2 and rerun the gate. Record the chosen value in
     `dataset_metadata.json`.
5. **Statistical-tuning gate.** Generate a 300-scene batch (10% of
   target). Run `tests/test_swarm_grf_128.py::test_swarm_grf_128_n_distribution`
   against this batch (§6.1). If the realized N distribution deviates
   substantially from §3.2.4 expectations (e.g., p50 < 4 or p50 > 12),
   investigate — likely a parameter mistuning.
6. **Full generation.** Run on all three training nodes in parallel per §5.2.
7. **Consolidation.** Aggregate the three per-node outputs to shared storage
   per §5.2.4.
8. **Verification.** Per §7.6.

### 5.2 Per-node partitioning

3000 scenes / 3 training nodes = 1000 scenes per node. Per-node `num_workers`
is set to `4` (matches the 4 batches of `batch_size=250` per training node; the
YAML's default `num_workers: 12` is sized for single-machine generation
of all 3000 scenes).

```bash
# node 1 (start_idx=0, count=1000)
python scripts/data/generate.py \
  --config configs/data/multi_target_swarm_128.yaml \
  --start_idx 0 --total_scenes 1000 --batch_size 250 --num_workers 4 \
  --output_root <node-local data>/multi_target_swarm_128x128x64

# node 2 (start_idx=1000, count=1000)
python scripts/data/generate.py \
  --config configs/data/multi_target_swarm_128.yaml \
  --start_idx 1000 --total_scenes 1000 --batch_size 250 --num_workers 4 \
  --output_root <node-local data>/multi_target_swarm_128x128x64

# node 3 (start_idx=2000, count=1000)
python scripts/data/generate.py \
  --config configs/data/multi_target_swarm_128.yaml \
  --start_idx 2000 --total_scenes 1000 --batch_size 250 --num_workers 4 \
  --output_root <node-local data>/multi_target_swarm_128x128x64
```

Alternative: single-machine generation of all 3000 scenes on the workstation
or one training node, using the YAML defaults (`--num_workers 12`) — appropriate
if a single node has spare capacity and the parallel multi-node coordination
overhead isn't worth it for a 3 GiB dataset.

Cross-node byte-reproducibility: per v2 §3.2.1, same
`(start_idx, batch_size, num_workers)` → byte-identical output. The
partition is contiguous (no scene-index gaps or overlaps), so simple
concatenation in §5.2.4 produces a valid 0..2999 indexed dataset.

### 5.2.4 Consolidation to canonical shared copy

```bash
mkdir -p $LSL_DATA_ROOT/multi_target_swarm_128x128x64
rsync -av node1:<node-local data>/multi_target_swarm_128x128x64/ \
  $LSL_DATA_ROOT/multi_target_swarm_128x128x64/
rsync -av node2:<node-local data>/multi_target_swarm_128x128x64/ \
  $LSL_DATA_ROOT/multi_target_swarm_128x128x64/
rsync -av node3:<node-local data>/multi_target_swarm_128x128x64/ \
  $LSL_DATA_ROOT/multi_target_swarm_128x128x64/
```

(Per-node generation writes to the same final directory structure;
`rsync` to shared storage merges them. Total ≈ 3 GiB, ≈ 30 s over 10 GbE.)

### 5.2.5 Replication strategy

Optional; cheaper than v2 because the dataset is small enough that
re-generation from spec is faster than copying. If replication is
desired (e.g., to node-local storage for fast local read), `rsync` from
the canonical shared copy post-consolidation.

---

## 6. Testing strategy

Six tests, mirroring v2 §6 structure. All in
`tests/test_swarm_grf_128.py` (new file). Tests 6.1–6.4 are new
adaptations of v2 §6.1–§6.4 with rescaled expected values; §6.5 is
the mask-and-absent-parity check; §6.6 is the backward-compat
regression (reuses v2 fixtures).

### 6.1 Test: `test_swarm_grf_128_n_distribution`

Sample 300 scenes from the YAML. Assert:

- Mean realized N (target-present scenes only) in `[6, 10]`
- Median in `[5, 9]`
- p25 in `[3, 6]`, p75 in `[8, 13]`
- Min ≥ 1, max ≤ 30 (the theoretical cap at k=16, m_c=3 is 48; in
  practice the Beta(2,5) tail makes k=16 vanishingly rare)
- Zero realized-N scenes are present-labeled (i.e., `has_mover==1 ⇒
  N ≥ 1`)

### 6.2 Test: `test_swarm_grf_128_fraction_distribution_shape`

Sample 300 scenes. Extract the per-scene `p = k/16` (number of
non-empty chips over 16). Assert the empirical distribution of `p`
is consistent with `Beta(2, 5)` scaled to `[0.10, 0.6667]`
(Kolmogorov-Smirnov D < 0.10, or equivalent — adapt v2 §6.2 test
exactly).

### 6.3 Test: `test_swarm_grf_128_field_contiguity`

Sample 300 present scenes. For each, extract the binary 4×4
chip-occupancy map (which chips are non-empty). Compute the
contiguity statistic (e.g., number of connected components of
non-empty chips under 4-neighbor adjacency; or mean inter-non-empty
distance vs random expectation).

**Threshold accommodation:** at 4×4 with typical k=3–4, the GRF
contiguity signal is weaker than at v2's 8×8 with typical k=17. The
test should assert that the GRF version is *more contiguous than
random uniform-without-replacement chip selection at the same k*
(Mann-Whitney p < 0.05, say), but should NOT assert the stronger v2
§6.3 contiguity bound (which assumes 8×8 grid statistics).

### 6.4 Test: `test_swarm_grf_128_count_gradient`

Sample 300 present scenes. For each, extract the per-chip GRF field
value and the per-chip realized target count. Assert:

- Spearman rank correlation between field value and target count
  (across non-empty chips, pooled across scenes) is positive and
  significant (ρ > 0.3, p < 0.001).
- This is the v2 §6.4 test exactly, retargeted at the new dataset.

### 6.5 Test: `test_swarm_grf_128_mask_and_absent_parity`

Sample 200 scenes. Assert:

- For each scene, the filename's `{label}` field
  (`vid{idx}_{label}.npy`, value `0` or `1`) matches its `has_mover`
  status (consistent with the v0/v1/v2 generator convention).
- For absent scenes (`has_mover==0`, `idx % 4 == 3`): the `m_*.npy`
  mask is all zeros.
- For present scenes: the mask has nonzero entries; the nonzero
  entries form contiguous 2×2 footprints (each connected component
  has exactly 4 pixels).
- The number of distinct 2×2 footprints in the mask at frame 0
  matches the realized N reported by the sampler.

### 6.6 Test: `test_swarm_backward_compat_slow_swarm_and_v1`

**Two-fixture regression check, mirroring v2 §6.6's pattern unchanged.**

- (a) Single-target regression: run `slow_swarm.yaml` with
  `--total_scenes 2` and the same RNG seed used in the v1-era
  fixture-prep. Assert byte-identical outputs against
  `tests/data/swarm_compat_ref/`.
- (b) Chip-aware regression: run `multi_target_swarm_1024.yaml`
  (v1) with `--total_scenes 2` and the v2-era fixture-prep seed.
  Assert SHA256 hashes match
  `tests/data/swarm_chip_aware_compat_ref_hashes.json`.

If either sub-check fails, the §4.1 zero-code-change property has
been violated by unrelated edits; investigate before landing.

**Note on the v2 grf_cluster path:** v2's own test suite covers the
`grf_cluster` sampler at 1024² directly (the existing v2 statistical
tests). This spec's §4.1 zero-code-change property implies the v2
grf_cluster path is untouched, so re-asserting it here would be
duplicate coverage. If `data_generator.py` is edited concurrently
with this spec landing in a way that touches the grf_cluster path,
that change is responsible for re-running the v2 test suite as part
of its merge gate.

---

## 7. Operations and dataset assembly

### 7.1 Per-node generation

Per §5.2. Each training node writes `(a_*.npy, m_*.npy, optional gif)` triples
to its node-local output_root.

### 7.2 Three-node parallel execution

Training nodes 1, 2, 3 launch in parallel; each completes ~1000 scenes in
~20–40 minutes at 128² (well under v2's 1024² generation time, which
is multi-hour per training node).

### 7.3 Consolidation to canonical shared copy

Per §5.2.4. `rsync` from each node's local storage to the canonical shared copy
directory. Total ≈ 3 GiB, ≈ 30 s.

### 7.4 Replication strategy

Optional per §5.2.5. The dataset is small enough that re-generation
from spec + YAML is fast (sub-hour total wall time across three
training nodes).

### 7.5 `dataset_metadata.json` schema

A new aggregation script `scripts/data/build_metadata.py`
writes `dataset_metadata.json` at the canonical shared storage output_root after
§7.3 consolidation, following the v2 pattern
(`scripts/build_multi_target_swarm_1024_v2_metadata.py`). The script
walks `data_all/` and `label_msk/`, counts present/absent scenes from
filenames, and computes the realized-N summary by counting 2×2 mask
footprints per scene at frame 0. Schema:

```json
{
  "spec": "docs/specs/multi_target_swarm_128_dataset_spec.md",
  "git_commit": "<hash at generation time>",
  "yaml_path": "configs/data/multi_target_swarm_128.yaml",
  "yaml_sha256": "<hash of resolved YAML>",
  "image_size": [128, 128],
  "n_frames": 64,
  "total_scenes": 3000,
  "target_present_count": 2250,
  "target_absent_count": 750,
  "target_sampler": "grf_cluster",
  "chip_size": 32,
  "n_chips_per_side": 4,
  "grf_corr_length": [0.75, 0.75],
  "grf_corr_length_overridden_at_visual_gate": false,
  "grf_fraction_range": [0.10, 0.6667],
  "grf_fraction_beta": [2.0, 5.0],
  "grf_count_range": [1, 3],
  "grf_count_noise_sigma": 1.0,
  "grf_absent_modulo": 4,
  "speeds": [0.0625, 0.125, 0.25],
  "n_circles": 30000,
  "samedir": true,
  "shape": "square",
  "square_edge": 2,
  "per_node_partitions": [
    {"node": "node 1", "start_idx": 0, "count": 1000},
    {"node": "node 2", "start_idx": 1000, "count": 1000},
    {"node": "node 3", "start_idx": 2000, "count": 1000}
  ],
  "realized_n_summary": {
    "p10": null, "p25": null, "p50": null,
    "p75": null, "p90": null, "p99": null,
    "mean": null, "stdev": null,
    "computed_at_consolidation": false
  }
}
```

The `realized_n_summary` block is populated by
`scripts/data/build_metadata.py` at §7.6
verification time, by reading all 2250 present-scene masks and
counting 2×2 footprints per scene at frame 0. If `grf_corr_length`
was overridden at the §5.3 visual gate, set
`grf_corr_length_overridden_at_visual_gate: true` and record the
final chosen value (the script reads this from the resolved YAML
that was used for generation, found via the per-node log records
captured by the generator).

### 7.6 Verification

After §7.3 consolidation, run:

1. **File count**: 3000 × 2 = 6000 `.npy` files (a_* and m_*),
   optionally 3000 `.gif` files.
2. **Filename label distribution**: 2250 files with `{label}=1`
   (`vid*_1.npy`), 750 with `{label}=0` (`vid*_0.npy`).
3. **Realized-N summary**: populate the `realized_n_summary` block
   in `dataset_metadata.json` and assert it matches §3.2.4 / §6.1
   expectations.
4. **Visual spot-check**: randomly sample 5 present and 2 absent
   scene GIFs; verify clusters and absent-emptiness visually.
5. **§6.6 regression**: re-run the backward-compat test against the
   current `data_generator.py` to confirm the §4.1 zero-code-change
   property holds.

---

## 8. Things deliberately out of scope

- **Architecture or training choices.** This spec is dataset-only.
  The paired training spec will decide arms, hyperparameters,
  supervision-grid resolution, and metric accumulator extensions.
- **Resolutions other than 128².** The argument for 128² is its
  status as the v0.6-success scale. A 256² version of this dataset
  would be a separate spec with separate rescaling derivations.
- **Object geometries other than 2×2 square.** Switching to diamonds
  or larger squares would move two variables from the v0.6 anchor
  and defeat the purpose of the diagnostic.
- **Velocity regimes other than slow (1/8 of v0).** Slow is the
  v0.6-success regime; varying velocity at 128² would be a different
  diagnostic.
- **Curved or stop-backtrack target trajectories.** v2 supports
  these via separate config; this spec uses the straight-line
  default (matching v0.6 / slow_swarm exactly).
- **`DetectionEpochAccumulator` multi-target extension.** Declared
  as a coupling dependency in §0.4; lives in the training spec, not
  the dataset spec.

---

## 9. Acceptance criteria for this spec

This spec is ready to merge when:

1. The YAML at `configs/data/multi_target_swarm_128.yaml`
   exists and validates against the generator's config schema.
2. Tests 6.1–6.6 are implemented in `tests/test_swarm_grf_128.py` and
   pass on a smoke-generation batch.
3. The §5.3 visual-tuning gate has been executed on a 60-scene
   inspection batch and the chosen `grf_corr_length` (default 0.75
   or override) is recorded.
4. A smoke generation of 200 scenes shows the §6.1 realized-N
   distribution matching §3.2.4 expectations within tolerance.
5. The §6.6 backward-compat regression passes (slow_swarm + v1
   chip_aware fixtures both byte-identical against the post-spec
   generator).

Full 3000-scene generation can proceed in parallel with §6.6 fixture
landing if §5.3 visual gate passes — the full generation does not
depend on §6.6 being committed, only on §4.1 being verified true.
