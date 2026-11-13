# Data: the synthetic swarm generator and the datasets

**What this is.** Every number in this project was measured on synthetic
data from `localizer.synth.swarm`. No dataset ships; this page tells you how
to regenerate each one and how to check you got the same bytes.

**State.** All four datasets regenerate from shipped configs: the three
multi-target sets (128², 512² v2, 1024² v2) and the 30k-scene set the
classifier and segmenter were trained on (`swarm_128x128x64`). The 512² v2
set and the 30k set have both been shown to regenerate bit-identically, on a
different CPU architecture from the one that generated them (below).

Pipeline status, for context: gate -> classifier is measured end to end on
512² synthetic data; the segmenter is validated standalone only (shipped
weights: epoch-19 checkpoint, val Dice 0.391), and its integration is open
work.

## Scene model

Each scene is a `(T, H, W)` = `(64, H, W)` uint8 video. A dense background
swarm of `n_circles` identical small squares (`shape: square`,
`square_edge: 2`, so 2 x 2 px) moves with one shared velocity drawn per
scene from `speeds` (pixels per frame) and a random direction. Targets are
the same 2 x 2 squares moving with a different velocity: a different
direction, or with `samedir: true` possibly the same direction at a forced
different speed. The truth mask marks target pixels in every frame (255 on,
0 off). Movers are clipped at the frame border (no wraparound).

Target placement (`target_sampler: grf_cluster`, the multi-target sets):

1. Scene `idx` is target-absent iff `idx % 4 == 3` (`grf_absent_modulo: 4`),
   so 25% of scenes are empty.
2. The scene is cut into chips of `chip_size` px. A per-scene fraction
   `u ~ Beta(2, 5)`, rescaled to `[1/10, 2/3]`, sets how many chips `k`
   hold targets. A Gaussian random field is drawn over the chip grid
   (squared-exponential kernel, correlation length `grf_corr_length` in
   chips) and the `k` chips with the highest field values are occupied, so
   occupied chips cluster.
3. Each occupied chip's target count maps its field value linearly onto
   `grf_count_range`, plus Gaussian noise (`grf_count_noise_sigma: 1.0`),
   rounded and clipped; positions are uniform inside the chip.

The older `single` sampler (one target on even indices, none on odd) is what
`swarm_128x128x64` used; it is still the code default. That set also differs
in the swarm: `shape: disk` with `radius: 2`, 10,000 members, speeds 0.5 /
1.0 / 2.0 px/frame.

Full design rationale: `docs/specs/multi_target_swarm_128_dataset_spec.md`
and `docs/specs/multi_target_swarm_1024_v2_dataset_spec.md` (scrubbed copies;
this page is authoritative where they differ). The 512² v2 set has no spec of
its own; its config was derived from the 1024² v2 design during the 512²
gate diagnostics and recorded at commit `60cc2bd` ("512² config audited
against scaling law").

## Seeds: how determinism works (read before regenerating)

The generator splits the index range into batches of `batch_size` scenes and
seeds each batch with `numpy.random.default_rng(start_idx_of_batch)`. So the
scenes depend on `batch_size`. **Regenerate with the config's `batch_size`**
(50 for 512² and 1024², 250 for 128²) or you get different, equally valid,
scenes. `num_workers` only sets parallelism and does not change output;
`--start_idx` / `--total_scenes` select a slice and must start on a batch
boundary to reproduce the canonical bytes.

`swarm_128x128x64` is the exception to "one run from index 0": it was
generated as six separate invocations, each cut into 250-scene batches from
its own `--start_idx` (so batch seeds such as 3334 and 6584 occur, and each
invocation ends in a short batch). Reproducing it needs those six
invocations (below); one run from 0 gives different, equally valid, scenes
at indices 3334-9999 and 16667-29999.

## The datasets

| Dataset (directory name) | Config | Scenes (present / absent) | Per-scene files | Total on disk | Original generation wall |
|---|---|---|---|---|---|
| `multi_target_swarm_128x128x64` | `configs/data/multi_target_swarm_128.yaml` | 3000 (2250 / 750) | 2 x 1,048,704 B | about 6 GiB (plus GIFs if enabled) | 504 s, 12 workers |
| `multi_target_swarm_512x512x64` (v2) | `configs/data/multi_target_swarm_512_v2.yaml` | 3000 (2250 / 750) | 2 x 16,777,344 B | about 94 GiB | about 6,865 s (inferred), 8 workers |
| `multi_target_swarm_1024x1024x64` (v2) | `configs/data/multi_target_swarm_1024_v2.yaml` | 3000 (2250 / 750) | 2 x 67,108,992 B | about 376 GiB | about 16,000 s per 1000-scene slice, 3 machines in parallel |
| `swarm_128x128x64` (classifier, segmenter) | `configs/data/swarm_128.yaml` | 30000 (15000 / 15000) | 2 x 1,048,704 B | about 59 GiB | 9,331 s (two phases, three machines x 20 workers) |

| Dataset | Key parameters | Measured realized content (300-scene sample) |
|---|---|---|
| 128² | speeds 0.0625 / 0.125 / 0.25 px/frame, 30,000 swarm members, 32 px chips, 1-3 targets per chip | 2-17 frame-0 target components per scene, mean 7.7 |
| 512² v2 | speeds 0.25 / 0.5 / 1.0, 480,000 swarm members, 128 px chips, 1-10 targets per chip | 8-51 frame-0 components, mean 20.3; 2-9 occupied 128 px chips, mean 4.0 |
| 1024² v2 | speeds 0.5 / 1.0 / 2.0, 1,920,000 swarm members, 128 px chips, 1-10 targets per chip, correlation length 1.5 chips | 28-155 frame-0 components, mean 77.4; 7-37 occupied chips, mean 17.3 |

Frame-0 component counts are a lower bound on the target count (adjacent
targets merge). On the 512² val split the system evaluation counts 4,143
3D target groups over 218 present scenes (6-45 per scene, mean 19.0).

<!-- Sources: scene model and sampler: src/localizer/synth/swarm.py
(_sample_grf_cluster_targets, _generate_video_grf_cluster); border clipping:
analysis/stage1_v0_10_percell/border_attribution.json meta.note_generator.
Per-file sizes: header + shape (64*H*W uint8 + 128 B .npy
header), confirmed by stat on the canonical copies; totals: originating repo
documents/notes/recon_2026_11_05_stage1_export_report.md section 4.1 (du -sh:
128² 6.0G, 512² 94G, 1024² 376G); 30k total: 2 x 1,048,704 B x 30000.
Present/absent, generation walls, realized counts: each dataset's
dataset_metadata.json (128²: wall_seconds_total 504, realized_n_summary mean
7.713 min 2 max 17; 512²: wall_seconds_total 6865 "inferred from scene
mtimes", realized_n_summary mean 20.257 min 8 max 51, realized_k_summary mean
4.043 min 2 max 9; 1024²: per-node wall 15911-16052 s,
per_scene_realized_target_counts mean 77.43 min 28 max 155,
per_scene_realized_k mean 17.28 min 7 max 37). 30k labels: classifier
manifest dataset field. 4,143 groups: originating run artifact
analysis/stage1_v0_10_percell/per_target_recall.json results.A
(total_targets 4143, per_scene_target_count min 6 max 45 mean 19.005). -->

## Regenerating

Set the data root once, then run the config as shipped. Output goes to
`$LSL_DATA_ROOT/<dataset>/{data_all,label_msk}/`.

```bash
export LSL_DATA_ROOT=/path/to/datasets
lsl-generate --config configs/data/multi_target_swarm_128.yaml        # about 6 GiB
lsl-generate --config configs/data/multi_target_swarm_512_v2.yaml     # about 94 GiB
lsl-generate --config configs/data/multi_target_swarm_1024_v2.yaml    # about 376 GiB

# swarm_128x128x64 (about 59 GiB): six invocations, as originally generated
for s in "0 3334" "3334 3333" "6667 3333" "10000 6667" "16667 6667" "23334 6666"; do
  set -- $s
  lsl-generate --config configs/data/swarm_128.yaml --start_idx $1 --total_scenes $2
done
```

The 30k set is slow on one machine: about 10.6 s per scene per worker
process (measured with one worker on an x86-64 laptop CPU), so about 88
CPU-hours in total; `num_workers` in the config is 20.

Split a long generation across machines by batch-aligned slices (each
machine writes its own index range into the same layout; merge the
directories afterwards):

```bash
lsl-generate --config configs/data/multi_target_swarm_1024_v2.yaml --start_idx 0    --total_scenes 1000
lsl-generate --config configs/data/multi_target_swarm_1024_v2.yaml --start_idx 1000 --total_scenes 1000
lsl-generate --config configs/data/multi_target_swarm_1024_v2.yaml --start_idx 2000 --total_scenes 1000
```

Then write the metadata record beside the data (realized-content statistics,
spot hashes):

```bash
python scripts/data/build_metadata.py --config configs/data/multi_target_swarm_512_v2.yaml \
    --data_root $LSL_DATA_ROOT/multi_target_swarm_512x512x64 \
    --dataset_name multi_target_swarm_512x512x64 --dataset_version v2 \
    --config_commit <commit> --generator_commit <commit>
```

## Checking you have the same data

sha256 of the first file of each set as originally generated (full hash where
recorded, otherwise the first 16 hex characters):

| Dataset | File | sha256 |
|---|---|---|
| 512² v2 | `data_all/vid0_1.npy` | `df49f98a56ff4b73056fa1d8f8f187cc245531c70d749c254d4f1d77853a04c5` |
| 512² v2 | `data_all/vid1500_1.npy` | `409d84c706084fc8483654219d5fdc42e19bacc7b3cda42270f587b5d159cd31` |
| 512² v2 | `data_all/vid2999_0.npy` | `f6bf39618d3907d73c358c32e5b4d93d04a388dc9c320f837cd5d1532c3583b3` |
| 512² v2 | `label_msk/vid0_1_mask.npy` | `d0ac7f143c8abc49158aeb5623722423e70bb8bcb938cda788adbec56cd01250` |
| 128² | `data_all/vid5_1.npy` | `13a2798fb088dc0a...` |
| `swarm_128x128x64` | `data_all/vid0_1.npy` | `6441d21127f253429383d1464ad5a2fed095a738d554eb846517c09053034c43` |
| `swarm_128x128x64` | `data_all/vid3334_1.npy` | `d1e39e57e073fd083ef76ad636d935cc0c3eb938304f41cf97dd4a14ca7e160a` |
| 1024² v2 | `data_all/vid5_1.npy` | `bc66936d3aaab943...` |

```bash
sha256sum $LSL_DATA_ROOT/multi_target_swarm_512x512x64/data_all/vid0_1.npy
```

**Bit-exactness, measured.** Batch 0 of the 512² v2 set (indices 0-49, seed
0) was regenerated on Linux x86-64 with numpy 2.4.6 and compared byte for
byte against the original, which was generated on aarch64: all 100 files
(50 videos, 50 masks) identical. This is a measured result for that platform
pair and numpy version, not a guarantee for other numpy versions; if your
hashes differ, compare distribution statistics (`build_metadata.py` output)
before assuming a bug.

The same holds for `swarm_128x128x64`: its first batch (indices 0-249, seed
0), regenerated on x86-64 with numpy 2.4.6 by the current generator with
`configs/data/swarm_128.yaml`, matched the original aarch64 bytes in all 500
files, as did two-scene probes at the start of each of the other five
invocations (3334, 6667, 16667, 23334) and at the short last batch (29834).

<!-- Sources: 512² hashes: $LSL_DATA_ROOT/multi_target_swarm_512x512x64/
dataset_metadata.json spot_hashes_sha256; 128² and 1024² prefixes: recon
report sections 4.4 and 4.5 ("vid5_1.npy is 13a2798fb088dc0a",
"vid5_1.npy is bc66936d3aaab943"); bit-exactness: originating repo
documents/notes/validation_2026_11_05_export.md section 6 ("identical=100
differ=0 missing=0", numpy 2.4.6); swarm_128x128x64: its dataset_metadata.json
(spot_hashes_sha256, per_partition_layout, wall_seconds_total 9331) and the
originating repo's validation record documents/notes/validation_2026_11_05_export.md section 9c
("identical=500 differ=0 missing=0", real 44m8s for 250 scenes, one worker). -->

## The classifier / segmenter training set (`swarm_128x128x64`)

30,000 whole 128² scenes from the `single` sampler (one anomalous mover on
target scenes), 15,000 / 15,000; config `configs/data/swarm_128.yaml`,
regenerated bit-identically as described above. Its generator config had been
retired in the originating repository and was recovered from that
repository's history; the generation parameters are verbatim.

A second copy generated in the originating environment from the same config
and indices differs from the canonical one at about two thirds of the indices.
That copy was generated as one run from index 0, so its batch starts (and
seeds) differ from the canonical six-invocation layout exactly where the two
grids diverge (indices 3334-9999 and 16667-29999); it is a different, equally
valid draw. Scoring the classifier on it moves the confusion counts by a few
samples (`docs/classifier.md`). Check `vid3334_1.npy` against the hash table
before relying on a copy.

A deployment-style training set (128² crops of 512² scenes) is the
recommended next training set for the classifier; see `docs/limitations.md`.

## Smoke sets

For checking that code runs, not for measuring anything:

```bash
# 40 scenes at 128², about 1 minute: used for the 1-epoch smoke training of all three models
lsl-generate --config configs/data/multi_target_swarm_128.yaml --output_root $LSL_DATA_ROOT/smoke_128 \
    --total_scenes 40 --batch_size 20 --num_workers 2 --save_gifs false

# 10 scenes at 512²: enough to run the pipeline evaluation end to end
lsl-generate --config configs/data/multi_target_swarm_512_v2.yaml --output_root $LSL_DATA_ROOT/smoke_512 \
    --total_scenes 10 --batch_size 2 --num_workers 5
```

The test suite builds its own tiny fixtures (`tests/data/` and `tmp_path`)
and needs no generated data.

## Known to be wrong or missing

- Bit-exactness is measured for one numpy version and two CPU
  architectures only.
- The 512² v2 generation commit and wall time are inferred from file
  timestamps; no generation log was kept.
- All data is synthetic and abstract: no sensor model, no clouds, parallax or
  jitter. Nothing has been measured on real imagery.
