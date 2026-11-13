# Concept document (original design; the shipped system has diverged)

> **Read this first.** `stage1_concept.pdf` (March 2026) is the *original*
> design for Stage 1 and the figures here are from it. It is kept because its
> problem formulation (small, single-pixel to subpixel movers in large
> overhead video, where no single frame is enough) still holds. Its
> architecture does **not** describe the code: the shipped gate is a CNN +
> Mamba hybrid with a single temporal Mamba block and a low-resolution
> (2, 4, 4) logit head, not a full-resolution bidirectional-Mamba backbone
> with motion-direction heads; proposals are a fixed 4 x 4 grid of 128 px
> chips, not motion-filtered connected components; and the memory figures
> are 1024² estimates, not measurements. For the system as built, read
> `docs/architecture.md`; for what was measured, `docs/results.md`.

| File | Content |
|---|---|
| `stage1_concept.pdf` | The design document |
| `fig_4_1_pipeline_overview.png` ... `fig_4_7_output_heads.png` | Section 4 figures (pipeline, preprocessing, projection, Mamba backbone, temporal aggregation, spatial mixer, output heads) |
| `01_parallel_scan_tree.png`, `02_mamba_block.png`, `03_bidirectional_fusion.png`, `04_se_block.png` | Background figures (parallel scan, Mamba block, bidirectional fusion, squeeze-excitation) |

Pipeline status, for context: gate -> classifier is measured end to end on
512² synthetic data; the segmenter is validated standalone only (shipped
weights: epoch-19 checkpoint, val Dice 0.391), and its integration is open
work.
