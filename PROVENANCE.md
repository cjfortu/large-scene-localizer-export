# Provenance

This directory is a curated snapshot of the source repository at one tagged
commit. **It ships no trained weights.** `checkpoints/` holds only the three
`<name>.json` manifests: they describe the weights every published number was
measured with, and their `val_metrics` are the targets for a retrain
(`README.md`, "No weights in this package: day one"). The snapshot contains no
weight files, no binary wheels, no pickles and no `.git` directory.

| Item | Value |
|---|---|
| Source repository | `large-scene-localizer` |
| Tag | `export-v5-2026-11-13` (annotated) |
| Commit | `3a9d21c61e1faa8e55d0f1c000ba05d6581eb0fd` |
| Commit date | 2026-11-13T12:21:33-05:00 |
| Export date | 2026-11-13 |
| Builder host architecture | x86_64, Linux |
| Removed from the tree | documents/, scripts/home/, export/, CLAUDE.md, .github/, wheels/ (home-only material) |
| Package | `large-scene-localizer-export-v5-2026-11-13.zip`: a deterministic zip (entries sorted by name, every timestamp the commit date, deflate, every file mode 644, no directory entries, no symlinks, no extra fields); its sha256 is in `large-scene-localizer-export-v5-2026-11-13.zip.sha256` |

## Checking the download

Linux:

```bash
sha256sum -c large-scene-localizer-export-v5-2026-11-13.zip.sha256
```

Windows PowerShell (prints `True` when the hash matches the `.sha256` file):

```powershell
(Get-FileHash large-scene-localizer-export-v5-2026-11-13.zip -Algorithm SHA256).Hash -eq (Get-Content large-scene-localizer-export-v5-2026-11-13.zip.sha256).Split(' ')[0]
```

or print the hash and compare it by eye with the first field of
`large-scene-localizer-export-v5-2026-11-13.zip.sha256` (case does not matter):

```powershell
(Get-FileHash large-scene-localizer-export-v5-2026-11-13.zip -Algorithm SHA256).Hash
```

Windows `cmd`:

```bat
certutil -hashfile large-scene-localizer-export-v5-2026-11-13.zip SHA256
```

Extract with `unzip -q large-scene-localizer-export-v5-2026-11-13.zip` (or `python3 -m zipfile -e large-scene-localizer-export-v5-2026-11-13.zip .`) on Linux,
`Expand-Archive large-scene-localizer-export-v5-2026-11-13.zip -DestinationPath .` in PowerShell. Each gives the
directory `large-scene-localizer-export-v5-2026-11-13/`; then restore the names (`README.md`, "Restoring the
transferred tree").

## Git tree hashes

Two git tree hashes identify the content independently of the manifests. Each
covers every file of the tree except `PROVENANCE.md`, `MANIFEST.sha256` and
`MANIFEST.transfer.sha256`: this file records the hashes and both manifests
hash this file, so none of the three can be inside the hashed set (the
restore script checks all three through the manifests).

| Tree | git tree hash (SHA-1 object format) |
|---|---|
| As transferred: names ending in `.txt`, before the restore | `0605ce120b8dc6c0c6a803994f6867b1853be9f5` |
| Restored: original names, after the restore, with the copied `restore_names.sh` / `restore_names.ps1` deleted | `5a7212561ddc4e0eef9a557a4562b12858d27e1a` |

Check them from inside `large-scene-localizer-export-v5-2026-11-13/` with git. The four commands are the same in
bash, PowerShell and `cmd`:

```
git init -q
git add -A -f
git rm -q --cached PROVENANCE.md MANIFEST.sha256 MANIFEST.transfer.sha256
git write-tree
```

1. Right after extracting, run the four commands: `git write-tree` must print
   `0605ce120b8dc6c0c6a803994f6867b1853be9f5`. Then delete the `.git` directory they created (`rm -rf .git`;
   PowerShell: `Remove-Item -Recurse -Force .git`).
2. Restore the names (`README.md`), then delete the copied script
   (`rm restore_names.sh`; PowerShell: `Remove-Item restore_names.ps1`).
3. Run the four commands again: `git write-tree` must print `5a7212561ddc4e0eef9a557a4562b12858d27e1a`.
   Delete `.git` again.

Notes:
- `-f` is needed: `.gitignore` lists `*.npy`, which would leave out the four
  `.npy` test fixtures under `tests/data/`.
- `.gitattributes` (`* -text`) stops git from converting line endings, so the
  hashes are the same on Windows. A new repository on Windows has
  `core.filemode false`, so every file is recorded as 100644, as on Linux,
  where no file has an executable bit.
- If your git creates SHA-256 repositories by default
  (`init.defaultObjectFormat`), use `git init -q --object-format=sha1`.
- A hint about the default branch name from `git init` does not matter.

## Checkpoints (manifests only; no weights)

| Manifest | Weight file it describes (not in this package) | Bytes | safetensors sha256 |
|---|---|---|---|
| `checkpoints/classifier_baseline_30k_ep64.json` | `classifier_baseline_30k_ep64.safetensors` | 69099252 | `2f9c96d643b6321465cc7bd92b19c2d6e9d98196ef32a742e9b0147c362a989f` |
| `checkpoints/gate_A_stem2_bs1_ep42.json` | `gate_A_stem2_bs1_ep42.safetensors` | 14337404 | `eb7932b53ec34ed931db94eddbd2cade5dfa0490d825a49538cd3ad826019a3e` |
| `checkpoints/segmenter_volseg_ep19.json` | `segmenter_volseg_ep19.safetensors` | 73659692 | `104d3543e47684375a3463cde1874cf9dfd025323482efcb9b78aebf34034578` |

If the weight files arrive separately, put them in `checkpoints/` (or the
directory `LSL_CKPT_ROOT` names) under these names and check them against the
manifests (`docs/runbook.md`, step 3). They are outside both manifests and
both tree hashes, so adding them changes neither check.

## Tier 1 compiled extensions

Not included. `mamba-ssm` and `causal-conv1d` are source pins in
`requirements/cuda-linux-x86_64.txt` (PyPI sdist sha256 values) and are
compiled on the target machine; see `docs/environment.md`, "Tier 1 from source".

## Transfer form (no executable file extensions)

Every script and executable-type file (`*.py`, `*.sh`, `*.ps1`, `Dockerfile`,
`.dockerignore`, any file with a shebang) ships with `.txt` appended, and no
file has an executable mode bit. `TRANSFER_RENAMES.txt` lists each
`<original><TAB><renamed>` pair. Restore the names before anything else:
`restore_names.sh.txt` (Linux, POSIX sh) or `restore_names.ps1.txt`
(Windows PowerShell 5.1 or later); see `README.md`, "Restoring the transferred
tree". `.gitattributes` (`* -text`) keeps git from changing line endings in a
clone, so the hashes hold on Windows too.

## Integrity

- `large-scene-localizer-export-v5-2026-11-13.zip.sha256`: the sha256 of the zip (above).
- `MANIFEST.transfer.sha256`: the sha256 of every file of the tree as
  transferred (renamed names), including `TRANSFER_RENAMES.txt`, both
  restore scripts and this file. The restore scripts check it first.
- `MANIFEST.sha256`: the sha256 of every file of the restored tree
  (original names) except the transfer files (the two manifests,
  `TRANSFER_RENAMES.txt` and the restore scripts). The restore scripts check
  it last; by hand: `sha256sum --check --quiet MANIFEST.sha256`.
- The two git tree hashes (above).
