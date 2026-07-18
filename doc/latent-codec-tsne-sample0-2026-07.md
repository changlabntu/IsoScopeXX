# Large-TIFF → codec encoding + latent t-SNE: sample0 results (2026-07-18)

First end-to-end run of the `inference/` codec pipeline on a full THX10 volume. Encodes a
large 2D-slice stack to VQ codebook indices patch-by-patch (no decoding), then maps the
per-patch code-usage histograms with t-SNE and flags outliers. Code committed in `e3663c1`
(inference package) on top of `830eb17` (vqclean staged `generation_test`).

## Pipeline

Three stages, all under `inference/` (a pure-tensor API, independent of `test/`):

1. **`encode_stack.py`** — loads a folder of large 2D `.tif` slices (sorted name = Z order)
   into RAM, tiles each into a `patch×patch×Z` grid, maps intensities through one **global
   robust percentile window** shared across all patches/chunks, runs each `(patch, patch, Z)`
   volume through the registered model's encoder, and writes per-scale codebook indices as
   `{rrr}{ccc}.npz` under `{out_base}/{exp}/codec/z{begin}-{end}/`. Codec only by default;
   `--tsne` chains stage 3 over the chunk.
2. **`registry.py` / `engine.py` / `load.py`** — the `skipU` registry entry pins the
   checkpoint (`vqcleanM0aMSskipE` recipe, `netG ed023emsfpnu`, epoch 300), `model_file`
   (`models/MSclean.py`), and normalization (`nm='11g'`, γ0.7 / floor −0.8). `Engine.encode()`
   returns the per-scale index tensors.
3. **`latent_tsne.py`** — per patch, builds an L1-normalized per-scale histogram of codebook
   usage, concatenates over scales → feature vector; PCA → t-SNE → 2D; IsolationForest flags
   outliers. Emits `tsne.png`, grid-colored `tsne_grid.png`, `anomalies.csv`, and (with
   `--thumbs`) a Z-MIP-thumbnail figure `tsne_thumbs.png`. Pure CPU/sklearn.

## Data: THX10 sample0

Slices Z0640–Z1280 (161 slices) cropped from the main THX10 folder with the standard rule
(memory `thx10-sample0-crop-params`):

| property | value |
|---|---|
| source | `/media/cheese/Ghc_data3/THX10/sample0_crop` |
| cropped slice size | 8960 × 14592 |
| patch | 256 |
| patch grid | 35 rows × 57 cols = 1995 patches / chunk |
| chunks | 5 × 32 slices → `z0000-0032` … `z0128-0160` |
| shared window | `[99, 495]` (p1 / p99.9, estimated on chunk 0, reused via `--window`) |
| scales per codec | 4 |

Run driver: `inference/encode_stack.py --model skipU --exp sample0 --source <sample0_crop>
--half --batch 8`, first chunk estimates the window, later chunks pass it back.

## Results

**Encoding.** 9,975 codecs total (5 chunks × 1995), ~11.6 patches/s on fp16 (~172 s/chunk).

| metric | value |
|---|---|
| codec size / chunk | ~42–54 MiB (grows with Z depth into tissue) |
| raw size / chunk | 7.8 GiB |
| compression | **~156×** (indices only, no decode) |

**t-SNE (combined, 9,975 patches, feature dim 1014, 5% contamination → 499 anomalies).**
The embedding separates by **content, not spatial position**:

- The map has two lobes. One is **pure background / empty patches** (grainy static in the
  Z-MIP thumbnails); the other is **fibrous tissue structure**, with a clean gradient between
  them. A small detached island of near-empty patches sits off to one side.
- `tsne_grid.png` colors the same embedding by row / col / z: colors are **well-mixed within
  clusters**, confirming patches group by what they contain rather than where they sit in the
  volume — the intended outcome.
- Anomaly flags land on two populations: the **densest / brightest tissue** patches and the
  **tissue↔background boundary**. So "anomaly" currently means *unusual code histogram*, which
  conflates genuinely interesting structure with mundane bright/empty edges.

Artifacts live under `/home/cheese/workspace/Output/sample0/` (`tsne.png`, `tsne_grid.png`,
`anomalies.csv`) and per-chunk under `codec/z*/`. A single-chunk thumbnail figure was also
produced for `sample0a` (`tsne_thumbs.png`).

## Known limitations / next steps

- **Multi-chunk t-SNE is not first-class.** `latent_tsne.load_features` globs `codec_dir/*.npz`
  (flat), but the combined 9,975-patch run needs `codec/*/*.npz` across chunk subdirs — the
  combined plot was produced by an ad-hoc recursive glob, not the committed script.
  `encode_stack --tsne` only analyzes its own single chunk.
- **Anomaly semantics.** Add a **foreground-fraction filter** before IsolationForest (or as a
  post-filter) if structure-only flags are wanted, so bright/empty edge patches stop dominating.
- **Codec storage format is undecided.** Codecs are flat `.npz` per chunk today; a companion
  Zarr grid keyed by (row, col, z) — and possibly an OME-Zarr raw converter + napari overlay of
  the anomaly/cluster map — is the open direction, along with where the zarr/napari deps live.
