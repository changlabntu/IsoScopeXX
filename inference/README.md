# `inference/` — encode → codec → decode

Self-contained inference package for this repo's isotropic-SR + compression
models. It turns a trained run into two things:

- a **pure-tensor API** (`Engine`) that exposes the model's staged
  `generation_test` interface — encode a volume to a compact VQ latent,
  decode it back to an isotropic super-resolved volume, or reconstruct it at
  input resolution;
- **CLI scripts** that stream large stacks through that API: encode an
  OME-Zarr / TIFF stack to per-patch codec `.npz` files, and decode a codec
  grid back to viewable TIFFs or a native OME-Zarr store.

It is independent of `test/` (the scenario tooling) and of `models/` internals
beyond the `generation_test` / `latents_from_indices` contract. See the repo
`CLAUDE.md` for the model lineage and training side.

## The pipeline at a glance

```
OME-Zarr / TIFF stack                                  viewable output
        │                                                     ▲
        │  encode_stack.py                    decode_stack.py │
        ▼                                                     │
   codec/z{a}-{b}/{rrr}{ccc}.npz   ──────────────────────────┘
   (per-scale int32 indices + norm_params.json)
```

- **Encode** = normalize → VQ-encode each patch → store codebook indices.
- **Codec** = the compact stored form (`scale_*` int32 index arrays +
  `norm_params.json` with the intensity window and everything needed to
  invert). Compresses ~160–200× vs raw uint16.
- **Decode** = indices → latents → the isotropic SR volume (**decode**, Z×
  uprate) or the input-res VQ reconstruction (**reconstruction**), restored
  to source uint16 units.

The full copy-pasteable workflow lives in **`pipeline.sh`**.

## Environments

- **`py38zarr`** (= `py38pl16` clone + `tensorstore`): required for any zarr
  path (`encode_stack --zarr`, `decode_stack --zarr` / `--orig`).
- **`py38pl16`**: enough for the TIFF-only paths (`inference_latent.py`,
  `decode_stack --tiff` without `--orig`) and the pure `Engine` API.
- `latent_tsne.py` is pure CPU/sklearn — no model, no GPU.

## Companion docs

- **`pipeline.sh`** — copy-pasteable encode → compare workflow.
- **`DECODE_ROUNDTRIP.md`** — the encode/decode round-trip contract & invariants.
- **`edge_recipe.md`** — the decoder-uncertainty + 3D edge pipeline end to end:
  the `--tta`/`--mc` `std`+`mean` maps (Stage A: TTA mechanics, flag/mode
  semantics), then edge extraction (instant-alpha flood fill + dim-dot rescue)
  with the self-contained code.

---

## Library API

### `__init__.py`
Re-exports the public surface: `Engine`, `load_model`,
`resolve_checkpoint_dir`, `available_epochs`, `normalize`, `invert_gamma`,
`MODELS`, `available`, `get`, `register`.

### `engine.py` — `Engine`
Model-agnostic wrapper around the staged `generation_test` interface. All
volumes are `(B, C, Y, X, Z)`, already normalized, on the model's device
(the caller owns device/dtype).

| method | in → out | notes |
|---|---|---|
| `Engine.from_registry(name, epoch=, device=, train_mode=)` | → `Engine` | load a run named in `registry.py` |
| `Engine.from_checkpoint(path, epoch=, device=, model_file=, train_mode=)` | → `Engine` | load a run by checkpoint path |
| `.encode(x)` | volume → `(scale_latents, indices)` | per-scale latents + storable int64 index lists |
| `.decode(scale_latents)` | latents → `(B,1,Y',X',Z')` | isotropic SR volume |
| `.full(x)` | volume → SR volume | `encode` + `decode` in one call |
| `.reconstruction(scale_latents)` | latents → `(B,out_ch,Y,X,Z)` | slice-wise VQ-head recon, no Z upsampling |
| `.latents_from_indices(indices)` | stored indices → `scale_latents` | codec round-trip, no encoder run |
| `.normalize(vol, …)` / `.denormalize(vol, …)` | — | apply / invert the run's `nm` transform |
| `.device`, `.spec`, `.cfg` | — | device; effective `nm/gamma/gamma_lo`; run config |

**`train_mode` must ALWAYS be `True` in testing — never pass `False`.** It is
the default on both constructors, so omit it. The generator components run in
`.train()` (batch-stat BN + MC dropout — each pass is one MC draw); the codec
stays deterministic either way (the 2D VQ stack is GroupNorm, dropout 0).

The reason is not stylistic: these runs train with `--norm batch` at `-b 2`, so
the BN *running* statistics are accumulated from batches of 2 and are too poorly
conditioned to reproduce training-time behaviour. `train_mode=False` swaps in
those bad running stats and silently changes the output. Any figure or
comparison produced with `train_mode=False` does not follow the convention and
should be regenerated.

### `load.py` — checkpoint loading
Rebuilds a run's `GAN` from its checkpoint snapshot + `config.json` and swaps
in the pickled component modules found on disk.
- `resolve_checkpoint_dir(path)` — accept a timestamped dir, a `checkpoints/`
  parent, or an experiment root; pick the newest run with `.pth` files.
- `available_epochs(ckpt_dir)` — epochs where every component has a `.pth`.
- `load_model(ckpt_dir, epoch=, device=, model_file=) → (gan, args)` —
  `model_file` overrides the snapshot with a current `models/*.py` (same
  component names) for running refactored code against old weights.

### `normalize.py` — intensity transforms
The same math training applies, as pure numpy/torch functions.
- `normalize(vol, nm, gamma=, gamma_lo=, norm_stats=, key=)` — modes `00`
  (untouched), `01` `[0,1]`, `11` `[-1,1]`, `11p` percentile-clip, `11g`
  noise-floor + compressive gamma.
- `invert_gamma(vol, gamma, gamma_lo)` — exact inverse of `11g` back to the
  pre-gamma `[-1,1]` scale (used by `Engine.denormalize`).

### `registry.py` — named model specs
Pins checkpoint, `model_file`, epoch and normalization per model name.
- `available()` — registered names.
- `get(name)` — the spec dict (a copy).
- `register(name, **spec)` — add/replace at runtime.
- `MODELS` — the dict itself. Current entries: **`skipU`** (thx10 MSclean),
  **`sa635`** and **`sa635_g03`** (Chulab SA635 runs, per-run
  `config.json` normalization).

### `zarr_io.py` — zarr helpers (lazy `tensorstore` import)
- `open_zarr(store, level)` — open one pyramid level read-only.
- `create_zarr(path, shape, chunks, dtype='uint16', overwrite=False)` —
  create/open a blosc-zstd zarr-v3 array (opens existing by default so chunk
  runs fill one store incrementally).
- `write_ome_group(store, name, axes=('z','x','y'))` — write the group
  `zarr.json` making `{store}/0` a valid single-scale OME-NGFF 0.5 store.

---

## CLI scripts

### `encode_stack.py` — stack → codec
Global-percentile normalize + VQ-encode a stack, patch by patch. Writes
`{out_base}/{exp}/codec/z{a}-{b}/{rrr}{ccc}.npz` (per-scale int32 indices) +
a per-chunk `norm_params.json` (intensity window, `uprate`, and everything
needed to invert).

- **`--zarr`** source (all z-chunks in one run; needs `py38zarr`): streams
  from an OME-Zarr `(z,x,y)` store, one exact global intensity window over the
  whole `--range` via a one-pass uint16 histogram, split into `--zchunk`
  chunks. Nothing staged in RAM.
- **`--source`** source: a folder of large 2D TIFF slices (sorted = Z);
  one z-chunk per run, window estimated with strided `np.percentile`.

Key flags: `--model --exp` (required), `--range B E`, `--patch 256`,
`--lo_pct/--hi_pct`, `--window LO HI` (reuse a fixed window), `--half
--batch`, `--tsne` (also run `latent_tsne` per chunk). Helper functions:
`open_zarr` (re-exported from `zarr_io`), `zarr_window`, `encode_grid`,
`finish_chunk`.

### `decode_stack.py` — codec → TIFF / zarr
Inverse of `encode_stack`. Decodes stored codecs to **uint16 in source
units** (`denormalize` → window `lo/hi` → round/clip). `--what decode`
(isotropic SR, Z× uprate, ZX pages) or `reconstruction` (input-res VQ head,
XY pages).

Sinks (one or both):
- **`--zarr STORE`** — a native `(z,x,y)` uint16 OME-Zarr 0.5 store. By
  default sized to the whole volume and written sparsely (for a full decode);
  **`--crop`** sizes it to the ROI bounding box (origin at the ROI corner),
  fully populated — small and napari-friendly. `--overwrite` recreates.
- **`--tiff DIR`** — per-patch TIFFs in `DIR/{what}/`, plus a stitched strip
  for `--row`/`--col`.
- **`--orig`** — also emit the matched source (raw input-res for
  reconstruction; trilinear Z-upsampled for decode) into `DIR/original/`
  and/or a sibling `<zarr>_original` store, so an ROI enhancement and its
  interpolated original overlay 1:1. Zarr source only.

ROI selection (both sinks): `--chunk zBBBB-EEEE` and one of
`--cells R0 R1 C0 C1` (half-open block), `--row R`, `--col C` (default: every
cell of every chunk). Determinism: deterministic by default; `--mc` for one
MC draw per patch. Helpers: `to_xy_pages`, `to_zx_pages` (from
`inference_latent`), `chunk_z_range`, `resolve_chunks`, `select_cells`,
`infer_uprate`, `load_indices`, `restore_uint16`.

### `inference_latent.py` — per-file TIFF round trip
Encode each `{stem}.tif` in `--source` to `codec/{stem}.npz` and decode it
back, writing `decode/{stem}.tif` (isotropic, ZX pages) + `input/{stem}.tif`
(trilinear input at the same size) for side-by-side inspection. The simple
per-volume path; `decode_stack.py` is the grid/large-stack counterpart.
- `to_zx_pages(vol_yxz)` — `(Y,X,Z)` → ZX tif pages `(Y,Z,X)`.

Extra flags:
- **`--tta [THRESHOLD]` / `--mc N`** — decoder-uncertainty maps: decode the
  latent-TTA variants (× N MC draws) and also write `std/{stem}.tif`
  (per-voxel disagreement) and `mean/{stem}.tif` (consensus). Feeds the edge
  pipeline — see `edge_recipe.md` (TTA mechanics + edge extraction).
- **`--gamma / --gamma_lo`** — override the `11g` gamma for both the input
  transform and the output inversion (only valid for `nm='11g'` runs).
- **`--eval`** — deterministic running-stat decode instead of the MC default.

### `latent_tsne.py` — codec map + anomaly flagging
Pure CPU/sklearn. Turns each patch's codec into a per-scale code-usage
histogram, then PCA → t-SNE 2D scatter (`tsne.png`), IsolationForest anomaly
flags (`anomalies.csv`), an optional grid-position-colored figure
(`tsne_grid.png`), and with `--thumbs` a Z-MIP-thumbnail figure
(`tsne_thumbs.png`). Callable as `run(codec, out_dir=, …)` (used by
`encode_stack --tsne`) or via CLI. Helper: `load_features(codec_dir)`.

---

## Quick start

```python
from inference import Engine
eng = Engine.from_registry('skipU')            # loads the pinned checkpoint
x   = torch.from_numpy(eng.normalize(vol))[None, None].to(eng.device)  # vol (Y,X,Z) in [-1,1]
iso = eng.full(x)                              # (1,1,Y,X,Z*uprate) isotropic SR
```

For large stacks, use the CLI — see **`pipeline.sh`** for the encode →
compare workflow (encode a whole OME-Zarr, reconstruction/enhancement TIFF
strips, and cropped-ROI enhancement-vs-original zarr stores).

## Codec / geometry notes

- Volume tensor layout is `(B, C, Y, X, Z)`; TIFF pages read as `(Z,Y,X)`.
  In the codec grid, `rrr` = row = Y index, `ccc` = col = X index; the zarr
  store axes are `(z, x, y)` (x ← cols, y ← rows).
- `uprate` (the decode Z factor) is recorded in `norm_params.json` by encode;
  `decode_stack` reads it there, falling back to `(cropsize//cropz)*dsp/usp`
  from the run config.
- Intensities above the encode window `hi` (p99.9 by default) saturate at
  `window_hi` on restore — faithful to the encoded signal, not to outlier
  hot pixels.
