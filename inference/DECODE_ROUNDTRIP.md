# Zarr-native encode → codec → decode round trip

**Status (2026-07-18): implemented and verified.** `inference/decode_stack.py`
+ `inference/zarr_io.py` exist; all five verification steps below passed on
the sample0 `z0000-0032` codec (skipU epoch 300, eval mode). The zarr and
TIFF paths are bit-identical per cell; recon correlates 0.84 with the raw
source (0.01 when Y/X-swapped — axes proven); incremental store fill works.
Decode needs `--batch 2` on a 24 GB GPU (batch 4 OOMs); throughput ~1.2
patch/s decode, ~4.6 patch/s reconstruction.

## Context

OME-Zarr (z, x, y) is the native input format for large stacks.
`inference/encode_stack.py --zarr` already covers **zarr → codec** (level 0,
streamed, validated bit-identical to the TIFF path): each patch →
`codec/z{za}-{zb}/{rrr}{ccc}.npz` with `scale_0..scale_{K-1}` int32 index
arrays, and a per-chunk `norm_params.json` holding the model spec +
`window_lo/window_hi`.

The missing half is the **decode side**: turn a codec grid back into (a) an
isotropic super-resolved volume and (b) a VQ-head reconstruction, usable both
as viewable TIFF comparisons and as a native zarr data store. This plan adds a
concise decode counterpart that reuses the existing `inference/` Engine stack.

## Confirmed decisions

1. **New `inference/decode_stack.py`**, mirroring `encode_stack.py` (not folded
   into `inference_latent.py`, which stays the per-file TIFF round-trip).
2. Encode stays **level 0 only** (already the `--zarr` default) — no change.
3. Output dtype **unified**: TIFFs **and** zarr store are both **uint16
   restored to source units** via `window_lo/hi` — one shared restore path,
   directly comparable to the raw data (ImageJ/Fiji read uint16 natively).
4. TIFF output = **per-patch files + optional stitched row/col strip**
   (comparison subset; full decode is ~334 GB and only goes to zarr).
5. **Keep** `engine/load/normalize/registry` where they are (no `utilities/`).

## Geometry reference (skipU, patch 256, uprate 8)

Grid for sample0: rows = 35 (Y), cols = 57 (X), z-chunks = 5 (each 32 slices).
Per codec cell `{rrr}{ccc}` in chunk `z{za}-{zb}` (`za = k*32`):

| output | per-patch shape (Y,X,Z) | full store (native z,x,y) | placement of patch (r,c,k) |
|---|---|---|---|
| **decode** (isotropic SR, Z×8) | (256, 256, 256) | (1280, 14592, 8960) ≈334 GB | z `za*8 : za*8+(zb-za)*8`, x `c*256:+256`, y `r*256:+256` |
| **reconstruction** (input res) | (256, 256, 32) | (160, 14592, 8960) ≈42 GB, matches source | z `za:zb`, x `c*256:+256`, y `r*256:+256` |

Z placement derives from the chunk dir name (`za`/`zb`), NOT a chunk counter —
so a short last chunk (tolerated by encode) still lands correctly.

- **Index → volume**: load `scale_k` int32 → int64 tensors on device →
  `eng.latents_from_indices(indices)` → `eng.decode(sl)` or
  `eng.reconstruction(sl)`. `sl` is the same list for both.
- **Model/window source**: read the chunk's `norm_params.json` (model, epoch,
  nm/gamma/gamma_lo, window_lo/hi). `Engine.from_registry(model)` reloads the
  same run; `--model` overrides.
- **uint16 restore** (the single output path, TIFF and zarr alike):
  `w = eng.denormalize(out)` (gamma → pre-gamma [-1,1]), then
  `raw = ((w+1)/2)*(hi-lo)+lo`, round, clip [0,65535], `uint16`.
- **Views**: decode → **zx** pages `transpose(0,2,1)` = (Y,Z,X) (reuse the
  `to_zx_pages` idiom from `inference_latent.py:47`); recon → **xy** pages
  `transpose(2,0,1)` = (Z,Y,X) (new `to_xy_pages`).
- **Write-back transpose** (Y,X,Z) → native (z,x,y): `transpose(2,1,0)`; this is
  the exact inverse of `encode_stack.py`'s read transpose, so the round-trip
  stays axis-consistent.

## Implementation

**New `inference/zarr_io.py`** (tiny shared helper, not part of the core API):
- `open_zarr(store, level)` — moved from `encode_stack.py` (that file imports it
  back, its only change).
- `create_zarr(path, shape, chunks, dtype)` — create a zarr-v3 array via
  tensorstore (`{'driver':'zarr3', 'kvstore':{'driver':'file',...},
  'create':True, 'delete_existing':True, metadata:{shape,chunk,dtype}}`).
- `write_ome_group(path, shape, axes=('z','x','y'))` — write the group
  `zarr.json` with a single-scale `multiscales` (dataset `0`) so the output is a
  valid native OME-Zarr, matching the input store's style.

**New `inference/decode_stack.py`** (CLI, mirrors `encode_stack.py` structure):
- Args: `--codec <dir>` (a `codec/` tree or one `z...` chunk dir), `--exp`/
  `--out_base` for outputs, `--what {decode,reconstruction}`, `--model`
  (default: from `norm_params.json`), `--epoch`/`--device`/`--half`, `--eval`
  (default **deterministic** for a data store; `--mc` for a stochastic draw),
  `--batch` (small: decode ~64 MB/patch on GPU).
- Zarr output: `--zarr <path>` — pre-allocate one store at the full shape from
  the table, chunk `(256,256,256)` for decode / `(32,256,256)` for recon
  (write-aligned), loop every chunk dir × cell, decode/recon → uint16 → write
  the placed slab. Write `write_ome_group` + a `decode_params.json`.
- TIFF output: `--tiff <dir>` with subset selection `--chunk z0000-0032` and
  one of `--cells r0 r1 c0 c1` / `--row R` / `--col C`; write per-patch uint16
  TIFFs (decode=zx, recon=xy; same restore as the zarr path) and, for
  `--row`/`--col`, a stitched strip (`np.concatenate` along the in-plane axis,
  as `test/inference.py:507` does).
- Reuse: `Engine`, `latents_from_indices`, `decode`, `reconstruction`,
  `denormalize`; `to_zx_pages` idiom. No change to engine/load/normalize/
  registry, `inference_latent.py`, or `models/`.

## Files

- **New**: `inference/decode_stack.py`, `inference/zarr_io.py`.
- **Edit**: `inference/encode_stack.py` — import `open_zarr` from `zarr_io`
  (move its body there); record `uprate` in `norm_params.json` so decode gets
  the Z factor from the codec itself (fallback for existing codecs: derive it
  from the run's config as `(cropsize//cropz)*dsp/usp`).
- **Unchanged**: `engine.py`, `load.py`, `normalize.py`, `registry.py`,
  `__init__.py`, `inference_latent.py`, `latent_tsne.py`, `models/`.

## Verification

Env `py38zarr`; codec already on disk at
`Output/sample0zarrval/codec/z0000-0032` (1995 patches).

1. **TIFF, decode zx strip**: `--codec Output/sample0zarrval/codec --tiff … --what
   decode --chunk z0000-0032 --row 17` → uint16 zx strip; confirm shape
   (Y=256 pages, Z=256, X=57*256) and that Z reads smooth/isotropic vs the
   blocky `input/` view.
2. **TIFF, recon xy strip**: same with `--what reconstruction` → uint16 xy strip
   (32 pages, Y=256, X=14592); eyeball against the source slices — same units,
   so a direct difference/overlay works.
3. **Zarr, small block**: `--zarr /tmp/…/dec_test.zarr --what decode --cells 16
   19 20 24` (a 3×4 block, one z-chunk) → read back with `zarrconv`/tensorstore:
   assert dtype uint16, the block region non-zero and in the source window
   range, the rest fill_value, and the group `zarr.json` parses as OME 0.5.
4. **Numeric sanity**: one patch — decoded (256,256,256), recon (256,256,32);
   both uint16 within the source window range.
5. Note: the **full** decode zarr (~334 GB, all 5 chunks) is large and slow;
   verification uses subsets only. A `--range`/chunk-select path allows running
   the full store deliberately.
