# test/ — checkpoint loading, the inference engine, and testing scenarios

All scripts must be run **from the repo root** (they read `ldm/*.yaml` and
`cfg/env.json` by relative path; `test/load.py` chdirs there defensively).

Layout: **`inference.py` is the single inference engine** (`--mode 3d/2d/row`);
the **`.sh` files are testing scenarios** driving it over the current model
bundle; the remaining `.py` files are support (`load.py` checkpoint loading,
`concat_views.py` view concats, `perceptual.py` offline metrics, `smoke.py`
sanity check).

## `test/load.py` — load any experiment's checkpoint

Rebuilds the `GAN` from the run's own snapshot (`{models}.py` + `config.json`
inside the checkpoint dir) and swaps in the pickled component modules
(`{name}_model_epoch_{N}.pth`). Component names are discovered from the
filenames, so it works across the whole lineage (`quant_conv/quantize` era as
well as `quantizers/quant_convs/post_quant_convs/inject_convs`).

```python
from test.load import load_model
gan, cfg = load_model('$LOGS/thx10-071226/vqcleanM0aMSskipU/roiD192gf/max5skip4')
# path may be the experiment root, .../checkpoints, or a timestamped run dir;
# epoch defaults to the latest complete one.
```

`model_file=` (CLI: `--model_file` on `inference.py`) builds the GAN from a
given models/*.py INSTEAD of the run's snapshot — for validating a refactored
model file against weights trained with the original code (the class must keep
the run's component module names). E.g. A/B the same checkpoint with and
without `--model_file models/MSclean.py` under `--eval` and
diff the output tifs; expect the difference to be within GPU run-to-run noise
(~7e-4 for the 384³ skipE output — measure it by rerunning one arm).

## `test/inference.py` — the inference engine

```bash
python test/inference.py --mode 3d|2d|row --checkpoint <run dir> --source ... [flags]
```

- **`--mode 3d`** (default): per-file isotropic inference, no stitching. Each
  input `.tif` (page order `Z,Y,X`) is one patch: normalize per the run's
  `--nm`, run `generation()`, save the isotropic `XupX` as float32 `.tif`.
  `--destination` is the output BASE (default `DEFAULT_OUT`); volumes go to
  `{destination}/output_3d/{tag}` (`--tag` defaults to the experiment name).
  `--limit/--skip` subset a directory source; `--save_input` writes the
  trilinear-upsampled input to the shared `{destination}/input/` dir (model
  dirs hold outputs only); `--save_zx` saves ZX page order (page y=k, rows Z),
  putting the synthesized axis in-plane.
- **`--mode 2d`**: slice-wise 2D VQ-head reconstruction (encoder → quantizer →
  decoder only, no 3D trunk, no Z upsampling) of one volume → `--out`
  (default `{DEFAULT_OUT}/summary_2d/{experiment}.tif`). Isolates what survives
  the VQ bottleneck. Ignores `--tta/--mc/--std_trd` (the 2D head is
  deterministic — GroupNorm, dropout 0).
- **`--mode row`**: a directory of adjacent `thAAABBBCCC.tif` patches forming
  a contiguous run on exactly one index (AAA=Y, BBB=X, CCC=Z) → per-patch
  outputs in `{source}/{tag}/` plus the concatenated strip `{source}/{tag}.tif`
  (XY page order always, so strips overlay the source's `original.tif`).

Shared behavior (all modes):

- Generator components run in `.train()` by default (MC dropout: batch-stat BN
  + active dropout, stochastic per run); pass `--eval` for deterministic
  running-stat inference. The `GAN` module itself always stays eval so
  `generation()` skips the training-time `cropz` crop.
- Normalization mirrors training: the run's `--nm` (override with `--nm`);
  `'11p'` needs `--norm_stats`; `'11g'` applies the config's
  `--gamma/--gamma_lo` and by default **inverts** outputs back to the
  pre-gamma [-1, 1] scale (`--no_invert` keeps gamma space;
  `--gamma_dec/--gamma_lo_dec` re-tone the decode inversion only).
- `--dsp 1` feeds a real anisotropic volume as-is; the default (config's
  `dsp`) mimics training by Z-subsampling an isotropic input first.
  `--cropsize/--cropz` center-crop before inference (3d mode).

Pass pooling (3d and row modes):

- `--tta` averages the original pass with an XY-transposed pass (output
  transposed back); `--mc N` repeats N times, so `--tta --mc 5` pools 10 runs.
  All passes are averaged in model-output space before any gamma inversion.
  In the default train mode every pass is an independent MC-dropout draw.
  A steady-state pass costs ~2 s per 384³ output (+~1 s one-time warmup).
- With more than one pass, the across-run population std is saved to
  `{destination}/output_std/{tag}/{stem}_std.tif` (row mode: the
  `{tag}_std.tif` strip) — model-output space, uninverted.
- `--std_trd <v>` binarizes every pass at `v` — given in the saved output
  scale, mapped through the forward gamma internally — and saves the mask's
  across-pass std `sqrt(p(1-p))` (p = fraction of passes calling the voxel
  foreground) as `{destination}/output_std/{tag}/{stem}_maskstd.tif` (row:
  `{tag}_maskstd.tif`):
  zero where all passes agree, 0.5 at a 50/50 split, so the nonzero band
  traces the uncertain foreground boundary.

## Scenarios (`.sh`, thx10-071226 bundle: skipE / skipU / skipUB)

| script | what it sweeps | output |
|---|---|---|
| `inference3d.sh` | mean isotropic outputs, first NVOL val volumes, `--tta` | `out/output_3d/{tag}/`, concats in `out/summary_3d/` |
| `inferencestd.sh` | boundary uncertainty, first val volume, `--tta --mc 4 --std_trd` | `out/output_std/{tag}/`, concats in `out/summary_std/` |
| `inference2d.sh` | 2D VQ-head recon of the first val volume | `out/summary_2d/` |

vqclean is in none of them: its only weights lived under `logs0`, which no
longer exists on this box (2026-07-16).

## Support

- `concat_views.py` — per-model two-panel view concats (`[page k | transposed
  page k]`) into `out/summary_3d/` (or `out/summary_std/` with `--std`,
  reading the maskstd maps).
- `perceptual.py` — offline FID/KID/LPIPS between XY and XZ views of output
  folders, convention-matched to `models/base.py`'s validation metrics
  (taming LPIPS on [-1,1], KID feature=64/subset=16, same uint8 map).
- `smoke.py` — one-command sanity check: loads the most recent checkpoint
  under `$LOGS` (or the given dir), pushes a random 64×64×cropz patch through
  `generation()`, asserts `XupX` is finite with Y/X preserved and Z expanded
  by `gan.uprate`. `python test/smoke.py [checkpoint_dir]`
