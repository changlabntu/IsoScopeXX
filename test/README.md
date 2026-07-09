# test/ — checkpoint loading & patch inference utilities

All scripts must be run **from the repo root** (they read `ldm/*.yaml` and
`cfg/env.json` by relative path; `test/load.py` chdirs there defensively).

## `test/load.py` — load any experiment's checkpoint

Rebuilds the `GAN` from the run's own snapshot (`{models}.py` + `config.json`
inside the checkpoint dir) and swaps in the pickled component modules
(`{name}_model_epoch_{N}.pth`). Component names are discovered from the
filenames, so it works across the whole lineage (`quant_conv/quantize` era as
well as `quantizers/quant_convs/post_quant_convs/inject_convs`).

```python
from test.load import load_model
gan, cfg = load_model('$LOGS/THX10SDM20xw/thx10/vqcleanM0aMSskipP/Scale4/band5')
# path may be the experiment root, .../checkpoints, or a timestamped run dir;
# epoch defaults to the latest complete one.
```

## `test/inference.py` — per-file 3D patch inference (no stitching)

Each input `.tif` (page order `Z,Y,X`) is one patch: normalize per the run's
`--nm`, run `gan.generation(deterministic=True)`, save the isotropic `XupX` as
float32 `.tif`.

```bash
python test/inference.py \
    --checkpoint /home/gary/workspace/logs/THX10SDM20xw/thx10/vqcleanM0aMSskipP/Scale4/band5 \
    --source /home/gary/workspace/Data/THX10SDM20xw/roiD/val/roiD \
    --destination /home/gary/workspace/Data/THX10SDM20xw/out/band5 --save_input
```

`--destination` defaults to `/home/gary/workspace/Data/THX10SDM20xw/out/{experiment
name}` (`DEFAULT_OUT` in `test/inference.py`). `test/concat_views.py` builds
per-model `[XY | ZX]` view concats into `out/summary/` (plus `input.tif`, the
trilinear-upsampled input, as reference) — see the sweep block in
`test/inference.sh`.

- Generator components run in `.train()` by default (MC dropout: batch-stat BN
  + active dropout, stochastic per run); pass `--eval` for deterministic
  running-stat inference. The `GAN` module itself always stays eval so
  `generation()` skips the training-time `cropz` crop.
- `--dsp 1` feeds a real anisotropic volume as-is; the default (the config's
  `dsp`) mimics training by Z-subsampling an isotropic input first.
- `--cropsize/--cropz` optionally center-crop before inference (Y/X must stay
  divisible by the encoder's downsample factor, 16 for the default vqgan yaml).
- `--nm` overrides the config's normalization; `'11p'` needs `--norm_stats
  <dataset>/norm_stats.json` (key = the tif's parent directory name).

## `test/smoke.py` — one-command sanity check

```bash
python test/smoke.py [checkpoint_dir]
```

Defaults to the most recently written run under `$LOGS` that has `.pth` files,
loads it, pushes a random 64x64xcropz patch through `generation()`, and
asserts `XupX` is finite with Y/X preserved and Z expanded by `gan.uprate`.
