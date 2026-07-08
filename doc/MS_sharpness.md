# Sharpness comparison: vqclean vs MS vs MSskipE

Single-volume inference study on `val/roiD/th000008003.tif` (real anisotropic input,
48×384×384 → 384×384×384 isotropic output, ×8 Z uprate). Run via `test/inference.py`.
Purpose: understand why some outputs look sharper in the XY plane than others, and what
that sharpness costs in the YZ (super-resolved) plane.

## Models compared

| tag | model | checkpoint | epoch | mode |
| --- | --- | --- | --- | --- |
| vqclean_ep600 | `vqclean` (single-VQ, **non-MS**) | `logs0/.../vqcleanVQ/max5skip4` | 600 | train |
| MS_ep100 | `vqcleanM0aMS` (vanilla MS) | `logs/.../MSadv0/.../20260704_011354` | 100 | train |
| MSskipE_ep200 | `vqcleanM0aMSskipE` | `logs/.../MSskipE/Scale4/ema999` | 200 | train |
| MSskipE_eval | same as above | same | 200 | eval |

Caveats that confound a strict ranking: (a) train mode = **one MC-dropout sample**, whose
high-frequency noise inflates apparent sharpness; only MSskipE was also run in eval.
(b) vqclean trained on direction `roiAdsp4`, the MS runs on `roiD`. (c) Different epochs.

## Sharpness metrics

XY = the native-resolution plane; YZ = the plane the model super-resolves. Gradient =
mean |first difference|; lapVar = variance of the in-plane Laplacian (classic focus
measure, higher = sharper). YZ/XY ≈ 1.0 means isotropic; the input is 2.3 (anisotropic).

| output | lapVar XY | gradX | gradZ | gZ/gY | YZ/XY isotropy |
| --- | --- | --- | --- | --- | --- |
| input (native) | 0.00436 | 0.0269 | 0.0360 | 1.75 | 2.34 |
| **vqclean_ep600** | **0.00189** | 0.0157 | 0.0141 | 0.92 | 0.98 |
| MS_ep100 | 0.00107 | 0.0148 | 0.0128 | 0.91 | 1.08 |
| MSskipE_ep200 | 0.00069 | 0.0120 | 0.0099 | 0.86 | 1.10 |
| MSskipE_eval | 0.00018 | 0.0083 | 0.0074 | 0.99 | 1.23 |

**XY-plane sharpness ranking: vqclean > vanilla MS > MSskipE (train) > MSskipE (eval).**

## What the images show (YZ plane is decisive)

- **vqclean (non-MS)** is the sharpest in both planes, but its YZ background carries a
  faint repeating **ripple/lattice texture** — the ConvTranspose alias artifact (the
  target of the `val_lat_*` metric and the resize-conv / BlurPool work in
  `doc/research_artifact_directions.md`). Much of its extra "sharpness" is this synthetic
  high-frequency grid, not recovered structure, which is also why it scores high on *both*
  XY sharpness and the gradient isotropy ratio (the lattice adds equal energy in both
  planes, so the ratio metric is misled here).
- **vanilla MS / MSskipE (train)** have cleaner backgrounds and coherent filaments through
  Z, no obvious lattice. Softer, but the structure reads as continuous morphology rather
  than imposed texture.
- **MSskipE (eval)** is the smoothest and cleanest — no lattice, but small features wash
  out.

## Conclusion

- **MS lowers peak XY sharpness but yields nicer, more physically plausible morphology.**
  The tradeoff vs vqclean is partly sharpness-vs-**artifact**, not sharpness-vs-detail:
  vqclean's crispness is inflated by the alias lattice the MS/anti-alias line is designed
  to remove. Consistent with MLflow: MSskipE had the best `val_kid` (3.99) of the thx-MS
  sweep.
- **Train vs eval:** a single train-mode pass is *one* dropout sample; its sharpness is
  variance, not signal. For a clean deterministic deliverable use `--eval`; for MC dropout,
  average N passes (mean = prediction, std = uncertainty) rather than shipping one draw.
- **skipE vs vanilla MS:** not settled by this single-slice test (different epochs, MC
  noise on both). MSskipE wins on isotropy realism (`val_kid`); vanilla MS is marginally
  sharper here. Needs a matched-mode comparison to call.

## Open follow-up

Rerun vqclean / MS / MSskipE all in `--eval` on this volume, then run the real
`utils/metrics_spectral.py` lattice-peak ratios (`p2diag`/`p4diag`) on the YZ planes —
removes the dropout-noise confound and quantifies the alias lattice directly instead of
inferring it from gradients.
