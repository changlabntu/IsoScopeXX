# Latent-based per-Z-slice registration

An independent experiment on top of the `inference/` API: show that the VQ
**latent** of the IsoScope model can drive per-slice registration of a 3D
stack whose slices have been misaligned by random affine transforms — the
serial-section scenario, where every physical section lands on the previous
one with a small random rotation/shift/scale.

## Why the latent supports this

The model's encoder is strictly **2D per-slice** (`models/MSclean.py`
`vol_to_slices`): each XY slice of a `(B, C, Y, X, Z)` volume passes through
the VQ encoder independently, producing its own `(4, Y/8, X/8)` latent plane.
Because the encoder is convolutional and the coordinates are treated centered,
affine geometry survives it: a rotation of the slice rotates the latent plane
by the same angle, and a translation of `t` px translates it by `t/8` latent
px. So slices can be aligned *in latent space* — including directly from the
stored codec (the compressed codebook indices), without touching raw data.

## Pipeline (3 commands, env `py38zarr`)

```bash
# 1. build ground truth: load a registered ROI from the sample0 OME-Zarr and
#    corrupt it with a per-slice random-walk of similarity transforms
#    (each slice ~U(±0.5°, ±3 px, 1±0.005) RELATIVE to the previous one)
python registration/perturb.py --nz 120 --size 1024 --name full1024

# 2. recover the transforms from per-slice features and re-register
python registration/register.py --dir $OUT/registration/full1024 --feat latent
python registration/register.py --dir $OUT/registration/full1024 --feat codec
python registration/register.py --dir $OUT/registration/full1024 --feat pixel
python registration/register.py --dir $OUT/registration/full1024 --method xcorr

# 3. score against ground truth + figures
python registration/evaluate.py --dir $OUT/registration/full1024
```

`$OUT = /home/cheese/workspace/Output` (perturb's `--out_base` default is
`$OUT/registration`).

## Method (`register.py`)

Per slice pair `(z, z+o)`, `o ∈ --pairs` (default `1,2,4`):

1. **Coarse** — channel-summed FFT phase correlation between the two feature
   planes → subpixel translation init. Phase correlation is nearly unbiased by
   real slice-to-slice content change (measured ~0.02 feat-px on an
   identity-perturbation run).
2. **Refine** — batched Adam over per-pair similarity params (rot, tx, ty,
   log-scale), warping the moving plane differentiably (`affine.py`) under a
   masked L1 loss. Crucially **regularized**: `--reg_t` anchors translation to
   the phase-corr init, `--reg_rs` anchors rotation/scale to identity —
   without this, the L1 fit absorbs *real structural drift* (the anisotropic-Z
   content change between sections) into the transforms: the "registered"
   stack becomes more self-similar than the ground truth itself
   (adjacent-slice NCC above the original's) while drifting several px off.
   Defaults tuned on the identity run: bias floor ≈ 1 px / 0.17°.
3. **Graph solve** — measurements `D(z, z+o) ≈ M(z+o)·M(z)⁻¹` (translations
   ×8 to pixel units) enter a linear least-squares for the absolute chain
   `M_z`, gauge-fixed at `M_0 = I` (`affine.solve_graph`). Multi-offset pairs
   damp random-walk error accumulation; offsets ≳8 hurt (content change
   overwhelms the correlation). `--anchor` (default 0.05) adds a weak
   minimum-norm prior pulling each `M_z` toward identity: the smooth
   low-frequency modes of a long chain are nearly unobservable from pairwise
   data, so residual structural drift integrates coherently without it
   (120 slices: 22 px → 10 px error). Its optimum ≈ 1/(expected drift px) —
   it also shrinks the true low-frequency corruption, so it encodes a prior
   on drift magnitude.
4. **Apply** — `M_z⁻¹` warps each corrupted slice back (bicubic, full res).

Feature modes (`features.py`): `latent` = continuous pre-VQ encoder output;
`codec` = quantized latent rebuilt from codebook indices alone
(`Engine.latents_from_indices`) — registration **from the compressed
representation**; `pixel` = 8×-pooled raw slices (model-free baseline at equal
resolution). `--method xcorr` = the repo's naive translation-only image-space
baseline (`utils/alignments.py`).

## Results (defaults: reg_t 5, reg_rs 10, anchor 0.05, pairs 1,2,4)

small512 — 40 slices, 512², default corruption (original adj-NCC 0.872):

| method | trans err (px, mean) | rot err (deg) | NCC vs original | adj-slice NCC |
|---|---|---|---|---|
| corrupted (none) | 13.3 | 0.77 | 0.37 | 0.789 |
| **latent** | **5.9** | **0.19** | **0.59** | 0.887 |
| **codec** | 6.3 | 0.17 | 0.57 | 0.886 |
| pixel | 6.3 | 0.49 | 0.55 | 0.885 |
| xcorr baseline | 13.4 (fails) | 0.77 | 0.37 | 0.796 |

full1024 — 120 slices, 1024² (original adj-NCC 0.925):

| method | trans err (px, mean) | rot err (deg) | NCC vs original | adj-slice NCC |
|---|---|---|---|---|
| corrupted (none) | 28.2 | 1.24 | 0.38 | 0.851 |
| **latent** | **9.8** | **0.34** | **0.62** | 0.924 |
| **codec** | 10.2 | 0.45 | 0.61 | 0.925 |
| pixel | 9.5 | 0.32 | 0.66 | 0.918 |
| xcorr baseline | 28.1 (fails) | 1.24 | 0.38 | 0.856 |

Readings:
- **Latent/codec registration works**: misalignment drops ~3× in translation
  and ~4× in rotation, adjacent-slice coherence returns to the pristine
  stack's level, and the codec path shows misaligned stacks can be
  re-registered **from the compressed codes alone** (no raw data, no
  re-encode).
- The latent's clearest edge over same-resolution pixel features is
  **rotation at small field of view** (512²: 0.19° vs 0.49°), holding under
  heavy noise+gain corruption (`--noise 0.15 --gain 0.15`: 0.18° vs 0.52°).
  At 1024² the pixel baseline catches up — with 4× more image evidence the
  problem is easier for every feature.
- Neighbor-pair (gauge-free) errors are **sub-pixel** (0.65 px); the larger
  absolute error is accumulated *coherent* drift — the intrinsic ambiguity of
  pairwise registration when real structure drifts through Z ("aligning to a
  banana"): no pairwise method can fully separate true section motion from
  structural motion. The `--anchor` prior bounds it; report absolute errors
  together with the identity-run bias floor.
- The VQ encoder's bottleneck attention is quadratic in latent area: for
  1024² slices use `--batch 1` (24 GB GPU); 256²-tiled encoding (the codec
  production path) is the way around it at scale.

## Outputs per experiment dir

`original.npy`, `corrupted.npy`, `gt_transforms.json`, `perturb_preview.png`
(perturb) → `registered_{label}.npy`, `recovered_{label}.json` (register) →
`metrics.csv`, `errors.png`, `reslice.png` (evaluate; reslice.png is the
qualitative XZ/YZ jagged→smooth comparison).

## Verification / sanity protocol

- `test in scratchpad` (synthetic): warp direction, compose/invert,
  theta consistency, phase-corr sign, refine recovery, graph-solve exact
  recovery, end-to-end synthetic stack (0.045 px mean).
- Identity run (`perturb --rot 0 --trans 0 --scale 0`): measures the
  content-change bias floor of each feature (~1 px / 0.17° for latent with
  default regularization) — rerun it when touching the loss/regularization.
