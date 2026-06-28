# Three-view fused data (`E2507218fuse`)

Findings from inspecting `Data/E2507218fuse/E2507218cube/train/{zcube,xcube,ycube}/`.
These are real multi-view optical-sectioning acquisitions of the **same sample**, not the
in-model faked anisotropy the original recipe assumed.

## What the data is

- Three folders `zcube/`, `xcube/`, `ycube/`, **512 TIFFs each**, with an **identical
  filename set** across all three (they pair by shared filename — fits `--direction
  zcube_xcube_ycube`). A `val/` dir also exists.
- Each file: **256³ cube, float32**, per-cube normalized to ~[-1, 1], very sparse
  (mean ≈ −0.9, mostly background floor).
- `tiff.imread` axis order is **`(Z, Y, X)`** = numpy `(axis0, axis1, axis2)`.

## Each cube is LR along exactly one axis (8× anisotropy)

Each view is an optical sectioning from a different direction: HR in two axes, **8× low-res
along the axis it is named for**. Verified two ways:

- Per-axis gradient energy is lowest along the named axis.
- `pool8 → upsample8` along the named axis is **lossless** (corr ≈ 0.99), but along the HR
  axes it destroys ~15% of structure (corr ≈ 0.82–0.89). So the LR axis genuinely carries
  only 256/8 = **32 independent samples** upsampled to 256.

| cube  | LR (blurry) axis | HR axes | pool8→up8 corr along Z / Y / X |
|-------|------------------|---------|--------------------------------|
| zcube | **Z** (axis0)    | Y, X    | **0.993** / 0.821 / 0.828      |
| ycube | **Y** (axis1)    | Z, X    | 0.860 / **0.990** / 0.850      |
| xcube | **X** (axis2)    | Z, Y    | 0.907 / 0.889 / **0.993**      |

`pool8` along the LR axis is therefore the exact **forward degradation operator**.

## They ARE registered, but NOT pixelwise-identical

- **Registration is correct.** `zcube/N` matches `xcube/N` at matched-resolution corr ≈ 0.60,
  while every *other* xcube tile scores ≤ 0.08. That 0.60-vs-0.08 gap proves the filename
  pairing maps to the same physical location. A ±6-voxel shift search gains almost nothing,
  so there is no spatial offset to correct.
- **The ~0.60 ceiling is a view/photometry difference, not misregistration** — and pooling
  cannot raise it because the mismatch was never spatial:
  - Perpendicular blur (each view blurred along a different axis).
  - Different sensitivity/brightness: **xcube carries ~2× the foreground of zcube**; per-cube
    `'11'` normalization scales each to its own [-1, 1]. zcube's signal is largely *contained*
    in xcube's (P(xcube fg | zcube fg) ≈ 0.75 vs reverse ≈ 0.33) — same location, xcube just
    brighter / more sensitive.

## Implications for supervision (input = zcube)

`XupX` is the isotropic full-res output. Each real view is its LR projection along one axis:

```
pool_Z(XupX) ≈ zcube   (current L1: tensor dim -1)
pool_X(XupX) ≈ xcube   (new: dim 3)   ← carry real HR-Z structure zcube lacks
pool_Y(XupX) ≈ ycube   (new: dim 2)
```

Pooling along X/Y preserves Z detail, and xcube/ycube are HR in Z, so these terms supervise
the Z super-resolution with **real** data. Caveats:

- **Tensor axis map** (dataloader yields `(B, C, Y, X, Z)`): dim4 = Z (zcube LR),
  dim3 = X (xcube LR), dim2 = Y (ycube LR).
- **Pool factor = 8** (the real anisotropy ratio), independent of `uprate` (Z-specific).
  Use `mean` pooling (matches optical-section integration).
- **Do NOT `dsp`/`usp` xcube/ycube** — apply only the shared `cropz` Z-window so they stay
  HR in Z; subsampling would discard the very information they supervise.
- **Intensity harmonization is mandatory**: normalize the three cubes to a *shared* scale
  (or per-projection normalize before L1), else the brightness gap dominates the L1 and the
  X/Y terms fight the Z term.
- Pixelwise projection L1 against xcube/ycube *is* valid here (registration is correct);
  the only blocker is photometry. They can alternatively/additionally feed the six-way
  discriminator as a real HR-Z slice distribution.

## Implemented in `models/vqcleanM0aSup0.py`

Supervised variant of `vqcleanM0a`. `generation()` captures the non-input views at
full Z (`self.aux_views`, before the `dsp` subsample), and `backward_g` adds the X/Y
projection terms: mean-pool `XupX` by `--aniso` (=8) along dim3 / dim2 and L1 against
the same pool of xcube / ycube, weighted by `--lamb_xy`. Run with
`--direction zcube_xcube_ycube --nm 11p --models vqcleanM0aSup0` (see `run.sh`).
