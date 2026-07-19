# regis2 — latent registration, round 2

Follow-ups to `registration/` (whose components are imported, not modified),
run on single 3D TIFF patches from the skipU300 grid
(`/home/cheese/workspace/Data/thx10/roiAdsp4`, 512 patches of 32×256×256 in
[-1,1]; default patch `007003005`). Outputs under
`/home/cheese/workspace/Output/regis2/`.

## Scripts

- **`perturb_options.py`** — previews the corruption models on one patch
  (ZX reslices, Z ×8 trilinear): `walk` (registration/ defaults), `jitter`
  (non-accumulated), `drift` (smooth sinusoid ±10 px / ±1°), `walk_big`
  (2× walk).
- **`run_walkbig.py`** — full corrupt → register-from-latent → enhance run on
  one patch. `--perturb walk|walk_big|walk_drift`; the latent-registered
  decode is compared against the unwarped and pixel-warped decodes in a
  7-column ZX tif (`{tag}_compare7.tif`: original | corrupted | original enh |
  corrupted enh | latent-reg enh | latent-reg input | latent-reg input enh).
  Latent-warp variants: `--latent_interp bilinear|bicubic`,
  `--warp_space postquant|prequant` (prequant = warp the continuous encoder
  output, then quantize — reimplements `MSclean.encode`'s VAR loop, verified
  ≡ to 5e-7), `--no_rot` ablation.
- **`tsne_corrupt.py`** — the skipU300 latent-t-SNE survey with a random
  `--frac` of patches corrupted before encoding, marked green.
  `--feat latent_mean|jump|both`, `--corrupt walk_drift|drift`
  (`--drift_scale`), `--thumbs` for the Z-MIP-annotated figure.

## Findings

1. **Latent-warp ≈ pixel-warp on the enhanced volume** in every corruption
   regime; registration must precede/inside the decode (corrupted-enhanced is
   visibly torn). Trade-off is consistent: latent-reg slightly better NCC to
   the reference decode, pixel-reg slightly better Z-coherence (adj-NCC
   ~0.987 vs ~0.988 on walk_drift).
2. **Closing the coherence gap**: bilinear→bicubic and postquant→prequant
   each recover part of it (0.9859 → 0.9867 → 0.9870 vs pixel 0.9884). The
   `--no_rot` run shows the residual gap is NOT encoder rotation
   non-equivariance — it is the stride-8 warp grid itself (irreducible
   without pixels). Best latent recipe: **bicubic + prequant** (prequant
   needs the continuous latent, so codec-only inputs fall back to postquant).
3. **t-SNE misalignment detection**: the content feature (`latent_mean`) is
   alignment-invariant — 0/51 corrupted detected (good for microstructure
   surveying, useless for QC). The **`jump` feature** (adjacent-latent-plane
   phase-corr |t| + NCC stats, computable from stored codes) detects
   walk-type corruption at **51/51** (precision@51). Sensitivity floor ≈1 px
   per slice: smooth drift ±10/±5/±3.3 px → 76% / 47% / 24%. Failure mode =
   weak-texture patches whose phase-corr noise mimics small jumps (the
   near-black thumbnails in `tsne_corrupt_thumbs_*.png`); known upgrades:
   offset-2/4 pair stats (drift accumulates, noise doesn't), texture-weighted
   confidence.
4. Registration behavior matches `registration/`: pairwise error ~0.7 px /
   0.15° in every regime; absolute error converges to the ~6–7 px coherent-
   drift gauge floor; ~0.5° spurious rotation is estimated even when none
   exists (content-drift bias), at negligible downstream cost.

QC loop closed end-to-end from the codec: detect misaligned patches (jump
feature) → recover transforms (`register.py`) → correct in latent space and
decode enhanced (`enhance.py`).
