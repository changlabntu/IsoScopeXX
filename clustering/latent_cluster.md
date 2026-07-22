# Latent clustering & texture survey (THX10, 2026-07-21)

Unsupervised survey of the skipU epoch-300 VQ latent over the THX10 codec
(`/media/cheese/Ghc_data3/THX10/thx10codec/codec`, 17 z-chunks × 814 patches of
384 px × 48 slices). Features are rebuilt from stored codes
(`Engine.latents_from_indices`, no re-encode), pooled over Z and adaptive-pooled
to an 8×8 plane. Outputs + `features_cache.npz` under
`Output/regis2/cluster_texture_all/`.

## Two feature families

- **Density** — summed-all-scale latent, **mean**-Z pooled. Coarse-scale
  dominated. Clusters cleanly: k=2 (background vs content) has silhouette
  **0.52**; k=7 just carves the content into concentric **density shells**
  (empty → sparse → moderate → dense → brightest). Groups mean *how much*
  structure, not *what kind*.
- **Texture** — per-scale **max** pooled, then scale-balanced +
  column z-score/winsorise(±4) + per-patch mean-subtract + L2-normalise
  (kills brightness). Content patches only (gated by density k=2, direction
  fixed by fine-scale s3 energy). Removes the density axis: clusters
  **interleave across the network** instead of nesting as shells.

## Key result: the latent has no discrete texture *types*

Intrinsic-dimensionality test on the 1442 content patches (all 17 stacks):

| feature recipe | PC1 var | PC1–3 var | best k | silhouette |
|---|---|---|---|---|
| all 4 scales + L2 (texture) | 10% | 25% | 2 | 0.14 |
| all 4 scales, no L2 | 20% | 36% | 2 | 0.19 |
| coarse s0+s1 + L2 | 14% | 27% | 2 | 0.19 |
| fine s2+s3 + L2 | 11% | 30% | 2 | 0.11 |
| finest s3 + L2 | 13% | 34% | 2 | 0.12 |

- **No recipe clusters well** (silhouette ≤0.19, best k=2 everywhere; PC1 only
  10–20%). Texture content is diffuse/high-dimensional, not a set of types.
  Forcing k=6 bins a continuum → adjacent, near-identical patches straddle bin
  edges and get opposite colours (a real artifact; centroid distances all
  0.48–0.81, i.e. equidistant blobs, no gaps).
- **Fine scales + L2 are the most continuum-like** (lowest silhouette). The
  moves that isolate texture (emphasise fine scales, L2-normalise) are exactly
  the moves that flatten cluster structure.
- **The only well-separated split is density.** Coarse-no-L2 (silhouette 0.19)
  is just amplitude: its PC1 correlates −0.74 with fine-scale energy, +0.57 with
  latent magnitude.

## Cross-stack & depth

Clustering all 17 stacks jointly, the same clusters appear at every depth
(markers intermix in the joint scatter). Composition shifts with depth: the
dense/fine textures peak mid-volume (z≈200–600), a single coarse texture
dominates the shallow/deep ends. Caveat: the stack **bottom** (z≈779–816) is
also where real geometric tears live (see `THX10_GAPS_DRIFT.md`), so the
edge-vs-core texture difference is partly acquisition, not only material.

## Recommendation

- **Texture → don't hard-cluster.** Colour each patch by its position along the
  texture PC1 (continuous orange→blue or turbo ramp). Faithful, artifact-free,
  and the depth trend reads directly (`pc1_overlay_*`, `coarse_spectrum_*`).
- **Regions by amount-of-structure → density k=2** (or coarse-no-L2), accepting
  that the label means density, not texture kind.
- Colour hard clusters by feature-space similarity order, never raw KMeans
  label index, if clusters are used at all.

## Fixed overlays: `latent_overlay.py` (2026-07-21, follow-up; script retired 2026-07-22 — in git history)

The `coarse_spectrum_*` overlays above were diagnosed with three display-side
faults and rebuilt. Outputs now live under `Output/regis2/cluster_fable/` as
`{name}.npy` + `{name}/` per-chunk PNGs (name = `zw{zwin}_{blocks}`, current
run `zw3_coarse`). The `.npy` is a float32 `(N, 6)` matrix, one row per gated
content patch: **(z, x, y, pc1, pc2, pc3)** — z the chunk's displayed mid
slice, x/y patch-centre pixel coords at the codec's zarr level (same frame as
`thx10codec/gaps.csv`), PCs sign-aligned (pc1 with fine-scale energy, pc2/pc3
with raw patch std). Fully-flat patches saturate the ±4 feature clip and
collapse to identical PC rows. The fixes:

1. **Z-window** — features now pool only the central ±3 latent planes around
   the displayed slice (`load_latent_features(..., zwin=3)`, new `--zwin` in
   `inference/latent_tsne.py`; latent Z is 1:1 with input slices).
2. **Raw-pixel gate** — content = mid-slice patch std > 2× a per-chunk noise
   floor (median std of the darkest-decile patches), replacing the latent-
   density k=2 gate. Keeps **7405**/13838 patches (vs 1442) — the dim
   structured lobes are now colored.
3. **2-D color** — PC1→hue (blue→red), PC2→lightness, 2-D legend inset with
   variance shares, replacing the 1-D turbo ramp.

Result: PC1 of the coarse no-L2 feature jumps from ~14–20% to **44%** of
variance on the broader population (corr +0.66 with raw patch std) and the
overlays show smooth spatial gradients that track visible structure — red
dense core → green ring → blue dim periphery — instead of salt-and-pepper.

**Attribution control:** re-running with `--zwin 24` (whole-chunk pooling)
does *not* restore the old noise — the old figures' incoherence was mostly
the restrictive amplitude gate + per-column z-score on that tiny homogeneous
subset + the 1-D turbo stretch. What the thin window buys is axis *meaning*:
at `zwin=24` PC1 collapses onto fine-scale energy (corr 0.95, pure
amplitude); at `zwin=3` it stays partly slice-specific (corr 0.68).

Caveats: at the stack bottom (z0768-0816, specimen mostly sectioned away)
the darkest-decile floor is exactly 0, so the gate keeps every non-constant
patch — output is still sensible (flat plate → uniformly low PC1). Per-chunk
feature caches `feat_zw{w}_{chunk}.npz` (shared at the `cluster_fable/` root
across experiment names) make re-rendering and re-analysis GPU-free.

**zw3_coarse.npy analysis** (`cluster_fable/zw3_coarse_analysis/`,
2026-07-21): PC1–PC2 splits into a featureless population (PC1 ≈ −20…−10,
dominant in the end chunks, funnelling into the clip-collapse point; 415/7405
rows are exact duplicates) and a diffuse content cloud (PC1 0–40). PC1 median
vs depth: flat −11 at both ends, plateau ≈ +10 at z 168–216, then a slow
monotonic decline to +1.5 at z=504 — broken by a +3.8 counter-trend rebound
at z=552 (the chunk holding the z=554 gap event / deepest stitch seam) before
collapsing at z=600; consistent with an acquisition episode around z≈554
shifting image statistics. Core (top-quartile-PC1) centroid wanders ~1.5 mm
x / ~2.3 mm y through the stack — material shape change, ~30× the measured
stage drift. Gaps: 184/193 land on gated patches; their within-chunk PC1
percentiles skew high (median 0.63; 37% in the top quartile vs 25% uniform,
≈3.7σ) — discontinuity events preferentially strike dense regions, though
part of this is detection sensitivity (NCC collapse needs structure).

`--style contour` (experiment `zw3_coarse_contour`) renders the PC1 field as
smoothed filled/line contours instead of boxes — far less occluding, but PC2
is not encodable in level sets, so that style is PC1-only (plain colorbar).
The patch grid is NaN-filled, gaussian-smoothed (σ=1 cell, normalised
convolution, support threshold 0.3) before contouring.

## Anomaly detection (`inference/latent_tsne.py`)

`--feat latent` + IsolationForest flags a fixed fraction (`--contamination`) —
count is a quota, not a measure of how anomalous the chunk is. mean vs max vs
per-scale features give substantially different top-N (13–21 of 41 overlap);
max pooling implicitly favours fine structure. Validate flags with the middle-
slice overlay (`anomalies_overlay_*`, emitted automatically with the thumbnail
figure), not with dot position on the t-SNE map. Script now also has
`--scale k` (single residual scale), `--zpool` (in
`registration/experiments/tsne_corrupt.py`),
and `--zwin w` (central-window Z pooling, added for the fixed overlays above).
