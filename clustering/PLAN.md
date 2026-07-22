# clustering/ — plan: mapping distinct material textures & structures from the VQ latent

**This is a materials-science project.** The volumes are 3D images of a
material sample, and the goal of this folder is to discover and map
**distinct material textures and structural motifs** (e.g. laminar /
layered regions, dense compact phases, speckled cellular fields, plate-like
domains, phase boundaries) directly **from the VQ latent space** of the
compression codec — i.e. to show *what kind* of material microstructure
occupies each region, not merely *how much* structure is there.

**Folder scope.** Everything about clustering / texture-mapping / material
microstructure identity from the stored VQ codecs lives here; registration,
drift, and gap detection live in `registration/` (formerly `regis2/`, merged
2026-07-22). Moved in from regis2: `latent_cluster.md` (findings log) and
`latent_overlay.py` (the box/contour PC overlay, output convention
`Output/regis2/cluster_fable/XXX.npy + XXX/`; retired 2026-07-22 together
with the concluded F4/F5 harness baselines — all three recoverable from git
history, their outputs archived under `Output/regis2/cluster_fable/` and
`Output/clustering/harness/`).
New experiments from this plan write to **`Output/clustering/`** with the
same `XXX.npy + XXX/` convention (cluster_fable stays as the archive of the
PC-overlay era).

**Anchors provenance.** `anchors.csv` (the frozen probe-label set `harness
eval` uses) was authored against the v0b classes of the superseded zwin=3
lineage; the current lineage is v1 → v1a → v1a1 (zwin=1). The labels remain
valid as ground truth for the probe, but their class names predate the v1
round.

## 1. Where we are

Three generations of overlays (`coarse_spectrum_*` → `zw3_coarse` boxes →
contours) all color patches by linear projections of **mean-pooled continuous
latents**. After fixing the display-side faults (Z-window, raw-pixel gate,
2-D color), the maps are smooth and honest — but they encode *amount* of
structure (PC1 ≈ material density/brightness, corr +0.66 with raw patch
std), not *kind* — they cannot tell a laminar region from a speckled one at
equal density. The earlier survey (`latent_cluster.md`) already showed why
this is intrinsic: silhouette ≤ 0.19 for every pooled-feature recipe,
material texture spread over a diffuse high-dimensional tail that linear
PCs cannot summarize.

## 2. Diagnosis — why pooled-latent PCs cannot show material texture types

1. **Pooling is the wrong statistic.** Mean/max over 384×384×(2w+1) voxels
   reduces a material texture to its average activation ≈ brightness.
   Texture identity — what distinguishes one microstructure from another —
   lives in *distributions* and *co-occurrences*, which pooling integrates
   out before analysis starts.
2. **Linear projection of a diffuse cloud yields amplitude.** When variance
   is spread thin across many directions, the top PCs align with the one fat
   direction — overall magnitude. Measured: PC1–3 ≈ 57% of variance, all
   amplitude-flavored.
3. **One vector per 384 px patch is the wrong unit.** Patches straddle
   structures; mixtures average into the middle of feature space.
4. **(New, measured) The continuous latent is only C=4 channels**
   (`ldm/vqgan.yaml embed_dim: 4`) — so channel-covariance/Gram statistics
   (the classic texture representation) are 4×4 and carry almost nothing.
   Any second-order signal must come from **spatial** arrangement, not
   channel mixing.

## 3. The under-used asset: the codes themselves

Per patch and slice the codec stores **discrete code indices** per scale:
s0 6×6 → s3 48×48 over a 256-entry codebook (`scale_k` arrays, int32,
shape (1, 48, hk, wk)). At s3 that is one token per **8×8 px** — 2304 tokens
per slice per patch, 36 tokens per 48-px cell per slice. The quantizer has
already factored the volume into a vocabulary of local material textures;
every analysis so far
re-embedded the codes into ℝ⁴ and averaged them, discarding exactly the
identity information we're after. The plan pivots to **code statistics**.

## 4. Success is measured, not eyeballed (build the harness first)

Every prior iteration was judged by looking at PNGs — that is how we ended
up optimizing prettiness instead of information. Before any new feature:

**E0 — anchor set + benchmark harness** (`harness.py`, no GPU).
- Hand-label ~5 visually distinct **material region classes** from the
  mid-slice figures (e.g. dim layered/laminar lobe, bright dense phase,
  speckled cellular field, phase-boundary rim, flat plate-like domain),
  ~15–25 patches each, spread over ≥ 4 z-chunks. Stored as
  `anchors.csv` (z, x, y, class) in this folder — 10–20 min of labeling
  using the existing `zw3_coarse/` boxes figures as the picking surface.
- Metrics, computed for ANY candidate feature matrix:
  - **probe**: leave-one-chunk-out kNN / linear-probe accuracy on anchor
    classes — the headline number: how much material-type identity the
    feature carries.
  - **amp-leak**: |corr| of the feature's top embedding axes with raw patch
    std — want low; quantifies the amplitude trap.
  - **coherence**: mean feature-space similarity of spatially adjacent
    cells/patches vs random pairs — smoothness without labels.
  - **z-stability**: same-column similarity across adjacent chunks.
- Ranking rule: probe first, amp-leak as disqualifier (> 0.8 → reject),
  coherence/z-stability as tie-breakers. Only winners get overlays.

## 5. Feature candidates (ranked)

**F1 — metacode histograms (first priority).** The 256 codes are too many
for 36-token cells and per-code counts are noisy; but codebook entries are
vectors in ℝ⁴ whose *usage contexts* define similarity. Build once per
scale: cluster the 256 codebook vectors (plus, better, their mean 3×3 code
context) into **K ≈ 24–32 metacodes**; then every patch/cell gets a K-bin
histogram per fine scale (s2, s3), tf-idf weighted, Hellinger-compared
(√-transform then Euclidean ≈ Hellinger, so PCA/UMAP/kNN apply directly).
Brightness enters only via which codes fire, not via magnitude. GPU-free
except one codebook read (`quantizers[k].embedding`).

**F2 — directional code co-occurrence (second order, replaces Gram).**
Per cell, count metacode *pairs* for right-neighbor, down-neighbor and
next-z-slice neighbor separately (3 × K×K upper-triangle, heavily sparse →
keep top-M entries or project). Captures how the material is *arranged*
(layered vs speckled vs fibrous microstructure) and its **anisotropy**
(H/V/Z asymmetry distinguishes laminae from isotropic speckle) — precisely
the structural signature that mean-pooling destroyed. Built from the same
index arrays, no GPU.

**F3 — latent textons.** K-means (K ≈ 64) on 3×3×C fine-latent
neighborhoods sampled across the volume → per-cell texton histograms.
Continuous-space sibling of F1/F2; more expressive, needs one GPU pass to
rebuild latents (cacheable like `feat_zw*`). Run only if F1/F2 plateau.

**F4 — cells-from-cache (free baseline).** Reshape the existing cached
`fps` (814 × 4-scale × C×8×8) into 64 per-cell C-vectors per patch and push
through the harness. Expected to lose to F1/F2 (it's still mean-pooled),
but it is the honest baseline the code features must beat, and it costs
nothing.

**F5 — raw-image control.** LBP or wavelet-energy histograms per cell from
the raw mid-slices. Decides the interpretation of everything above: if raw
descriptors separate the anchor classes and the code features don't, the
codec entangles material texture (→ actionable for codec training, e.g.
embed_dim); if neither separates them, the material is a microstructural
continuum at this scale and distinct texture "types" don't exist — a
legitimate terminal answer that closes this line cleanly.

## 6. Coloring (only after the harness picks a winner)

- **UMAP → 3D → RGB via LAB** (fixed a/b plane scaling, L clamped to
  [35, 80]): similar material textures get similar colors, distinct
  microstructures distinct hues, no single axis has to carry the story. Replaces PC1→hue in
  `latent_overlay.py`'s renderer (factor the box-drawing out so it takes a
  per-cell RGB array; cell-level = 8× finer mosaic than current patches).
- **Anchor-similarity maps** as the interpretable companion: one hue per
  anchor class, per-cell color = barycentric mix of class similarities, and
  the probe confusion matrix printed alongside.
- Keep the `(z, x, y, …)` matrix convention: winners export
  `XXX.npy` = (z, x, y, cell_row, cell_col, emb1..emb3 [, class, p_class]).

## 7. Order of work

| step | what | needs GPU | gate |
|---|---|---|---|
| M0 | `harness.py` + `anchors.csv` | no | — |
| M1 | F4 baseline through harness | no | establishes the bar |
| M2 | F1 metacode histograms (+ tf-idf) | codebook read only | beat F4 probe |
| M3 | F2 co-occurrence + anisotropy | no | beat F1 |
| M4 | UMAP→RGB overlays + npy for the winner | no | harness winner only |
| M5 | F5 raw control (always run before concluding) | no | interpretation |
| M6 | F3 textons | one pass | only if F1/F2 plateau |
| M7 | conditional: probe/anchor coloring, codec feedback | no | — |

Rough sizes: M0 ~150 lines + labeling session; M1 ~50; M2 ~150; M3 ~150;
M4 mostly refactor of `latent_overlay.py`. Everything after the anchors is
scriptable and cache-driven; only M6 touches the GPU.

## 8. Risks / honest outcomes

- **The continuum answer.** All prior evidence says the material's texture
  may genuinely be a continuum here — gradual microstructural variation
  with no sharp type boundaries. The harness makes that a *measured*
  conclusion (low
  probe accuracy for every feature incl. F5) instead of another round of
  disappointing PNGs — and the UMAP-RGB continuum map is still a useful
  deliverable in that world.
- **Anchor bias.** 5 classes chosen by eye may not carve nature at its
  joints; mitigate by also reporting label-free coherence, and revising the
  class list once after the first probe round.
- **Code sparsity per cell** (36 tokens/slice): mitigated by metacodes
  (K≪256), pooling the z-window's 7 slices (~252 tokens), and cell-level
  smoothing; if still noisy, fall back to 96-px cells (4×4 per patch).
- **Cache staleness**: cache filenames must encode scale set + K + zwin
  (fix the known gap where `--spatial/--model/--epoch` are not in the key).
