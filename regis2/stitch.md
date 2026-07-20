# THX10 stitch-tile grid — the dark 5x8 grid in drift_overlay.png (2026-07-21)

The darkened stripes / "boxes with darkened edges" visible in
`find_discont_thx10/drift_overlay.png` are the **microscope stitch-tile grid
in the raw acquisition** (raw filenames: `TH_X10_20xw_..._stitch_Z*.tif` —
each slice is a stitched 20x montage), NOT the 384-px zarr chunk grid. The
grid has two signatures:

1. **Broad vignetting bands (dominant, the visible 5x8 grid):** each tile is
   ~1-2% dimmer toward its edges, so every tile boundary is a wide shallow
   dark band. Best seen in a **mean** blend (`overlay.png`,
   `overlay_grid.png`); a max blend suppresses it.
2. **Narrow blend seams (subtle):** sharp per-slice dips (~13 gray levels,
   ~3.3 sigma) at the tile boundaries that **wander ~0.2 raw px/slice**
   (~160 px over the stack) — what a max blend picks up instead.

## Fitted tile grid (comb fit on the mean of every 10th z, level 2)

| | pitch (l2) | pitch (l0) | phase (l2) | band centers (l2) |
|---|---|---|---|---|
| x | 472 | **1888** | 396 | 396, 868, 1340, 1812, 2284, 2756, 3228 |
| y | 475 | **1900** | 352 | 352, 827, 1302, 1777 |

- Grid spans 7.5 x 4.4 cells over the 3552 x 2112 level-2 image → reads as
  ~8 x 5 boxes.
- Level-0 (cropped-frame) band centers: x = 1584, 3472, 5360, 7248, 9136,
  11024, 12912; y = 1408, 3308, 5208, 7108.
- In **raw uncropped** coords (crop offset col 452 / row 523) the x
  boundaries fall at 2036, 3924, 5812, 7700, 9588, 11476, 13364 — first one
  ~2048, so plausibly a **2048-px camera tile with ~150 px (~8%) overlap**.
- Mean band depth 1-2% of local brightness (deepest y band ~5%).

**Not chunk-aligned:** band positions mod 384 are scattered; the grid pitch
(472/475 l2) is unrelated to the 384-px chunk grid at any pyramid level. A
few lines land near chunk boundaries by coincidence, which is what makes it
read as a chunking artifact.

## Narrow-seam component (max-blend analysis)

Strongest narrow bands in the 82-slice max blend, level-0 px (level-2):
x = 4470 (1117), 5261 (1315), 5545 (1386), 6182 (1545), 6423 (1605),
8967 (2241); y = 3280 (820), 5234 (1308); depths 19-28 gray levels. The y
seams sit exactly on the grid's y lines (820/1308 vs 827/1302); x seams
scatter around the grid because of the z-wander — the main x-seam is
detectable in 290/816 slices and migrates ~+0.2 raw px/slice, so band pairs
like 1545/1605 are **one seam at two z-epochs**. Verified present in the
raw uncropped TIFF (z=561, `stitch_Z2244.tif`, raw col ~6577).

## Pipeline exonerated

`crop_all.py` is a pure rectangular crop (cropped TIFF verified byte-identical
to the raw slice region), `tif_to_zarr.py` downsamples by chunk-agnostic 2x2
block mean, Blosc/zstd is lossless — none can create or darken a grid.

## Connections to the regis2 results

- The seam migration is an independent fingerprint of the acquisition drift
  found by `measure_drift.py` (same order of magnitude; the tile grid is
  fixed to the stage frame, not the specimen).
- The main seam is deepest at z ~ 553-564 — exactly the z=554 major CONTENT
  gap event from the gap scan (`THX10_GAPS_DRIFT.md`). That event may be a
  stitching/acquisition episode rather than a material transition.
- **Caveat for drift measurement:** the grid is stationary-in-image-frame
  structure, so patches straddling a boundary could bias phase-correlation
  toward zero (same family as the background fixed-pattern lock). Effect
  should be small (1-2% per-slice contrast).
- **Possible cleanup:** the vignetting is stable and multiplicative — a
  per-tile flat-field estimated from the mean image could divide it out.

## Figures (in `<out>/find_discont_thx10/`)

- `overlay.png` — mean of every 10th z, level 2 (the grid is obvious).
- `overlay_grid.png` — same with the fitted grid lines.
- `seams_marked.png` — max blend with the narrow-seam bands marked, plus
  profile vs chunk grid; band positions in `bands.npz`.
- `stitch_overlap.png` — single slice z=408 with the narrow-seam positions.
