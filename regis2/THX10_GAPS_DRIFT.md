# THX10 whole-stack Z-scan: gaps + drift (2026-07-20/21)

**Context.** THX10 is a 3D **materials-science volume** — a material sample
imaged slice-by-slice along Z, the anisotropic-acquisition regime this repo
targets (good X/Y, undersampled Z; see the top-level CLAUDE.md). Serial Z
acquisition of a physical specimen accumulates two acquisition defects that
this scan quantifies from the learned VQ latent, over the full 816-slice
volume: **sparse inter-slice discontinuities** (stage tears / re-alignments /
sectioning events between planes) and **smooth field-of-view drift** (the
imaged X/Y window creeping across the specimen through Z). Both are geometry
of the acquisition, not of the material microstructure — though the two can
masquerade as each other, which is the recurring caveat below.

Two whole-volume analyses of the codec
`/media/cheese/Ghc_data3/THX10/thx10codec/codec` (17 z-chunks x 814 patches
of 384 px x 48 slices, skipU epoch 300, source zarr `thx10cropzarr` level 0,
py38zarr env): sparse discontinuities (`find_discont.py`) and smooth global
drift (`measure_drift.py`). Zarr source/level/window/patch all read from each
chunk's `norm_params.json`.

## Gaps (find_discont.py)

Latent spike-screen of all patches, then raw-pixel verify of the top 50 per
chunk (GEOMETRIC = recoverable rigid shift, CONTENT = NCC collapse with no
shift, NOISE = neither).

```bash
CUDA_VISIBLE_DEVICES=0 python regis2/find_discont.py \
  --codec .../thx10codec/codec --chunks z0288-0336 --tex_min 0 \
  --out_dir <out>/find_discont_thx10/z0288-0336   # one call per chunk, ~5 min
```

**193 gaps from 850 verified: 173 CONTENT, 20 GEOMETRIC.** Clean score
separation (events 24-38+ vs ~2 floor); no chunk saturated top-50. Major
events (>=5 patches on one plane):

| z (gap z->z+1) | patches | character |
|---|---|---|
| 66 | 17 | CONTENT |
| 203 / 221 | 8 / 20 | CONTENT (one 16 px GEOMETRIC at 221) |
| 308 | 18 | CONTENT |
| 362-363 | 26 | CONTENT + 3 GEOMETRIC (to 45 px) |
| 553-554 | 19 | CONTENT |
| 636-638 | 35 | CONTENT, three adjacent planes |
| 779-790 | 11 | all GEOMETRIC, scattered, 3-28 px |

Each mid-stack event is a compact raft (~5x5 patches, ~2x2 mm of the
specimen) at one plane, different x/y per event — independent regional
changes, not a propagating column, rest of the volume clean. Only the stack
bottom (z ~ 779-816) has true rigid tears (the acquisition-misalignment
regime the TV-registration path could correct). Mid-stack CONTENT events are
undetermined between a genuine material transition (a real interface / phase
boundary in the specimen) and an acquisition artifact (focus/beam event,
debris) — the montages are the fastest route to telling them apart.

**`--tex_min 0` (texture gate off).** Latent-std barely separates background
(~1.197) from sample (~1.2-1.4); the default p5 gate dropped a real event
patch (010007, z=308) while the raw-pixel verify already rejects every
degenerate (screen scores to 665 on near-black planes -> all NOISE). Verify
also caught a 19-patch false consensus at z=791. `--topn` (default 50) is the
verify budget; re-run larger if the consensus histogram saturates on one z*.

**Outputs:** `gaps.csv` (running `x,y,z` list, patch-center level-0 coords,
z = earlier slice; `merge_gaps` dedups so re-runs are safe) at the codec
root; per-chunk `candidates/verified.csv` + `verify_montage.png`; whole-volume
`gaps_3d.png`.

## Drift (measure_drift.py)

Integrates the cross-patch **median** of the same adjacent-plane phase-corr
shift (3553 subgrid patches; textured subset ~90/plane) into a stack
trajectory — the smooth motion the spike-screen ignores.

**Steady, spatially uniform drift: dy ~ +0.09, dx ~ -0.02 px/slice,
accumulating to +63 px y / -14 px x over 816 slices** (~0.65 deg apparent
tilt at 8x z-spacing). Direction constant stack-wide, coherence R ~ 0.56,
row/col halves agree to ~0.01 px/slice — one rigid motion, not content noise.
Rate is ~100x below the tears; emerges only statistically.

**Selection pitfall:** background patches phase-corr-lock to ~0 on the
encoder's fixed-pattern structure, so any aggregation they dominate flatlines
at an artifactual zero (a per-chunk tex-percentile filter failed this way).
Fix: keep patches with absolute tex >= 1.21, the antimode of the bimodal
stack-wide tex distribution (background spike 1.188-1.200).

**Gauge caveat:** pairwise measurement can't separate true stage drift (the
FOV creeping across the specimen) from genuinely tilted microstructure (the
material's own features slanted vs the sectioning axis) — needs the
acquisition log or a fiducial to attribute. Either way it is benign at
per-slice scale, but any analysis assuming the column at (x, y) tracks the
same material through Z should account for the ~63 px total wander.

**Outputs:** `drift.png`, `drift_per_z.csv`, `drift_3d.png` (drifting-ROI
box), `drift_raw.npz` (raw per-patch series; re-aggregate via `--raw_npz`, no
GPU).

## Open items

- Pairs are within-chunk only, so gaps exactly at 48-slice boundaries
  (z = 47/48, ...) are blind spots — the 4-patch z=96 cluster is suspicious;
  a boundary-straddling pass would close it.
- GEOMETRIC gaps detected but not yet corrected — the bottom-of-stack cluster
  via latent-warp + TV-solve (see README) is the natural follow-up.
