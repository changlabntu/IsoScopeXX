# TTA edge-map pipeline — final procedure & settings

Turn a compressed-codec decode into a **3D boundary/edge map** using the
decoder's own test-time uncertainty. Three stages: (A) TTA + MC statistics,
(B) instant-alpha flood-fill base edge on the consensus mean, (C) rescue the
dim dots the flood misses. Demo/dev volume: `000001001` (thx10/roiAdsp4);
validated across all 10 roiAdsp4 volumes.

Model: registry `skipU` — MSclean (vqcleanM0aMS lineage), gen `ed023emsfpnu`,
epoch 300, nm=11g gamma 0.7 / gamma_lo −0.8. All decodes run in the default MC
mode (generator in `.train()`: batch-stat BN + live dropout); the 2D VQ codec
stays deterministic.

---

## Locked setting (annotation-seed = ACCURACY end)

For manual-annotation fine-tuning we favor **boundary accuracy over
continuity**: fragmented-but-well-placed edges are cheap and unambiguous to
hand-correct (bridge a visible gap), whereas over-connected/loose edges force
expensive, error-prone splitting and silently poison labels. So:

| stage | param | value |
|-------|-------|-------|
| A | TTA mask thr | −0.3 |
| A | MC draws     | 5  (4 TTA × 5 = 20 pooled variants) |
| B | flood TOL    | **−0.4**  (validated on the |grad| ridge — see note) |
| B | sigma        | 1.0 |
| B | island floor | 100 |
| C | shell thr    | **0.45**  (conservative rescue; drops fuzzy specks) |
| C | rescue floor | 8 |

(For maximum continuity instead, TOL −0.5 / shell 0.45 gives the fewest
components but over-traces boundaries and merges structures — not recommended
for annotation seeds.)

---

## Stage A — TTA + MC statistics  (`inference/inference_latent.py`)

    CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 \
    py38zarr python inference/inference_latent.py \
        --model skipU --exp <out> --source <tif-dir> --range 0 10 \
        --tta -0.3 --mc 5

Per volume, decodes 4 latent-TTA variants × 5 MC draws = 20 aligned outputs
and writes:
- `mean/{stem}.tif` — mean of the 20 denormalized variants (stable consensus).
- `std/{stem}.tif`  — per-voxel std of the binary masks `M > −0.3`.
(also `decode/`, `input/`, `codec/`.) ~12 s/volume on one GPU (~0.9 s fixed +
~0.42 s/variant + a CPU std reduction ~linear in N).

**How the TTA works.** The latent list from `latents_from_indices` (per-scale
`(B, C, Y, X, Z)`) undergoes each self-inverse op; each variant is decoded and
the SAME op realigns the output (decoder output shares the dim layout):

| variant | op (latent dims) | inverse |
|---|---|---|
| identity | — | — |
| transpose | `t.transpose(2, 3)` | same op |
| flipx | `torch.flip(t, dims=[3])` | same op |
| flipy | `torch.flip(t, dims=[2])` | same op |

Validity: net_g (`ed023emsfpnu`) is Y/X-isotropic (equal kernels/strides/
upsample), so the transpose is an exact pipeline symmetry; flips trivially so.
The transpose is skipped (with a notice) when latent Y != X. The encode side is
deterministic (GroupNorm, dropout 0), so indices are encoded once and every
variant decodes from the same codec.

**Modes.**
- **bare `--tta`** — std of the raw denormalized outputs (pre-gamma [-1, 1]).
- **`--tta THRESHOLD`** — variants binarized as `M > THRESHOLD`; std over the
  binary masks (segmentation-stability). For V variants the levels are
  `sqrt(k(V-k))/V`, k = 0..V/2.
- **`--mc N`** — repeats the TTA set N times; the generator runs in `.train()`
  (batch-stat BN + live dropout) so each pass is a fresh MC draw, pooling 4*N
  variants. Requires `--tta`; refuses `--eval` for N > 1.
- **`--eval --tta`** (N = 1) — pure transform sensitivity, no dropout.

`decode/` stays the identity decode and is reused as variant 1, so `--tta --mc N`
costs `4N − 1` extra decoder passes.

**Implementation traps.** `TTA_OPS` + the loop live in `inference_latent.py`;
ops act on dims 2/3 of both the 5D latents and the decoder output. Bare-vs-
threshold `--tta` uses `nargs='?'` with a `TTA_RAW = object()` sentinel —
argparse passes a STRING const through `type=float` (crashes), a non-string
object passes untouched. std/mean are on the denormalized scale, so thresholds
are in the [-1, 1] frame (-0.8 = gamma noise floor).

## Stage B — base edge: instant-alpha flood fill on the MEAN

Flood removes background that is BOTH below TOL AND connected to the volume
border (the "click seed"); the edge is the boundary of a filled solid → a
single, closed, 1-voxel surface (no double layers, no holes). Runs on the MEAN
(not a single decode) so the low TOL is stable, not grainy.

    D    = gaussian_filter(mean, 1.0)
    lab  = label(D < TOL)                       # TOL = -0.4
    bg   = components of lab touching the volume border
    fg   = ~bg                                   # objects + interior pores
    fg   = drop fg components < 100 voxels
    base = fg & ~binary_erosion(fg)              # 1-voxel boundary

## Stage C — rescue the dim dots the flood missed

    shell  = std > 0.45
    near   = binary_dilation(base, iters=2)
    label shell (26-conn); RESCUE a component if
        (not touching near) AND (size >= 8 voxels)
    union  = base | rescued            # final edge map

Rescued dots are raw std fragments (less boundary-accurate) — kept minimal at
shell 0.45. The clean `base` is the accuracy-critical product; treat `union`'s
extra dots as an optional recall overlay.

Stages B+C are a seconds-long scipy sweep on the cached mean/std — no model
rerun.

## Stages B+C — self-contained implementation

Reads a Stage-A output dir (`mean/`, `std/`) and writes `edge/{stem}.tif`
(uint8 0/255). Locked accuracy setting baked in as defaults.

```python
import glob, os
import numpy as np, tifffile as tiff
from scipy import ndimage

TOL, SIGMA, ISLAND_FLOOR = -0.4, 1.0, 100      # Stage B (flood on the mean)
SHELL_THR, RESCUE_FLOOR = 0.45, 8              # Stage C (dim-dot rescue)
ST3 = ndimage.generate_binary_structure(3, 3)  # 26-connectivity

def union_edge(mean, std):
    # Stage B: instant-alpha flood fill on the smoothed mean
    D = ndimage.gaussian_filter(mean, SIGMA)
    lab, _ = ndimage.label(D < TOL)
    border = np.unique(np.concatenate([
        lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
        lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    fg = ~np.isin(lab, border[border > 0])     # objects + interior pores
    lf, _ = ndimage.label(fg, ST3)
    sz = np.bincount(lf.ravel()); sz[0] = 0
    fg = (sz > ISLAND_FLOOR)[lf]
    base = fg & ~ndimage.binary_erosion(fg)    # 1-voxel boundary
    # Stage C: rescue std-shell components the flood missed
    shell = std > SHELL_THR
    near = ndimage.binary_dilation(base, ST3, iterations=2)
    labs, nc = ndimage.label(shell, ST3)
    covered = set(np.unique(labs[near]).tolist()) - {0}
    szs = np.bincount(labs.ravel())
    ids = [i for i in range(1, nc + 1)
           if i not in covered and szs[i] >= RESCUE_FLOOR]
    union = base | np.isin(labs, ids)
    return base, union                         # base = accuracy product; union += dim dots

def run(src, out):                             # src = Stage-A dir with mean/ std/
    os.makedirs(os.path.join(out, 'edge'), exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src, 'mean', '*.tif'))):
        stem = os.path.basename(p)[:-4]
        mean = tiff.imread(p).astype(np.float32)
        std = tiff.imread(os.path.join(src, 'std', stem + '.tif')).astype(np.float32)
        _, union = union_edge(mean, std)
        tiff.imwrite(os.path.join(out, 'edge', stem + '.tif'),
                     union.astype(np.uint8) * 255)
```

For annotation seeds return `base` (drop the rescue) — see Notes.

---

## Notes & validation

- **Gradient-snap tested, not adopted.** Snapping the iso-contour to the
  |grad mean| ridge (Canny-style non-max suppression along the gradient) moved
  the edge only ~1.4 vox and raised mean|grad| 1.02× (negligible) while adding
  noise — confirming the −0.4 iso-contour already sits on the true boundary
  ridge. Skip it unless TOL is poorly chosen.
- **std map is an annotation aid** — hand it to annotators to flag where the
  model was uncertain (the gaps / ambiguous boundaries worth their attention).
- **std shell as the alpha input = WRONG** (double-layer edges): the shell is a
  thin band, so its boundary has two faces. Always flood the mean/decode
  (a filled solid), never the std shell.

## Tunables (grouped by stage)

| stage | param | value | role |
|-------|-------|-------|------|
| A | TTA mask thr | −0.3 | intensity level std traces; couple with flood TOL |
| A | MC draws     | 5    | more = smoother stats, linearly slower |
| B | flood TOL    | −0.4 | background cut; lower = more dim reach, over-traces past −0.5 |
| B | sigma        | 1.0  | pre-flood smoothing; suppresses MC grain |
| B | island floor | 100  | min base-foreground component; keeps base clean |
| C | shell thr    | 0.45 | `std >` cut for the rescue shell; higher = fewer, cleaner dots |
| C | rescue floor | 8    | min dim-dot size added back |

Cross-stage: lower the Stage-B island floor before loosening the Stage-C shell
thr — it shifts structure from fuzzy rescued-fragment into the clean base.
TTA mask thr (A) and flood TOL (B) should move together; mismatching them (base
traces a different level than the shell) breaks coherence.
