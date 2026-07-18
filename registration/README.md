# Latent-based Z-slice registration for 3D materials-imaging volumes

**Domain:** this is a materials-science 3D volume-imaging problem. The data are
anisotropic 3D scans of material samples (e.g. THX10 tomography of the sample0
volume), the same volumes IsoScope encodes into a VQ latent/codec. A recurring
defect in such stacks is **inter-slice misalignment**: successive Z-slices of
the material volume are shifted/rotated relative to each other — from
stage/section drift during acquisition (the serial-section scenario) — which
corrupts the through-Z microstructure (grain boundaries, voids, inclusions look
jagged or torn across slices).

**Question:** can the IsoScope VQ **latent** re-align such a material-volume
stack after each slice has been hit by a random affine transform — and can it
do so directly from the stored compressed **codec**, without the raw scan? Yes
to both.

**Why it works:** the encoder is strictly 2D per-slice (`models/MSclean.py`
`vol_to_slices`), so each XY slice of the material volume owns an independent
`(4, Y/8, X/8)` latent plane, and affine geometry survives the convolutional
encoder (rotation → same rotation; translation `t` px → `t/8` latent px). So
slices of the sample align *in latent space*, with no raw data needed for the
codec path.

## Pipeline (env `py38zarr`; `$OUT=/home/cheese/workspace/Output`)

```bash
# 1. take a well-registered ROI of the sample0 material volume and simulate the
#    defect: a per-slice similarity random-walk (known ground-truth transforms)
python registration/perturb.py --nz 120 --size 1024 --name full1024
# 2. recover transforms + re-register (--batch 1 for 1024² to fit 24 GB)
python registration/register.py --dir $OUT/registration/full1024 --feat latent --batch 1
python registration/register.py --dir $OUT/registration/full1024 --feat codec  --batch 1
python registration/register.py --dir $OUT/registration/full1024 --feat pixel
python registration/register.py --dir $OUT/registration/full1024 --method xcorr
# 3. score vs ground truth + figures
python registration/evaluate.py --dir $OUT/registration/full1024
```

## Method (`register.py`)

Per slice pair `(z, z+o)`, `o ∈ --pairs` (default `1,2,4`):
1. **Coarse** — channel-summed FFT phase correlation → subpixel translation.
2. **Refine** — batched Adam over similarity params (rot, tx, ty, log-scale),
   differentiable warp (`affine.py`) under masked L1. Regularized (`--reg_t`,
   `--reg_rs`) so the fit can't absorb real Z structural change into the
   transforms.
3. **Graph solve** — pairwise measurements → absolute chain `M_z` by
   least-squares, gauge `M_0 = I` (`affine.solve_graph`). `--anchor` (0.05)
   damps the coherent drift that pairwise data leaves unobservable in long
   chains.
4. **Apply** — warp each slice by `M_z⁻¹` (bicubic, full res).

**Feature modes** (`features.py`): `latent` = continuous pre-VQ encoder output;
`codec` = latent rebuilt from stored codebook indices alone
(`Engine.latents_from_indices`) — registration *from the compressed codes*;
`pixel` = 8×-pooled raw slices (baseline). `--method xcorr` = the repo's naive
translation-only baseline (`utils/alignments.py`).

## Results (defaults reg_t 5, reg_rs 10, anchor 0.05, pairs 1,2,4)

full1024 — 120 slices, 1024² (pristine adj-slice NCC 0.925):

| method | trans err px | rot err ° | NCC vs orig | adj-slice NCC |
|---|---|---|---|---|
| corrupted (none) | 28.2 | 1.24 | 0.38 | 0.851 |
| **latent** | 9.8 | 0.34 | 0.62 | 0.924 |
| **codec** | 10.2 | 0.45 | 0.61 | 0.925 |
| pixel | 9.5 | 0.32 | 0.66 | 0.918 |
| xcorr | 28.1 (fails) | 1.24 | 0.38 | 0.856 |

small512 — 40 slices, 512² (pristine adj-slice NCC 0.872): latent **5.9 px /
0.19°**, codec 6.3 / 0.17, pixel 6.3 / **0.49**, corrupted 13.3 / 0.77.

**What holds up:**
- **Registration works**: misalignment of the material volume drops ~3×
  (translation) / ~4× (rotation); adjacent-slice coherence of the
  microstructure returns to the pristine level. `xcorr` fails (can't handle
  rotation).
- **Codec ≈ latent**: a misaligned material stack can be re-registered from the
  compressed codes alone — no raw scan, no re-encode. This is the headline;
  pixels can't do it.
- **Latent > pixel only in the hard regime**: rotation at small field of view
  (512²: 0.19° vs 0.49°, robust to `--noise/--gain`). At 1024² pixels catch up
  — 4× more image evidence makes it easy for any feature. Translation is a tie
  everywhere.

**Caveats:**
- Absolute error is dominated by accumulated *coherent* drift — pairwise
  registration can't fully separate true section motion from real structure
  drifting through Z ("aligning to a banana"). Gauge-free neighbor-pair error
  is sub-pixel (~0.65 px). Report absolute error alongside the identity-run
  bias floor (~1 px / 0.17°).
- Efficiency of `codec` is about **data flow, not FLOPs**: register from a few
  MB of stored codes without the raw volume or the encoder pass. The latent
  feature is 4 channels (4× the pixel plane), so the aligner inner loop is not
  cheaper.
- The decoded-isotropic-volume smoothness claim is **not yet tested** — only
  input-resolution reslices are shown (`reslice.png`).

## Outputs / verification

Per dir: `original/corrupted.npy`, `gt_transforms.json` (perturb) →
`registered_{label}.npy`, `recovered_{label}.json` (register) → `metrics.csv`,
`errors.png`, `reslice.png` (evaluate; `reslice.png` = XZ/YZ jagged→smooth).

Sanity: synthetic self-tests (warp/compose/phase-corr/graph-solve, end-to-end
0.045 px); identity run (`perturb --rot 0 --trans 0 --scale 0`) measures each
feature's content-change bias floor — rerun when touching the loss.
