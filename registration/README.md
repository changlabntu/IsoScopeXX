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

## Enhanced-volume comparison: latent-space registration → decode

Closes the caveat above ("decoded-isotropic-volume smoothness not yet
tested"), and tests a NEW way of applying the recovered transforms. Old way:
the latent only *estimates* `M_z`, which is then applied to the **pixels**
(`register.py` → `registered_latent.npy`). New way (`enhance.py
--transforms`): encode the corrupted stack once, warp each per-scale latent
plane by `M_z⁻¹` directly **in latent space** (translation ÷ 8;
rotation/scale are resolution-independent in `affine.py`'s centered
convention), then decode the registered 3D latent straight through `net_g`
into the 8× Z-super-resolved volume — registration and enhancement without
ever warping (or needing) the pixel stack. Both ways reuse the same
`recovered_latent.json`, so the comparison isolates *where* the warp happens.

```bash
python registration/enhance.py --dir $OUT/registration/full1024 --stack corrupted.npy --self_test
python registration/enhance.py --dir $OUT/registration/full1024 --stack original.npy          --out original_enh
python registration/enhance.py --dir $OUT/registration/full1024 --stack corrupted.npy         --out corrupted_enh
python registration/enhance.py --dir $OUT/registration/full1024 --stack registered_latent.npy --out registered_latent_enh
python registration/enhance.py --dir $OUT/registration/full1024 --stack corrupted.npy \
    --transforms recovered_latent.json --out latentreg_enh     # NEW way
python registration/evaluate_enhanced.py --dir $OUT/registration/full1024
```

All four volumes go through the identical encode → tiled-decode path
(full-res per-slice encode, 32²-latent × 24-Z tiles → 256²×192 out,
deterministic `train_mode=False`), so tiling seams (X/Y multiples of 256, Z
multiples of 192) are a shared constant. Outputs under `{dir}/enhanced/`:
`{original,corrupted,registered_latent,latentreg}_enh.npy` (960×1024×1024
float16, pre-gamma [-1,1]) + per-run settings json, `metrics_enhanced.csv`,
`reslice_enhanced{,_zoom}.png` (6 columns: original/corrupted trilinear +
the four enhanced; reference = `original_enh`).

Notes: a bilinear latent warp stays in-distribution for the decoder
(`encode()` itself bilinearly upsamples coarse code planes), but the encoder
only approximately commutes with rotation/subpixel shift — that gap is what
this comparison measures. Latent 0 is not background (`--fill background`
encodes a constant −1 slice instead); overlap-blended decoding to hide tile
seams is future work.

---

# Round 2 — one prior for every corruption + real-data application

Follow-ups to the pipeline above (formerly the separate `regis2/` package),
run on single 3D TIFF patches from the skipU300 grid
(`/home/cheese/workspace/Data/thx10/roiAdsp4`, 512 patches of 32×256×256 in
[-1,1]; default patch `007003005`). Outputs remain under
`/home/cheese/workspace/Output/regis2/` (the artifacts predate the merge).

**Headline:** a single fused-lasso (TV) prior on the transform chain handles
tears, walks, and drift **without model selection** — large corruptions are
corrected, sub-detectable ones are left nearly untouched instead of being
damaged.

## Code map (post-merge)

- **`graph_tv.py`** — the TV graph solver (see Methods); reachable from
  `register.py` via `--solver tv`.
- **`perturb.py`** — besides the step-1 CLI, hosts all corruption generators:
  `sample_transforms` (walk), `chunk_transforms` (inter-chunk tear),
  `drift_transforms` (smooth sinusoid), and the `apply`/`upz` warp helpers.
- **`experiments/run_walkbig.py`** — full corrupt → register-from-latent →
  enhance run on one patch, ending in a 7-column ZX tif
  (`{tag}_compare7.tif`: original | corrupted | original enh | corrupted enh
  | latent-reg enh | latent-reg input | latent-reg input enh). Key flags:
  `--perturb` (scenario), `--solver anchor|tv`, `--latent_interp
  bilinear|bicubic`, `--warp_space postquant|prequant`, `--drift_scale`,
  `--no_rot`; each non-default option suffixes the output tag so runs
  coexist.
- **`experiments/tsne_corrupt.py`** — the skipU300 latent-t-SNE survey with a
  random `--frac` of patches corrupted before encoding, marked green
  (`--feat latent_mean|jump|both`, `--corrupt`, `--drift_scale`, `--thumbs`).

## Corruption scenarios (per-slice similarity transforms)

| scenario | model | magnitude |
|---|---|---|
| walk / walk_big | random walk, accumulating per slice | rel. ±0.5°/±3 px (big: ±1°/±6 px) |
| drift (×1, ½, ⅓) | smooth sinusoid, one cycle | ±10/5/3.3 px, ±1/0.5/0.33° |
| walk_drift | walk + drift combined | ~14 px / 1.2° accumulated |
| chunk | single step at Z/3 (inter-chunk tear) | one walk_big-size jump (~7 px / 0.9°) |

## Methods tried

- **Registration estimate** (fixed throughout): latent-plane phase-corr →
  regularized refine → graph solve (`register.register_features`).
- **Transform application**: pixel-warp then re-encode (old way) vs
  **latent-warp then decode** (new way), latent-warp variants bilinear →
  bicubic, post-quant → pre-quant (`--warp_space prequant` warps the
  continuous encoder output, then quantizes; reimplements `MSclean.encode`'s
  VAR loop, verified ≡ to 5e-7).
- **Graph solver**: **anchor** (`affine.solve_graph`, L2 pull of absolute
  transforms toward identity) vs **TV** (`graph_tv.py`, group fused-lasso L1
  on slice-to-slice increments, IRLS, warm-started from the anchored solve;
  translation and rot/scale as separate penalty groups, tv=4 px /
  tv_lin=2 px-equiv; linear data rows rescaled to px-equivalents — without
  that, rotation evidence is ~100× weaker than any px-denominated penalty
  and gets flattened).
- **Detection** (t-SNE QC from stored codes): Z-pooled content feature vs
  adjacent-plane "jump" feature (phase-corr |t| + NCC stats).

## Round-2 results (NCC of registered-enhanced vs original-enhanced, central 3/4)

1. **Latent-warp ≈ pixel-warp** in every regime (within ±0.01); best latent
   recipe is **bicubic + pre-quant**; the tiny residual coherence gap is the
   stride-8 warp grid itself, not encoder rotation non-equivariance
   (`--no_rot` ablation).
2. **Anchor solver**: good on walk-class (0.26 → 0.53), but *harmful* below
   its ~1 px/slice bias floor (drift ⅓: 0.815 → 0.61; chunk: 0.688 → 0.61 —
   it repairs the local tear but injects wander into untouched slices).
3. **TV solver — one setting, all regimes**: chunk 0.688 → **0.82**
   (transform error 0.87 px, tear kept sharp, no wander), drift ⅓
   near-harmless (**0.79** vs 0.815 do-nothing), walk_drift **0.61** (best
   yet). Increments below the noise floor snap to zero; genuine jumps
   survive at full size.
4. **Detection**: the content feature is alignment-invariant (0/51 corrupted
   found — good for microstructure surveying, useless for QC); the **jump
   feature** finds walk-class corruption at **51/51** (precision@51),
   degrading through smooth drift (76/47/24% at ±10/5/3.3 px) with a
   ~1 px/slice floor set by weak-texture phase-corr noise — matching the
   regime where correction stops being beneficial (detectability and
   treatability track each other).

Caveats: sub-noise-floor smooth drift is "left alone", not corrected (0.786
vs 0.815 is still slightly net-negative); rotation is the weakest component
(~0.5–1° hallucinated rotation leaks through the refine; cheap at these
magnitudes); evidence is one patch × one seed per regime with tv/tv_lin
chosen on the same cases — a patches × seeds × regimes sweep with fixed
hyperparameters is the missing experiment for a stronger claim; and no
pairwise solver can recover the unobservable smooth global gauge.

QC loop closed end-to-end from the codec: detect misaligned patches (jump
feature) → recover transforms (`register.py` + TV solve) → correct in latent
space and decode enhanced (`enhance.py`).

**Real-data application:** `find_discont.py` + `measure_drift.py` swept the
whole THX10 volume codec — 193 verified gaps (seven major event planes,
geometric tears only at the stack bottom) and a steady +63 px global drift.
See **`THX10_GAPS_DRIFT.md`** (and `stitch.md` for the stitch-tile grid found
in the overlays).
