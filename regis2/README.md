# regis2 — latent registration, round 2

Follow-ups to `registration/` (whose components are imported, not modified),
run on single 3D TIFF patches from the skipU300 grid
(`/home/cheese/workspace/Data/thx10/roiAdsp4`, 512 patches of 32×256×256 in
[-1,1]; default patch `007003005`). Outputs under
`/home/cheese/workspace/Output/regis2/`.

**Headline:** a single fused-lasso (TV) prior on the transform chain handles
tears, walks, and drift **without model selection** — large corruptions are
corrected, sub-detectable ones are left nearly untouched instead of being
damaged.

## Scripts

- **`perturb_options.py`** — previews the corruption models on one patch
  (ZX reslices, Z ×8 trilinear) and hosts their generators
  (`chunk_transforms`, `drift_transforms`, ...).
- **`run_walkbig.py`** — full corrupt → register-from-latent → enhance run on
  one patch, ending in a 7-column ZX tif (`{tag}_compare7.tif`: original |
  corrupted | original enh | corrupted enh | latent-reg enh | latent-reg
  input | latent-reg input enh). Key flags: `--perturb` (scenario), `--solver
  anchor|tv`, `--latent_interp bilinear|bicubic`, `--warp_space
  postquant|prequant`, `--drift_scale`, `--no_rot`; each non-default option
  suffixes the output tag so runs coexist.
- **`graph_tv.py`** — the TV graph solver (see Methods).
- **`tsne_corrupt.py`** — the skipU300 latent-t-SNE survey with a random
  `--frac` of patches corrupted before encoding, marked green
  (`--feat latent_mean|jump|both`, `--corrupt`, `--drift_scale`, `--thumbs`).
- **`build_flatfield.py`** — de-striping: estimates the stitch-tile
  vignetting gain field of a codec's source zarr (see `stitch.md`) into
  `{codec_root}/flatfield.npz`, consumed by `inference/decode_stack.py
  --flatfield`.

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

## Results (NCC of registered-enhanced vs original-enhanced, central 3/4)

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
See **`THX10_GAPS_DRIFT.md`**.
