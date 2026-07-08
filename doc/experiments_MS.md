# MS (multi-scale) model experiments

Consolidates the former `MS_refine.md`, `MS_refine2.md`, and `MS_summary.md`.

## Setup

All runs on THX10SDM20xw/roiD (192×192×24 → 192³ isotropic, `num_scales 4`), tracked in
`$LOGS/thx-MS.db`. The MS family: `vqcleanM0aMS` variants with progressive output heads
(out0 192³ / out128 96³ / out64 48³) tapping the 3D decoder.

Metrics (↓ better):
- `val_lpips_pred` — XY-vs-YZ isotropy of the prediction (trilinear do-nothing baseline: 0.634)
- `val_kid` — realism of synthesized YZ slices vs real XY

Design constraints held across all iterations: (a) **one decode pass** producing all scale
outputs; (b) **~unchanged parameter count**.

## Results (final-window means)

| run | change vs parent | val_lpips_pred | val_kid | verdict |
|---|---|---|---|---|
| MS | baseline vqcleanM0aMS (lr 5e-4, adv_ms 0) | 0.589 | 5.71 | reference |
| MSfpn | strict scale→head visibility (restricted trunk) | 0.627 | 7.31 | NEGATIVE (stopped @104ep) |
| **MSskip** | skips as zero-init reinforcement + pyr_detach / adv_ms 0.5 / lr 2e-3 | **0.577** | 6.39* | WIN vs MS; **LPIPS champion** |
| **MSskipE** | + EMA-at-eval + real-data coarse L1 (l1c) | 0.616 | **4.97** | SPLIT — **KID champion** |
| MSskipP-band | + band-limited dense projection L1 | 0.633 | 7.01 | NEGATIVE (checkerboard null space) |
| MSskipP-lse | + soft-max projection, random phase | 0.635 | 5.62 | NEGATIVE (mild, clean attribution) |
| MSskipE-ema999 | skipE with EMA horizon 65 → 6.5 epochs | between parents on both axes | | PARTIAL — A1 ablation |

*Raw-weight KID is noisy (±1 between windows); EMA runs report much smoother curves.

## The narrative

### 1. Architecture (MSfpn → MSskip): add paths, never remove them

MSfpn hard-routed VQ scales to matched output heads: the trunk received only the coarse
partial sum, with `quants[-2]`/`quants[-1]` injected laterally after the out64/out128 taps.
This starved the full-res output on both diagnosed mechanisms:

- **Non-detached pyramid loss blurred out0** — with `--lamb_pyr 1.0`, no `--pyr_detach`, and
  `--adv_ms 0`, the two-way L1 pulled the pooled full-res output toward what coarse codes can
  express, unopposed by any coarse adversarial loss.
- **The finest scale got almost no 3D processing** — the 24×24 scale carries 576 of 765
  tokens/slice (most in-plane detail) but received one 3D stage instead of eight; its
  injection was per-slice 2D content trilinearly stretched 24→96 in Z, arriving XY-sharp but
  Z-smooth — exactly what the XY-vs-YZ isotropy metric punishes.

Matched-epoch data confirmed it (epochs 55–73: LPIPS 0.637 vs 0.602, barely beating the
0.634 trilinear baseline; KID 7.67 vs 5.94; gap not closing). Stopped at ~104 epochs.

**MSskip** recast the same lateral connections as **zero-init reinforcement skips on a
full-sum trunk** (`inject_convs` zero-initialized, ControlNet-style, so the model is
numerically identical to `vqcleanM0aMS` at step 0 — no cold-start handicap; skips can only
learn to help). Combined with the loss fixes (`--pyr_detach --adv_ms 0.5`, lr 2e-3), it
overtook MS on both metrics by ~epoch 70 (epochs 96–121: LPIPS 0.580 vs 0.598, KID 4.73 vs
5.75). Caveat: lr and loss config changed together with the architecture; the
`MS + pyr_detach/adv_ms/lr 2e-3` control was never run.

**Lesson: add paths, never remove them; zero-init everything so training can only go up.**

### 2. Evaluation weights (MSskipE): EMA is real but double-edged

`--use_ema` had been silently dead code (the LitEma import path didn't exist). MSskipE wired
EMA into validation and epoch checkpoints (`LitEma` from `ldm/modules/ema.py`,
`on_validation_start/end` + a `training_epoch_end` wrapper) and added real-data coarse L1
(`--lamb_coarse`, logged `l1c`): out128/out64 Z-projections vs XY-avg-pooled `oriX`.

Result: best KID of any run (4.97, still falling), but LPIPS plateaued 0.04 above MSskip.
Diagnosis: at decay 0.9999 the ~65-epoch averaging horizon mutes the high-frequency YZ
texture the isotropy metric rewards, even as it cancels GAN oscillation artifacts (the KID
win). Weight averaging ≠ output smoothing — except when it averages over genuinely
different texture solutions, which a 65-epoch horizon does.

The **ema999 ablation** (decay 0.999, ~6.5-epoch horizon) confirmed a genuine trade, not a
bug: it recovered about half the LPIPS and lost most of the KID gain, landing between its
parents on both axes. Proposed way to hold both: long-horizon EMA + also-checkpoint raw
weights (dual eval).

### 3. Projection forward model (MSskipP ×2): closed — sparse `max` stays

Both alternative L1-anchor forward models failed with single-variable attribution
(run lines changed only `--l1how` vs skipE):

- **`band`** (Gaussian-low-pass both prediction and observation along Z, dense L1 — full
  agreement in the measured band, zero constraint above): the freed high-frequency band
  filled with axis-aligned CHECKERBOARD. The ConvTranspose trunk wants to emit it, the
  stride-2 patch discriminator aliases it into invisibility, and the sparse `max` projection
  had secretly been the only damping. Spectrally stable over 130 epochs (not a transient).
  Band's dense anchoring remains theoretically right but is only viable after an
  anti-checkerboard generator.
- **`lse`** (bias-corrected soft-max projection + random window phase, `--lse_tau`): mild
  but clear regression — the softly-averaged constraint adds diffuse smoothing pressure.

### 4. Baseline declaration (2026-07-07)

**`vqcleanM0aMSskipE` is the baseline model going forward.** All future iterations chain
from its file and are measured against its numbers. Rationale: it contains everything
validated — full-sum trunk + zero-init skips (proven win), EMA-at-eval (proven KID lever,
tunable via `--ema_decay`), real-data coarse supervision (togglable via `--lamb_coarse`) —
and nothing architecturally suspect; both debated ingredients are command-line switchable
without touching the file.

**Baseline config** = the skipE run line
(`--lamb 5 --lr 0.002 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --l1how max`,
`--netG ed023emsfpn`, `num_scales 4`), with two knobs explicitly marked OPEN:

- `--lamb_coarse` — suspected ~0.017 LPIPS cost (the half of skipE's regression the ema999
  ablation could not attribute to the EMA horizon). A2 run (`--lamb_coarse 0`) pending; if
  confirmed, the canonical config flips to `lamb_coarse 0`.
- `--ema_decay` — a genuine trade: 0.9999 (≈65-epoch horizon) buys the KID win at LPIPS
  cost; 0.999 recovers about half the LPIPS and loses most of the KID gain. Long-horizon
  EMA + also-checkpoint raw weights (dual eval) is the proposed way to hold both.

**Long-training update (Kubeflow, ~450 epochs, ~114k steps):** skipE's KID keeps descending
far past where the short brcb runs stopped (window mean ≈3.9, single epochs 3.0) and LPIPS
≈0.607 is creeping toward MSskip's raw-weight 0.577. The standings table was measured at
≤164 epochs and understates the EMA configurations at scale — epoch indices are
machine-dependent (258 vs 153 vs 1073 steps/epoch); compare runs by optimizer steps or
wall-clock, not epochs.

## Standings and queue

- **MSskip holds LPIPS (0.577); MSskipE holds KID (4.97).** Goal state: one run, both crowns.
- **Anti-lattice line (see doc/research_artifact_suggestions.md):** `MSskipU` =
  skipE + `--netG ed023emsfpnu` (Z3 resize-conv trunk; removes the ConvTranspose
  alias-lattice source) and `MSskipUB` = MSskipU + `--netD patchblur_16` (BlurPool;
  removes the D's stride-2 blindness). Both reuse the skipE model file (deltas are
  CLI-only); fresh trunk weights → compare at matched optimizer steps, not step-0.

  **MSskipU interim verdict (fuse dataset, 2026-07-08, step ~10.4k / ep ~275,
  still running):** the lattice fix WORKS — `val_lat_p2diag` ≈ 1.3 / `p4diag` ≈ 1.1
  (clean = 1; skipE outputs measured 17–94×), mild falling residuals on the
  axis probes (p2a ≈ 2.8, p2z ≈ 2.2); GIFs show continuous filaments, no beaded
  diagonal chains. LPIPS at parity with fuse-skipE at matched steps (0.467 vs
  0.466), currently 0.460 and improving. **KID regressed: 5.42 vs 3.13 at matched
  steps, near-flat.** Reading (per Schwarz et al.): each upsampler biases a
  different spectral error — ConvTranspose → checkerboard, resize-conv → too
  LITTLE high frequency; with the lattice gone the G has nothing to fake texture
  with, and the stride-2 D is as blind to an HF deficit as it was to the HF
  excess (EMA compounds the smoothing). Sharpens the pair hypothesis: UB
  (BlurPool D, queued) is the mechanism that lets the D demand real texture;
  spectral-D/FFL behind it. Side effect: with the lattice gone, the `band`
  projection retest and Z2 noise injection are live options again.
- If ema999-style tuning fails → A2 (`--lamb_coarse 0.5/0`) is the remaining skipE suspect.
- Architectural candidates still untested (from the direction catalog below): **Z1**
  zero-init Z-mixing latent conv (most mechanism-aligned for isotropy — slices never
  interact before `up3`); **Z3** anti-checkerboard upsampling (evidence-backed by the band
  failure; prerequisite for ever retesting band's dense anchoring).
- Optional hygiene: the never-run control `MS + pyr_detach/adv_ms/lr 2e-3` to attribute
  MSskip's original win between architecture and loss config.

## Untested direction catalog

Mechanism analysis — remaining architectural weaknesses against KID (unrealistic slice
texture) and isotropy-LPIPS (directional asymmetry):

1. Slices are encoded independently by the 2D VQ encoder and nothing mixes Z until `up3` —
   the trunk must invent all cross-slice coherence three layers deep.
2. ConvTranspose3d (k4, s2) upsampling is checkerboard-prone (empirically confirmed by the
   band failure).
3. The generator has no stochastic source for high-frequency texture; the deterministic
   per-slice latent can't supply Z detail, so YZ planes stay smoother than XY.

Candidates, roughly by expected quality-per-effort:

- **Z1. Zero-init Z-mixing latent stage** (~768 params). After `decoder.conv_in`:
  `x = x + zconv(x)` with a depthwise Conv3d kernel (1,1,3), zero-init — step-0 identical to
  parent. The cheapest possible cross-slice communication, at latent resolution.
- **Z3. Anti-checkerboard upsampling.** The generator ctor already has `use_upsample`
  (Upsample+Conv3d instead of ConvTranspose3d) but the registry never passes it → new netG
  file with it defaulted on. Slightly fewer params. Caveat: all trunk weight shapes change →
  fresh training, no step-0 equivalence with any prior run. **Now the most-motivated next
  architectural iteration** — implicated in both skipE's LPIPS ceiling and band's failure.
- **Z2. Stochastic noise injection (StyleGAN-style)** (~224 params). Additive Gaussian noise
  with learned per-channel scales (zero-init) after each up stage; eval uses noise=0.
  Targets the YZ-smoothness half of the LPIPS gap.
- **F3. Structural projection losses** — `--lbm_ms_ssim` (currently 0) or Laplacian-pyramid
  L1 on the Z-projection.
- **F4. Discriminator feature matching** — L1 between D intermediate features of real vs
  generated slices; needs `patch_16` to expose features.
- **S1. Residual-over-trilinear output** — heads predict a delta over the trilinear upsample;
  guarantees the data-fidelity floor.
- **S2. Bottleneck attention at 24³** — one attention block on the trunk input (~0.26M
  params, +0.9%); global 3D context for long-range structure.
- Post-injection Conv3d blocks at the skip sites — designated
  fallback, deferred since the full-sum trunk can suppress a harmful skip.

Explicitly out (violate constraints): multi-pass schemes (dsp-phase consistency, partial-sum
decodes); extra per-orientation discriminators (the six-way D already dominates step cost).

## Method notes (why attribution stayed clean)

- One change per iteration; each iteration is its own model file
  (`models/vqcleanM0aMS{fpn,skip,skipE,skipP}.py`), snapshot-verified against running jobs.
- Zero-init discipline: every added module starts contributing nothing, so each variant
  begins at its parent's exact quality — negatives cost compute, never confusion.
- EMA-evaluated runs need matched-window comparison (the average lags the raw model by
  ~half the horizon); raw-weight runs need window means (KID noise ±1).
- Compare at matched epochs/steps — fresh modules and lr changes create early-epoch
  handicaps that systematically flatter the incumbent.
- `l1` values are not comparable across runs with different `--l1how` (different functionals).
