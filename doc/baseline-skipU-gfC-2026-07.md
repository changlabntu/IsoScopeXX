# Baseline decision: skipU + gfC is the new roiD-line baseline (2026-07-15)

**Decision: yes — skipU (resize-conv generator, plain patch discriminator) in the gfC
intensity regime (γ0.7 / floor −0.8) is the new baseline for the roiD line**, replacing
skipE-gf as the declared reference config. This is supported by the full evidence chain
below, subject to the scope conditions in §4 — most importantly: the gamma *floor* and the
γ0.7 exponent are roiD-specific and do NOT transfer to fuse, and skipU checkpoints must
always be screened by `val_lat`.

Reference run: **`b63909b3`** (`skipU_gammaC`, MLflow experiment 12, prj
`thx10/vqcleanM0aMSskipU/roiD192gfC/max5skip4`, git `83f769a`). Still RUNNING as of
2026-07-15, epoch 1116 / step 139.6k.

## 1. The baseline config

```bash
CUDA_VISIBLE_DEVICES=... NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr \
  --prj thx10/vqcleanM0aMSskipU/roiD192gfC/max5skip4 --env brcb \
  --dataset THX10SDM20xw/ --direction roiD/ \
  --nm 11g --gamma 0.7 --gamma_lo -0.8 \
  --cropsize 192 --cropz 24 --dsp 1 --lamb 5 \
  --models vqcleanM0aMSskipE --num_scales 4 --lr 0.0005 \
  --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 \
  --tracking_uri https://mlflow.ntugarylab.dpdns.org/
```

Deltas vs the old skipE-gf baseline: `--netG ed023emsfpnu` (resize-conv, was `ed023emsfpn`
ConvTranspose) and `--gamma 0.7` (was 0.5). The model class is still `vqcleanM0aMSskipE` —
"skipU" is expressed purely through `--netG`; filter MLflow by `prj`, not the models tag.

## 2. Evidence (three legs, each single-variable)

1. **Architecture — the decoupling run** (`a46f1425`, §6 of
   `experiments_MS_skipE_skipU_gamma_2026-07.md`): resize-conv netG alone puts the diagonal
   upsampling lattice at the clean floor (p2diag/p4diag ≈ 1.1–1.8 vs skipE's 8× peaks) with
   no BlurPool needed, and dropping BlurPool recovers about half of skipUB's KID regression.
2. **Intensity regime — the gamma ladder** (§7): on identical skipU architecture, γ0.7/−0.8
   beats γ0.5/−0.8 on every axis at matched steps (stable ~7% LPIPS margin vs decaying,
   half the KID, half the p2z stripe), and γ0.75/−0.85 bounds the optimum from above
   (late band-limited collapse: spec→0.006, KID→14.6).
3. **Longevity check (2026-07-15, full-history pull at ep 1116)**: gfC did **not** repeat
   gfB's collapse at 3× the §7 snapshot horizon — spec_xy_hi holds ~0.05–0.07, KID ~1.6,
   LPIPS margin averages 7.0% over the last 20 validations. Checkpoint-level head-to-head:
   gfC@ep200 beats or ties *every* saved checkpoint of the γ0.5 sibling on all
   regime-comparable axes (p2z 2.3 vs ≥4.5, p2diag 1.38 vs ≥1.6, p2a tie), giving up
   nothing measurable.

## 3. Checkpoint policy for the baseline

`epoch_save 100`; each row = mean of the 5 validation points nearest the checkpoint.

| ckpt ep | p2diag | p4diag | p2z | p2a | kid | lpips margin | note |
|---|---|---|---|---|---|---|---|
| **200** | 1.38 | 1.09 | **2.3** | 3.2 | 1.48 | 6.0% | **safe default** — pre-rough-patch, pre-disc |
| 1100 | 2.08 | 1.22 | 4.5 | 4.1 | 1.47 | 6.9% | best late; post-disc-activation regime |
| 500–1000 | — | — | up to 8 | up to 12 | — | down to 1.6% | **avoid** — rough-patch window |

- **ep 200 is the recommended inference/baseline checkpoint** until the post-disc regime
  proves itself.
- The run traversed a rough patch at steps ~60k–110k (LPIPS margin went negative at ~100k,
  p2z ≈ 12–14) and recovered. skipU-family equilibrium swings are real: **never take a
  checkpoint without screening `val_lat` around it.**

## 4. Scope conditions & standing caveats

1. **Floors and exponents are per-dataset.** roiD: 0.7/−0.8. **Fuse stays at 0.5/−0.9** —
   fuse foreground is dim-skewed, so the highlight-headroom motive for γ0.7 barely applies
   and raising γ would cost faint-end lift. Backed by measured noise stats (roiD noise
   median −0.85; fuse zcube −0.964, med+4σ edge −0.909).
2. **The p2z stripe is reduced, not eliminated** (≈2–5 at good checkpoints vs 11–17 for
   γ0.5-skipU). BlurPool (skipUB) suppresses it further but costs KID; skipUB remains the
   fallback if a downstream use is stripe-sensitive. The frozen-feature-D / notch-D research
   directions (research-MS-plan) still target the all-three operating point
   (lattice-gone + KID-recovered + stripe-suppressed).
3. **Regime boundary at step 125,124 (~ep 1000): the vendored VQGAN internal 2D
   discriminator (`disc_start`) activated** — the first run in the campaign to reach it.
   Post-activation val metrics improved (lpips_pred 0.468→0.446 mean, p2z 12.6→5.6,
   spec_recon_hi rising to 0.35), **but the `ae` generator-adversarial term is climbing
   unboundedly (0.08 → 8.4 over 14k steps) while disc loss falls (1.05 → 0.67)** — classic
   pre-destabilization signature. Watch it; do not compare post-125k metrics of any long
   run against short-run siblings without noting the regime change.
4. **Sharpness deficit is inherent to the baseline generator**: spec_xy_hi ≈ 0.06–0.09 vs
   skipE's 0.40, and the notch test showed skipE's HF advantage is genuine broadband, not
   just checkerboard. Any claim for this baseline must say "the diagonal lattice is
   eliminated", not "artifact-free" and not "no sharpness cost".
5. Regime caveat as always: gfC metric scales (lpips_tgt 0.4763) differ from gf (0.5211)
   and from nm=00. Only self-normalized `val_lat_*` ratios and margins-below-own-baseline
   travel across regimes.

## 5. What would overturn this decision

- gfC's post-disc `ae` climb ending in collapse **and** no clean recovery on restart →
  cap baseline training before `disc_start` or retune `lossconfig`.
- A discriminator variant (ADD/DINOv2 frozen-feature, notch/wavelet) that beats skipU-gfC
  on p2z *without* the BlurPool KID cost → that becomes the baseline.
- Fuse-line evidence that resize-conv's HF deficit harms real-anisotropy recovery where
  genuine HR supervision exists (vqcleanM0aSup0 line) → revisit netG per-line.

Cross-references: `doc/experiments_MS_skipE_skipU_gamma_2026-07.md` (§3 head-to-head,
§6 decoupling, §7 gamma ladder), `doc/skipU.md` (architecture), `doc/research-MS-plan.md`
(discriminator directions). Metrics pulled 2026-07-15 from the MLflow REST API
(`metrics/get-history`), experiment 12.
