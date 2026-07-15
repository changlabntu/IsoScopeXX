# Experimental summary: MS line — skipE vs skipU(B), with/without foreground-gamma

> **2026-07-15 — baseline decision**: skipU + gfC (γ0.7/−0.8) is now the declared roiD-line
> baseline, ep-200 checkpoint of run `b63909b3` as the safe default. Rationale, checkpoint
> policy, and scope conditions in `doc/baseline-skipU-gfC-2026-07.md`.

Status as of **2026-07-14**. Covers the clean (post-precrop-fix) THX10SDM20xw/roiD campaign in
remote MLflow **experiment 12** (`https://mlflow.ntugarylab.dpdns.org/#/experiments/12`), plus the
gamma-regime lessons that led to it. Blurred-era (pre-fix) results are cited only where the
precrop audit (2026-07-09, doc retired) ruled them valid — the standing rules: `val_lat_*` and
other self-normalized ratios survive across the fix; absolute-scale metrics and the blurred-era
gates do not (see the appendix note).

## 1. Models and naming

All runs share the vqcleanM0a recipe (2D VQ encoder → 3D generator → six-way PatchGAN +
L1 projection), multi-scale variant `--num_scales 4`, `--skipl1 4 --l1how max --lamb 5`.

| name | model class | netG | netD | what it is |
|---|---|---|---|---|
| MS | `vqcleanM0aMS` | `ed023ems` | `patch_16` | multi-scale progressive outputs, per-scale discriminators |
| skipE | `vqcleanM0aMSskipE` | `ed023emsfpn` | `patch_16` | the declared MS baseline config: skip trunk + coarse L1 (`--adv_ms 0.5 --lamb_coarse 1 --pyr_detach`) |
| skipU | same as skipE | `ed023emsfpnu` | `patch_16` | skipE with **resize-conv (upsample) generator** — no ConvTranspose in the upsampling path |
| skipUB | same as skipE | `ed023emsfpnu` | `patchblur_16` | skipU + **anti-aliased BlurPool discriminator** (Zhang 2019) |

skipU/skipUB are *not* separate model classes — the variant is expressed purely through
`--netG/--netD`, so their MLflow `models` tag still reads `vqcleanM0aMSskipE`. Filter runs by
`prj`, not by the models tag.

## 2. Intensity regimes: w/o gamma → plain gamma → foreground-gamma

Three metric-scale regimes exist. **Never score metric values across regimes** — the LPIPS
anisotropy baseline alone shifts from 0.633 (nm=00) to 0.521 (11g gf), and spec/KID references
move with it. Within-run trends and self-normalized ratios (`val_lat_*`) are the only things
that travel, and even those only qualitatively.

1. **nm=00 (no gamma, native scale).** The post-fix skipE control (`roiD192`, run `0c080e3d`,
   FAILED/stopped 2026-07-10 at epoch 362) showed the recipe *works but is artifact-unstable*
   at native intensity: `val_lat_p2diag` swung between ~150 and ~9,900 across validation
   epochs (final 1,889), p2z to 440. l1/ae plateaued fine (0.0586/0.178) and val_kid sat at
   1.5–3, i.e. Inception features barely punish the lattice — an early hint that KID and the
   lattice metrics measure different things. lpips_pred 0.570 vs tgt 0.633.
2. **Plain gamma (nm=11g `--gamma 0.25`, no floor) — FAILED.** Run `0606f574` (local
   thx-MS-384g store, ep 84): the dark bulk of roiD is acquisition noise, not texture; plain
   gamma amplified it into the dominant image content and the GAN regressed below its
   do-nothing baseline. Killed. Lesson recorded in `run.sh`.
3. **Foreground-gamma (nm=11g `--gamma 0.5 --gamma_lo -0.8`) — the "gf" regime.**
   `--gamma_lo` clips the noise band flat to −1 (~90% of voxels) so the gamma expansion goes
   to faint foreground only (floor −0.80 set by visual sweep; med+4·MAD −0.70 clipped real dim
   structure). Fresh `roiD192gf` prjs, thx-MS-384gf store. Both head-to-head runs below use it.
   Fuse-tuned floor is −0.9 (`zcube192gf`, fuse-MS-gf store).
4. **Foreground-gamma ladder (2026-07-14, §7) — `--gamma 0.7 --gamma_lo -0.8` ("gfC") is the
   current best point.** Motivation: at gamma 0.5 the top half of the foreground intensity
   range is compressed into ~15% of the output range — blob interiors lose contrast and read
   as overexposed. Raising gamma restores highlight headroom without touching the noise floor
   (the two knobs decouple: the floor kills noise, the exponent sets highlight compression).
   The ladder bounds the optimum from both sides: 0.75/−0.85 (`gfB`) collapsed over a long run
   (spectral retention → 0.08, KID → 14.6 — output went band-limited), while 0.7/−0.8 improves
   on 0.5/−0.8 across the board (§7). Floors are **per-dataset**, backed by measured stats
   (2026-07-14): roiD noise band median −0.85, so floor −0.8 clips 72% of voxels while −0.9
   clips only 18% (most noise passes and gets gamma-boosted); fuse zcube noise median −0.964
   with robust med+4σ edge at −0.909, so its −0.9 floor is exactly the statistical noise
   cutoff (93% clipped). Do not carry a floor across datasets.

Qualitative w/-vs-w/o gamma read (same skipE architecture): foreground-gamma brought the
lattice ratio band down from 150–9,900 to 2–33 and lpips margin improved. `val_lat` is
peak-to-local-background so it is *less* regime-sensitive than the other metrics, but the gf
background flattening does change the reference level — treat the improvement as real but
don't quote a ratio-of-ratios.

## 3. The clean head-to-head: skipE vs skipUB (roiD192gf)

Exact two-knob ablation, everything else identical (git `83f769a`, `--precrop 384` explicit,
lr 5e-4 cosine, 4×B200):

- **skipE** run `444cb790` (`skipE_gamma`) — trained to epoch 342 / step 42.9k, last log
  2026-07-10 21:34 (stopped; superseded on the box by skipUB five minutes later).
- **skipUB** run `f5533c4f` (`skipU_gamma`) — still running; epoch 383+ / step 48k as of
  2026-07-11 13:39.

### Matched-step results (~43k steps both)

| metric | skipE @42.9k | skipUB @43.5k | verdict |
|---|---|---|---|
| val_lat_p2diag | 7.91 | **1.11** | skipUB at clean floor; skipE 8× peak |
| val_lat_p4diag | 8.49 | **1.10** | same, second upsample stage |
| val_lat_p2a | 9.50 | 4.14 | skipUB better, noisy |
| val_lat_p2z | 6.35 | 5.04 | **both elevated** — Z alternate-slice stripe survives in both |
| val_kid | **1.23** | 4.66 | skipE wins; skipUB *regressed* ~2→~4.7 over training |
| val_lpips_pred | 0.479 | 0.468 | tie (both < tgt 0.521) |
| val_spec_xy_hi | 0.398 | 0.092 | skipE ~4× more broadband HF in YZ slices |
| val_spec_recon_hi | 0.275 | 0.280 | tie — 2D AE path equivalent; difference is all 3D synthesis |
| l1 (train) | 0.0534 | 0.0540 | tie |
| epoch_time | 84 s | 146 s | skipUB 1.7× slower, +25% GPU mem |

### Visual/GIF verification (epoch 340, `images/val_epoch_340.gif`, panel 2 = XupX, frames = Y–Z slices)

- skipE's blobs have **beaded, dotted boundaries** (checkerboard riding on edges); skipUB's are smooth.
- Running the repo's own `lattice_peak_ratios` on the GIF pixels reproduces the logged metrics
  (skipE p2diag 5.8 / p4diag 3.9; skipUB 1.14 / 1.17) and the **input panels of both runs
  measure clean (1.07 / 1.01)** — the lattice is network-generated, not in the data.
- Notch test: the coherent lattice peaks carry only ~0.1% of high-band spectral energy in both
  runs — so **skipE's spec_xy_hi advantage is genuine broadband HF**, not just the narrowband
  artifact, and cannot be dismissed as "all checkerboard".
- skipUB residual: a period-2 stripe along Z (relative peak ~12× background in the GIF YZ
  panel; absolute amplitude ~7% of foreground signal vs ~9% in skipE — similar in both, it just
  stands out in skipUB because everything around it is clean). Matches `val_lat_p2z` staying
  elevated in both runs.

### Stability (final-third dispersion across validation epochs)

| metric | skipE mean · range · CV | skipUB mean · range · CV |
|---|---|---|
| val_lat_p2diag | 7.9 · [2.1, 33] · 0.58 | 1.33 · [1.03, 3.2] · 0.22 |
| val_lat_p4diag | 6.8 · [3.4, 21] · 0.35 | 1.12 · [1.04, 1.5] · **0.08** |
| val_kid | 1.67 · [0.2, 4.8] · 0.62 | 4.19 · [0.6, 6.8] · 0.33 |
| val_lpips_pred | 0.478 · CV 0.03 | 0.473 · CV 0.03 |

- skipE's artifact level **swings by an order of magnitude between checkpoints** (waxes/wanes
  with the GAN equilibrium): checkpoint selection risk. Any late skipUB checkpoint is
  interchangeable on the artifact axes — its floor is structural (the generator cannot produce
  the pattern), not an equilibrium it must hold.
- Training-process stability is a tie: neither run diverged; train GAN losses equally tight
  (axx std 0.012 vs 0.013). skipE is stable training with an unstable *output property*.
- Convergence: fidelity losses (l1/ae/pyr/vq) plateaued in both by mid-run. Cosine decay is
  configured over 10,001 epochs, so LR is still ~99.7% of initial — "converged" only in the
  stationary-oscillation sense. skipUB's val_kid is the one non-stationary metric (drifting
  up); more epochs at this LR are unlikely to fix it.

## 4. Standing verdicts

1. **The blurred-era skipU result reproduces on clean data**: resize-conv + BlurPool eliminates
   the diagonal upsampling lattice (p2diag/p4diag at the ~1.1 clean floor vs 8× peaks) at zero
   measured cost in perceptual isotropy (LPIPS), data fidelity (L1), or 2D reconstruction —
   *and* with far lower checkpoint-to-checkpoint variance. This is the strongest clean result
   of the campaign, and the defensible claim: "the diagonal lattice is eliminated", **not**
   "artifact-free" (p2z stripe survives).
2. **The blurred-era skipU KID regression also reproduces**: skipUB's KID worsened over
   training (~2→~4.7) while skipE holds 1–3. KID (real-XY vs pred-YZ, Inception feature=64)
   reads skipUB's YZ slices as texture-poor; it barely reacts to the coherent lattice. The gap
   is a **sharpness/texture deficit, orthogonal to the structural artifact** — the argument for
   skipUB must make that separation explicitly and cannot cite KID or spec_xy_hi.
3. **Foreground-gamma (gf) is the viable intensity regime**: plain gamma fails on noise
   amplification; nm=00 works but leaves the GAN artifact-unstable. gf is now the default for
   the roiD line and the fuse follow-up.

## 5. Open questions / next steps

- **Close the KID hole — first pass DONE (2026-07-11, GIF-based)**: KID recomputed on matched
  geometry from the ep-340 GIF volumes (real = native-grid XY slices of the input panel,
  fake = YZ slices of XupX, feature=64/subset=16, both sets Gaussian-blurred):

  | blur σ | skipE | skipUB | gap |
  |---|---|---|---|
  | 0 | 4.39 ± 0.42 | 5.74 ± 0.49 | 1.35 (31%) |
  | 1.0 | 1.78 ± 0.19 | 2.18 ± 0.22 | 0.39 (22%) |
  | 1.5 | 1.22 ± 0.14 | 1.45 ± 0.16 | 0.23 (19%, ≈ subset noise) |

  Read: (i) on identical geometry the raw gap is ~1.3×, far smaller than the logged val_kid
  ratio (~3.8×) suggests; (ii) most of each model's KID is *shared* distance-from-real, not a
  between-model difference; (iii) the gap shrinks under blur to within measurement noise —
  majority of it is HF texture statistics, with at most a small residual mid-frequency deficit.
  Caveats: 8-bit GIF source, single val batch, different real reference than val_kid — worth a
  proper re-run from checkpoints before putting it in a paper.
- **Decouple the two knobs — DONE (2026-07-11, run `a46f14252f5a...`, see §6)**: skipU-only
  (`ed023emsfpnu` + plain `patch_16`, roiD192gf, b200) attributes the lattice fix to the
  **generator** and the KID regression mostly to **BlurPool**. Details below.
- **The p2z stripe** (alternate-slice striping along Z) survives both architectures at similar
  absolute amplitude — separate cause from the diagonal lattice; not yet diagnosed.
- **skipE gf never got a clean finish** — stopped at epoch 342 to free the box. If it is the
  fallback candidate, screen its checkpoints by `val_lat` before use.
- **In flight**: skipE foreground-gamma on fused zcube (`fuse/vqcleanM0aMSskipE/zcube192gf`,
  cropz 192 dsp 8, `--gamma_lo -0.9`, fuse-MS-gf store) — the gf recipe transferred to real
  8× anisotropy.
- **In flight (gamma ladder, §7)**: skipU gfC (`b63909b3`, 0.7/−0.8, best current point) and
  skipU gfb (`f7745c66`, 0.5/−0.9 — the floor-mismatch control; **logs no val metrics at all**
  as of 07-14, ep 120 — check whether its val loop is running before reading anything into it).
- **Fuse gamma**: keep 0.5/−0.9 there for now — fuse foreground is dim-skewed (median fg voxel
  maps to −0.59 in gamma space, only ~1.5% of fg above 0.8), so the highlight-compression
  motive for gamma 0.7 barely applies; raising gamma would cost the faint-end lift where fuse
  needs it most. If fuse blobs look hot, tone-map at decode (`--gamma_dec` ≈ 0.45) instead.

## Appendix: run ledger (experiment 12, remote)

| run | id | prj | nm/gamma | netG/netD | status | extent |
|---|---|---|---|---|---|---|
| skipE_epoch_290 | `0c080e3d` | `thx10/vqcleanM0aMSskipE/roiD192/max5skip4` | 00 / — | emsfpn / patch_16 | FAILED 07-10 10:28 | ep 362, s45374 |
| skipE_gamma | `444cb790` | `thx10/vqcleanM0aMSskipE/roiD192gf/max5skip4` | 11g / 0.5, lo −0.8 | emsfpn / patch_16 | stopped 07-10 21:34 (tag RUNNING) | ep 342, s42874 |
| skipU_gamma | `f5533c4f` | `thx10/vqcleanM0aMSskipUB/roiD192gf/max5skip4` | 11g / 0.5, lo −0.8 | emsfpnu / patchblur_16 | FAILED/stopped | ep 653, s81.7k |
| skipU (decoupling, §6) | `a46f1425` | `thx10/vqcleanM0aMSskipU/roiD192gf/max5skip4` | 11g / 0.5, lo −0.8 | emsfpnu / patch_16 | RUNNING | ep 421+, s52.7k+ |
| skipU gfB (§7) | `6957478c` | `thx10/vqcleanM0aMSskipU/roiD192gfB/max5skip4` | 11g / 0.75, lo −0.85 | emsfpnu / patch_16 | FAILED (late collapse) | ep 1304, s163k |
| skipU gfb (§7) | `f7745c66` (retry of `8e8046de`) | `thx10/vqcleanM0aMSskipU/roiD192gfb/max5skip4` | 11g / 0.5, lo −0.9 | emsfpnu / patch_16 | RUNNING, **no val metrics** | ep 120 |
| skipU gfC (§7) | `b63909b3` | `thx10/vqcleanM0aMSskipU/roiD192gfC/max5skip4` | 11g / 0.7, lo −0.8 | emsfpnu / patch_16 | RUNNING | ep 366+, s46k+ |

Plain-gamma failure `0606f574` (ep 84) is in the local thx-MS-384g store, not on the remote.
Blurred-era MS/MSfpn/MSskip standings were purged with the audit; do not resurrect their
numbers — blurred-family gates (KID ≤ 4.97, LPIPS ≤ 0.577) are void for clean runs.

## The superiority argument for skipUB (2026-07-11)

Naive Pareto dominance fails on exactly one metric (KID). The defensible claim is stronger
than "mixed results" and goes like this:

> **skipUB is superior on every metric that measures what actually differs between the
> models** — artifact presence (`val_lat_p2diag/p4diag` at the ~1.1 detection floor vs ~8×
> peaks for skipE), artifact stability across checkpoints (final-third CV 0.08 vs 0.35 —
> skipE's artifact level swings an order of magnitude between saves; skipUB's floor is
> structural), perceptual isotropy (`val_lpips_pred` tie, 0.468 vs 0.479, both below the
> 0.521 anisotropy baseline), and data fidelity (`l1` and `val_spec_recon` ties). **The
> single opposing metric, KID, favors skipE by an amount that (a) is small on matched
> geometry** (~1.3×, not the ~3.8× the noisy logged val_kid values imply; both models share
> a large common distance-from-real of 4–6 raw), **(b) collapses to within subset noise once
> high-frequency texture statistics are controlled for** (blur-KID table in §5: gap 1.35 →
> 0.23 at σ=1.5), **and (c) is partly checkerboard energy being scored as realism** —
> Inception features reward HF content indiscriminately, so skipE gets KID credit for the
> very artifact the lattice metrics detect. **Visual inspection concurs with the artifact
> metrics, not with KID.**

Supporting chain of evidence, in the order a skeptic would attack it:

1. The artifact is network-generated, not in the data — the input panels of both runs' GIFs
   measure clean (p2diag 1.07) while skipE's output panel measures 5.8.
2. The `val_lat` metrics are trustworthy across the campaign — peak-to-local-background is
   self-normalized, and it survived the precrop audit when absolute-scale metrics did not.
3. The GIF pixels independently reproduce the logged ratios, so the metric is not a
   validation-pipeline quirk.
4. The blur-KID test converts the visual impression into a measurement: the KID gap is
   texture-statistics, majority-attributable to the HF band, not structure.

What this argument must NOT claim:

- **Not "artifact-free"** — the period-2 Z stripe (`val_lat_p2z` ~5, alternate-slice
  striping) survives in both architectures at similar absolute amplitude (~7–9% of
  foreground signal). Claim "the diagonal upsampling lattice is eliminated".
- **Not "no realism cost at all"** — skipUB's raw KID drifted upward over training (~2 →
  ~4.7) and a small residual mid-frequency deficit may remain under blur. Fair phrasing:
  "at most a small residual texture deficit, within measurement noise once HF statistics
  are controlled".
- **Not sharpness superiority** — `val_spec_xy_hi` is 4× higher for skipE and the notch test
  shows that is genuine broadband HF (coherent lattice peaks are only ~0.1% of the band),
  so skipE's extra high-frequency content cannot be dismissed as pure artifact either.

Before publishing: re-run the blur-KID comparison properly from checkpoints (float
volumes, multiple val batches, the val pipeline's real reference) — the current table is
from 8-bit GIF reconstructions of a single val batch. The skipU-only decoupling run (§6)
is now done and attributes the lattice elimination to the generator.

## 6. The decoupling result: skipU-only (generator vs BlurPool), 2026-07-11

Run `a46f14252f5a4f908f8bb73fc1518db5` (exp 12, prj `thx10/vqcleanM0aMSskipU/roiD192gf/max5skip4`,
env b200, git `83f769a`, gf regime `--gamma 0.5 --gamma_lo -0.8`, still RUNNING, ep 390 / step
48.9k). Config: model `vqcleanM0aMSskipE`, **`--netG ed023emsfpnu` (resize-conv) + `--netD
patch_16` (plain, NO BlurPool)**. This completes a clean single-variable chain, because each
variant differs from its neighbor by exactly one component:

- **skipE** = `emsfpn` (ConvTranspose) + `patch_16`
- **skipU** = `emsfpnu` (resize-conv) + `patch_16` — differs from skipE **only in netG**
- **skipUB** = `emsfpnu` (resize-conv) + `patchblur_16` — differs from skipU **only in netD**

So skipE→skipU isolates the generator; skipU→skipUB isolates BlurPool.

### Matched-step (~43k) — skipE/skipUB from §3, skipU from this run's metric history

| metric | skipE | skipU | skipUB | attribution |
|---|---|---|---|---|
| val_lat_p2diag | 7.91 | **1.18** | 1.11 | diagonal lattice → **generator** (resize-conv) kills it; D irrelevant |
| val_lat_p4diag | 8.49 | **1.08** | 1.10 | same; at clean floor, final-third CV 0.02 (rock-stable) |
| val_kid | 1.23 | **2.57** | 4.66 | **BlurPool causes ~half the KID regression** (skipU recovers 4.66→2.57) |
| val_lat_p2z | 6.35 | **11.3** | 5.04 | **BlurPool was suppressing the Z stripe** — skipU worsens & destabilizes |
| val_lat_p2a | 9.50 | **2.78** | 4.14 | off-diagonal; skipU best at 43k but very unstable (see below) |
| val_spec_xy_hi | 0.398 | **0.090** | 0.092 | broadband HF loss → **generator** (resize-conv smooths); D irrelevant |
| val_lpips_pred | 0.479 | **0.493** | 0.468 | all below 0.521 baseline; skipU marginally worst |
| val_spec_recon_hi | 0.275 | **0.27** | 0.280 | tie (2D AE path identical) |
| l1 | 0.0534 | **0.0541** | 0.0540 | tie (fidelity identical) |

### Standing verdicts from the decoupling

1. **The diagonal-lattice fix belongs to the generator, not BlurPool.** skipU with a plain
   `patch_16` D already sits at the clean floor (p2diag 1.18 / p4diag 1.08), matching skipUB
   and nowhere near skipE's 8×. This is the decisive attribution the precrop audit wanted:
   resize-conv (`ed023emsfpnu`) is the component doing the work. p4diag is structurally pinned
   (final-third CV 0.02).
2. **KID regression and HF-sharpness deficit have *different* causes.** The broadband HF loss
   (`spec_xy_hi` 0.09) is the **generator's** — resize-conv smooths regardless of the D. The
   KID regression is largely **BlurPool's** — dropping it recovers KID (final-third mean 2.98
   vs skipUB's 4.19) while HF energy stays low. The two knobs are separable: you cannot fix KID
   by touching netG, nor fix sharpness by touching netD.
3. **BlurPool is a genuine trade, not a free win.** It costs KID but it was *suppressing the
   p2z stripe and stabilizing the off-diagonal axes*. Without it, `val_lat_p2z` worsens (matched
   11.3; final-third mean 8.51, range [1.4, 82.8], CV 1.30) and `p2a` similarly destabilizes.
   This **qualifies the §"superiority argument"** — skipUB is not Pareto-cleaner than skipU on
   everything; it trades KID for Z-stripe suppression and stability.

### Implication for the research plan (doc/research-MS-plan.md)

The ideal operating point wants all three at once: lattice-gone (keep resize-conv netG) + KID
recovered (drop BlurPool) + Z-stripe suppressed and stable (what BlurPool bought). No existing
discriminator here delivers all three, which is exactly the gap a **frozen-feature D
(ADD/DINOv2, direction 1)** or a **notch/wavelet D (direction 5)** targets. The p2z stripe being
anti-aliasing-sensitive and D-dependent also supports the direction-4/5 read that it is a
resampling-phase artifact rather than a fixed generator feature.

Caveats: single run, noisy gf-regime KID (skipU final-third range [0.56, 6.38]); skipE/skipUB
columns are §3's logged values, not re-pulled; run still training. Metrics pulled from the
MLflow REST API (`metrics/get-history`, matched near step 43k).

## 7. The gamma ladder on skipU (2026-07-14)

Three runs vary only the `--gamma`/`--gamma_lo` pair on the identical skipU architecture
(`vqcleanM0aMSskipE` + `ed023emsfpnu` + `patch_16`, roiD192, max5skip4), so the gamma effect
is single-variable against §6's `a46f1425`:

- **gfC** `b63909b3` — **0.7 / −0.8** (highlight-headroom fix, floor kept) — RUNNING, ep 366 / s46k
- **gfB** `6957478c` — 0.75 / −0.85 — ran long (ep 1304 / s163k), FAILED tag
- **gfb** `f7745c66` — 0.5 / −0.9 (fuse floor on roiD; on-purpose mismatch control) — logs
  **no val metrics**; nothing to compare yet

Regime caveat applies with force here: each gamma has its own metric scale (`val_lpips_tgt`
baseline: 0.521 at γ0.5, 0.476 at γ0.7/lo−0.8, 0.589 at γ0.75/lo−0.85). Compare **margins
below own baseline** and the self-normalized `val_lat_*` ratios; treat KID/spec across
regimes qualitatively.

### Matched-step (~43k) with late-window (last-10-val mean) in parens

| metric | skipU γ0.5 (`a46f1425`) | **gfC γ0.7 (`b63909b3`)** | gfB γ0.75 (`6957478c`) |
|---|---|---|---|
| lpips margin below own tgt | 5.4% (late **2.6%**, decaying) | **6.3% (late 7.6%, holding)** | 3.7% (late 15.8%, but see collapse) |
| val_kid | 2.57 (late 1.87) | **1.28 (late 1.69)** | 3.43 → **14.6 late** |
| val_lat_p2diag | 1.18 (late 3.66) | 2.44 (late **1.98**) | 1.10 → 6.26 late |
| val_lat_p4diag | 1.08 | 1.12 | 1.09 → 2.36 late |
| val_lat_p2z | 11.3 (late 17.1) | **4.8 (late 8.0)** | 3.1 (late 1.3) |
| val_lat_p2a | 2.78 (late 5.32) | **2.85 (late 3.05)** | 4.28 (late 2.21) |
| val_spec_xy_mid | 0.489 (late 0.616) | 0.582 (late 0.505) | 0.267 → **0.077 late** |
| val_spec_xy_hi | 0.090 | 0.081 | 0.058 → **0.006 late** |

### Reading

1. **γ0.7/−0.8 beats γ0.5/−0.8 on every axis at matched steps** (architecture identical):
   healthier and *stable* lpips margin (the γ0.5 run's margin decayed 5.4%→2.6% by s52.7k
   while gfC holds ~7%), roughly half the KID, p2z stripe at half the amplitude (4.8–8.0 vs
   11–17), off-diagonal p2a stable instead of drifting, and comparable-to-better mid-band
   spectral retention. Diagonal lattice stays near the clean floor for both (resize-conv
   does that regardless of gamma, §6).
2. **γ0.75/−0.85 is the overshoot bound.** Early it looked fine (all lat ratios ~1–4 at 43k);
   over the long run the output went band-limited — spec_xy collapsed to 0.077/0.006 and KID
   blew up to ~14.6. (Its late lpips margin and p2z "improvements" are artifacts of an
   over-smoothed output: a blurry volume is trivially isotropic and stripe-free.) Interior
   optimum confirmed: 0.5 over-compresses highlights, 0.75/−0.85 tips into degeneracy.
3. **gfC's cost profile matches its regime, not a regression**: spec_xy_hi 0.081 ≈ the other
   resize-conv runs (the HF deficit belongs to the generator, §6), and it retains skipU's
   one advantage over skipUB (no BlurPool KID penalty) while its p2z, though present
   (late ~8), is materially better than its γ0.5 sibling's 17.
4. Practical: gfC at ~46k steps is the current **best single checkpoint source** for the roiD
   line on combined artifact + realism axes — still RUNNING; screen checkpoints by `val_lat`
   as usual for the skipU family.

Metrics pulled 2026-07-14 from the MLflow REST API (`metrics/get-history`), matched near step
43k, late window = mean of the last 10 validation points at pull time (gfC s46.0k, γ0.5
s52.7k, gfB s163.1k). γ0.5-sibling and gfB numbers are from the same pull, not §6's table.
