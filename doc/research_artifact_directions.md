# Research: future directions against generative artifacts (2026-07-07)

Deep-research report on reducing artifacts in the MS models (synthesized volumes show
artifacts vs reference). 21 primary sources fetched, 100 claims extracted, top 25
adversarially verified (23 confirmed 3-0, 2 refuted). Companion to
`doc/experiments_MS.md` — directions here map onto its failure modes:
(a) ConvTranspose checkerboard aliased away by the stride-2 patch D,
(b) YZ smoother than XY (deterministic per-slice latent has no Z texture),
(c) EMA muting high-frequency texture, (d) projection-loss null space filling
with checkerboard.

## Headline

Checkerboard is a **signal-processing defect of the pipeline, not of the GAN
objective** — and it lives on BOTH sides:

- **Generator side:** ConvTranspose produces checkerboard even at random init and
  even in the "even-overlap" k4s2 case we use — the architecture biases the spectrum
  before training starts (Distill 2016; Aitken 2017; StyleGAN3).
- **Discriminator side:** stride-2 convs in the D (i) inject checkerboard-period
  gradients back into G and (ii) violate Nyquist, aliasing low-magnitude high
  frequencies into a phase-dependent, non-shift-invariant response — the D literally
  cannot reliably see the artifact it should punish (Distill; Zhang ICML 2019;
  Schwarz NeurIPS 2021). This is independent confirmation of the in-house band-probe
  diagnosis.
- **Training-signal side:** a trained G *can* compensate for upsampling checkerboard
  when the loss penalizes it (Schwarz) — so persistent checkerboard means the loss/D
  doesn't see it. Fixing G's upsampler alone treats the symptom; the null-space
  failure mode (d) is only closed by a loss that covers the frequencies.

## Ranked directions

### 1. Resize-conv (or ICNR sub-pixel) generator upsampling — confirms Z3

Replace ConvTranspose3d k4s2 with nearest/trilinear upsample + stride-1 Conv3d
(`use_upsample` already exists in the generator ctor), or sub-pixel conv with ICNR
init. Resize-conv experimentally eliminates checkerboard; ICNR is artifact-free at
init only (not guaranteed after long adversarial training). **Attacks (a), (d).**
Refuted claim to ignore: ICNR is NOT a free lunch matching resize-conv cost with
more power — they are trade-offs (resize-conv costs more memory than sub-pixel).
[distill.pub/2016/deconv-checkerboard, arXiv:1707.02937, StyleGAN3]

### 2. BlurPool the PatchGAN discriminator — new, cheap, untried

Insert a fixed low-pass blur before every stride-2 downsample in `patch_16`
(Zhang ICML 2019 anti-aliased CNNs). Zero learned parameters, drop-in, standard in
StyleGAN2/3 and VideoGigaGAN. Makes the D shift-invariant so faint checkerboard gets
a stable response instead of aliasing into invisibility. **Attacks (a) from the D
side — the half of the diagnosis Z3 doesn't touch.** Open question flagged by the
verifiers: blurring also attenuates the high frequencies the D must perceive
(Schwarz lists BlurPool among downsamplers that can impair the training signal), so
the net effect on checkerboard detection specifically needs an empirical check here.
[arXiv:1904.11486, arXiv:2111.02447]

### 3. Spectral (Fourier) discriminator ALONGSIDE the spatial D + focal frequency loss

Spectral Ds beat spatial Ds at detecting high-frequency differences; spatial wins at
low frequency — use both (Luo ICCV 2023, DualFormer; corroborated by SSD-GAN and
Fourier-space-losses ICCV 2021). Separately, D2R (MICCAI 2025) weights Focal
Frequency Loss at λ=100 vs SSIM λ=1 in a microscopy volumetric-SR decoder,
explicitly to recover high-frequency detail. FFL is loss-only, single-pass,
self-supervision-compatible. **Attacks (a), (b), (d): puts explicit training signal
on exactly the frequency bands the stride-2 spatial D misses — plausibly the
prerequisite for ever retesting the band projection.** Caveats: spectral D is an
MLP with resolution-scaling limits; FFL evidence is design-intent, not an isolating
ablation. [arXiv:2307.12027, MICCAI 2025 paper 2453]

### 4. Alias-free nonlinearity handling (StyleGAN3-style)

Pointwise ReLU itself creates non-bandlimited signals that alias on sampling; the
StyleGAN3 recipe is upsample → nonlinearity → filtered downsample around each
nonlinearity. **Attacks (a) at its deepest root.** Refuted overreach: this is a
strong *mitigation*, not a provable zero-leakage guarantee. Heaviest engineering of
the architectural options; do after 1–3. [nvlabs.github.io/stylegan3]

### 5. Diffusion-distilled Z-texture (D2R pattern) or stripe-completion reformulation

The microscopy field's answer to failure mode (b) — where our deterministic
per-slice latent supplies no Z-frequency texture:

- **D2R (MICCAI 2025):** train a 2D diffusion model on real XY slices *offline*,
  generate pseudo-HR XZ/YZ volumes, distill into a feed-forward 3D net — no
  diffusion at inference (sliding-window, constant cost). Averaging many noisy
  diffusion samples during distillation cancels per-sample hallucinations —
  a concrete anti-hallucination mechanism with no HR ground truth. **Attacks (b),
  (d)**; interaction with EMA muting (c) is an open question.
- **DiffuseIR (MICCAI 2023):** diffusion prior trained only on XY slices,
  conditioned on the low-Z input at sampling; generalizes across anisotropy factors
  without retraining. Iterative sampling at inference — violates our one-pass budget
  unless distilled.
- **CRIS (2026 preprint, unreplicated):** no GAN, no diffusion — insert blank slices
  at unobserved Z locations and train 2D conditional *stripe completion* under a
  known sampling mask. Reconstruction-under-known-sampling instead of
  deblurring-an-interpolation; conceptually adjacent to our projection
  self-consistency but with no free null space. **Reframing candidate for (d).**

[MICCAI 2025 paper 2453, arXiv:2306.12109, arXiv:2606.15967]

## Mapping to the MS queue (doc/experiments_MS.md)

- **Z3 is confirmed as the right next architectural iteration** — but the research
  says do it as a *pair*: G-side resize-conv (Z3) + D-side BlurPool (direction 2),
  since a trained G compensates only when the D can see the artifact. Per the
  one-change-per-iteration discipline: Z3 first (fresh training anyway — weight
  shapes change), BlurPool as the following iteration.
- **Direction 3 (spectral D + FFL) is the new loss-side lever** the in-house catalog
  didn't have; it is also what would make retesting `--l1how band` viable.
- **Z2 (noise injection) got NO verified support or refutation** — the volumetric
  noise-injection search angle produced no claims surviving verification. Keep Z2 as
  an in-house hypothesis; texton-broadcasting + noise (arXiv:2203.04221) is the
  closest unverified lead.
- **VQ/FSQ/VAR angle also unverified** (budget-dropped before verification): FSQ
  (arXiv:2309.17269) simplifies away codebook-collapse machinery but reportedly
  degrades in *residual* multi-stage stacks (residual magnitudes decay
  exponentially — arXiv:2508.15860), which cautions against naive FSQ in our
  VAR-style residual quantizer. VARSR (arXiv:2501.18993) extends next-scale
  prediction to SR. Treat all as leads, not evidence.

## Update (2026-07-07): symptom refined — diagonal structure growth, not overlay checkerboard

Observed artifacts are structures that FOLLOW the checkerboard direction and grow in
diagonal patterns, rather than visible checkerboard texture. Reinterpretation:

- This is **aliasing-as-positional-encoding / texture sticking** (StyleGAN3's core
  phenomenon): the residual lattice harmonics of stride-2 ConvTranspose act as a
  hidden coordinate system; the generator nucleates and orients structure on it.
- **Diagonal specifically** because the k4 kernel's low-pass is weakest at the
  spectral corners (±π,±π) — diagonal harmonics dominate the residual lattice,
  composed over three upsampling stages.
- Consistent with Schwarz et al.: the generator compensates the *visible*
  checkerboard (losses punish it) while still *using* the lattice (nothing punishes
  that). All three supervision signals are blind to orientation statistics:
  stride-2 D shift-variance is worst at near-Nyquist diagonals; patch_16's receptive
  field can't see extended-structure orientation; max projection is nearly invariant
  to in-Z orientation. Failure mode (d) expressed in content, not texture.
- The vendored taming 2D decoder already uses resize-conv, so the ConvTranspose3d
  trunk is the prime suspect (consistent with the band-probe finding).

Re-ranking under this symptom:

1. **Diagnostic first (no training):** angular FFT power spectra of generated vs
   real slices per plane orientation; check the diagonal excess sits at lattice
   harmonics (period 2/4/8 voxels); XY-only vs Z-containing planes localizes 2D
   decoder vs 3D trunk; random-init generator FFT should already show the lattice.
2. Z3 resize-conv trunk (removes the lattice source).
3. **Z2 noise injection promoted:** the deterministic latent gives no other
   high-frequency seed, so the grid is the only texture scaffold — noise removes the
   *incentive*, Z3 removes the *grid*. (Mechanism-argued; no verified literature.)
4. **Spectral D promoted:** FFT-magnitude discriminator sees global
   orientation-energy anisotropy (excess diagonal power) directly — the loss that
   punishes grid-following itself.
5. StyleGAN3 filtered nonlinearities — the deep fix, after 1–4.

## Diagnostic result (2026-07-07): mechanism CONFIRMED on val_epoch_110 GIF

Mean windowed FFT power spectrum over all 128 lateral frames (X×Z planes; frames step through Y — note base.py's local names swap X/Y vs the verified (B,C,Y,X,Z) layout) of the
enhanced-output panel (XupX), peak-to-local-background ratios:

| lattice harmonic | (fY, fZ)/Nyquist | XupX panel | out128/out64 panels |
|---|---|---|---|
| period-2 **diagonal** (X–Z) | (1, 1) | **94×** | 1.1× |
| period-2 along X (in-plane) | (1, 0) | **56×** | 1.0× |
| period-2 along Z | (0, 1) | 9.7× | — |
| period-4 **diagonal** (X–Z) | (0.5, 0.5) | **21×** | 1.2× |
| period-4 axis-aligned | (0.5, 0) / (0, 0.5) | ~2× (noise) | — |
| period-8 (any) | (0.25, ·) | ~1.5× (none) | — |

Reading:
- The diagonal corner dominates, exactly as the k4s2 spectral analysis predicts
  (kernel attenuation weakest at (±π,±π)). Period-4 exists ONLY diagonally →
  at least the last TWO ConvTranspose3d stages imprint, and their composition is
  diagonal-dominant. Fixing only the final stage is insufficient; Z3 must replace
  all trunk upsampling stages.
- Visually the artifact is not overlay checkerboard but **beaded chains of dots
  whose spacing and direction match the period-2/4 diagonal lattice** — content
  nucleated on lattice sites (texture sticking), matching the "structures follow
  the checkerboard direction" symptom.
- Period-2 along in-plane X (56×) survives even though the XY-slice discriminators see that
  axis — period-2 IS the Nyquist frequency that a stride-2 first conv aliases into
  a phase-dependent response. Direct corroboration of the D-side blindness
  mechanism (BlurPool direction).
- The coarse-head panels are clean only because the GIF trilinear-upsamples them
  (low-passes period-2); not evidence about the trunk.

Diagnostic recipe (rerun anytime): per-frame Hann-windowed |FFT|², averaged over
frames, peak vs 21×21 median background at the lattice frequencies above.

**Cross-plane analysis** (volume reconstructed from the 128 GIF frames → (Y=128,
X=256, Z=256); same probe on all three orientations; peak/bg):

| harmonic | X–Z planes | Y–Z planes | X–Y planes |
|---|---|---|---|
| period-2 in-plane diagonal | 17.7× | **35.7×** | 5.1× |
| period-2 along X | 15.4× | — | **17.0×** |
| period-2 along Y | — | 2.0× | 3.5× |
| period-2 along Z | 9.9× | 9.7× | — |
| period-4 in-plane diagonal | **21.1×** | 9.8× | **16.5×** |

Reading:
- The diagonal beading is in BOTH lateral orientations (X–Z and Y–Z), so the 3D
  lattice couples Z with each in-plane axis — every trunk upsampling stage/axis
  imprints; the fix must cover all of them (as concluded above).
- X–Y planes — the orientation the XY discriminators directly view — still carry
  17× period-2-along-X and 16.5× period-4-diagonal. Period-2 aliases at the D's
  first stride-2 layer, period-4 at its second: each patch-D depth is blind to the
  harmonic at its own Nyquist. Strengthens the BlurPool-all-layers case.
- Unexplained X↔Y asymmetry: pure-X period-2 is strong (15–17×) while pure-Y is
  near-clean (2–3.5×), despite an architecturally symmetric trunk. Worth checking
  for an axis-asymmetric op (latent reshaping/downbranch, crop sizes, or GIF-side
  quantization) before reading too much into it.
- Caveat: 8-bit GIF quantization; ratios approximate (direct-frame X–Z analysis
  gave larger ratios, 94×/56×, with a different background estimate — relative
  pattern is what matters).

## Refuted claims (do not rely on)

1. "ICNR sub-pixel conv matches resize-conv cost with strictly more power" — 0-3.
2. "StyleGAN3's changes guarantee zero alias leakage / full sub-pixel equivariance" — 0-3.

## Standing caveats

All 2D-GAN results (StyleGAN3, BlurPool, spectral-D, frequency-bias) transfer to our
ConvTranspose3d volumetric setting *by mechanism, not by 3D benchmark*. Costs under
the 3D memory budget (BlurPool activation memory, resize-conv vs sub-pixel memory,
spectral-D MLP scaling) were not quantified for this pipeline. CRIS is a single
2026 preprint.

## Key sources

- Odena, Dumoulin, Olah — Deconvolution and Checkerboard Artifacts (Distill 2016)
- Aitken et al. — Checkerboard artifact free sub-pixel convolution / ICNR (arXiv:1707.02937)
- Zhang — Making Convolutional Networks Shift-Invariant Again / BlurPool (ICML 2019, arXiv:1904.11486)
- Schwarz et al. — On the Frequency Bias of Generative Models (NeurIPS 2021, arXiv:2111.02447)
- Karras et al. — Alias-Free GANs / StyleGAN3 (NeurIPS 2021)
- Luo et al. — Effectiveness of Spectral Discriminators for SR (ICCV 2023, arXiv:2307.12027)
- D2R — Diffusion-guided structure distillation for axial SR (MICCAI 2025, paper 2453)
- DiffuseIR — unsupervised isotropic reconstruction via diffusion (MICCAI 2023, arXiv:2306.12109)
- CRIS — conditional stripe-completion isotropic restoration (arXiv:2606.15967, preprint)
- Wavelet-domain GAN losses for artifact control (CVPR 2024, arXiv:2402.19215) [unverified lead]
- Focal Frequency Loss (ICCV 2021, mmlab-ntu.com/project/ffl) [unverified lead]
- FSQ (ICLR 2024, arXiv:2309.17269); residual-FSQ decay (arXiv:2508.15860); VARSR (arXiv:2501.18993) [unverified leads]
