# MS line — research plan: directions from recent CVPR-adjacent work

Drafted **2026-07-12**. Candidate innovation directions for the next phase of the MS campaign,
ranked by how directly they attack the documented open problems
(`doc/experiments_MS_skipE_skipU_gamma_2026-07.md`). Literature added 2026-07-12 from a
3-track deep-research fan-out (108 agents, 26 primary sources fetched, 124 claims extracted).

**Confidence tags used below:**
- **[V]** = 3-vote adversarially verified (2/3-refute kill threshold), quotes confirmed against
  the primary source. Only Track A directions (1, 2) got into the verify budget.
- **[X]** = extracted from the primary source with a supporting quote by a single fetch agent,
  but NOT adversarially verified (fell outside the 25-claim verify cap). Directions 3–6 rest on
  [X] evidence — treat the citations as real and the numbers as author-reported-but-unaudited;
  spot-check before quoting in a paper.

## Open problems these directions target

- **P1 — KID/HF-texture deficit**: skipUB eliminates the diagonal lattice but its val_kid
  drifts upward (~2→~4.7); BlurPool bought lattice-freedom by desensitizing the D to HF.
- **P2 — p2z stripe**: period-2 alternate-slice striping along Z survives *both* architectures
  (~7–9% of foreground amplitude); cause undiagnosed, likely in the Z-resampling path.
- **P3 — artifact-equilibrium instability**: skipE's artifact level swings an order of
  magnitude between checkpoints (non-convergent GAN oscillation).
- **P4 — under-claimed compression**: the VQ latent is "compact" but there is no entropy model,
  no bits-per-voxel, no rate–distortion curve.
- **P5 — no trust/uncertainty story**: no measure of which synthesized Z structure is reliable.

---

## 1. Modernized GAN objective + feature-space discriminator — targets P1, P3

**R3GAN [V]** (Huang, Gokaslan, Kuleshov, Tompkin — *The GAN is dead; long live the GAN!*,
NeurIPS 2024; arXiv 2501.05441; code github.com/brownvc/r3gan). A regularized relativistic loss
(RpGAN + zero-centered R1-on-real / R2-on-fake) with a **proven local-convergence guarantee**
(Props I/II), which the authors use to discard all StyleGAN2-era stabilization tricks and
modernize the backbone. Surpasses StyleGAN2 on FFHQ/ImageNet/CIFAR/Stacked MNIST. The objective
is separable from the backbone — portable as a loss-level swap in `BaseModel.add_loss_adv`
(a few lines of autograd; R1/R2 adds a double-backward cost).

**Frozen-feature / projected discriminators [V]**:
- **ADD / SDXL-Turbo** (Sauer et al., ECCV 2024; arXiv 2311.17042): D = a **frozen DINOv2
  ViT-S** + lightweight trainable heads at multiple layers, hinge loss, **feature-space R1**
  (beneficial above 128²). Ablation Table 1a: DINOv2 ViT-S (FID 20.6) beats ViT-L (24.0). The
  paper's own explanation for why 4-step ADD beats its 50-step teacher is that the adversarial
  component **specifically enhances HF texture and reduces oversmoothing** — directly on P1.
- **Projected GAN** (Sauer et al., NeurIPS 2021; arXiv 2111.01007): the lineage foundation;
  naive projection underperforms — needs CCM/CSM random-projection feature-mixing.
- **Caveat [V]**: Kynkäänniemi et al. (ICLR 2023; arXiv 2203.06026) show an ImageNet-pretrained
  projected D can *game* Inception-FID. Validate with DINO-based KID or human eval, not FID.
  (The "40× faster / 5d→3h" Projected-GAN stability framing was **refuted 1-2** — do not cite.)

**Prior-art overlap**: all validated on 2D natural images; none on volumetric/microscopy SR.
Applying either to a six-way PatchGAN across three orthogonal axes is genuine headroom.

**Integration**: R3GAN = loss swap in `models/base.py` (lowest risk, py38-safe — only the
penalty autograd ports). Feature-D = new `--netD` in `networks/registry.py` wrapping a frozen
DINOv2 (adds a torch-hub/timm weight dependency, but inference-only, no custom CUDA).

**Claimable**: "a convergence-guaranteed relativistic objective removes skipE's
checkpoint-to-checkpoint artifact instability (P3)"; "a frozen-feature discriminator restores
the HF texture that BlurPool suppressed, closing the skipUB KID deficit (P1) without
re-admitting the alias lattice." Both are directly measurable on the existing val_lat / val_kid
/ blur-KID / val_spec framework.

## 2. Orthogonal 2D diffusion priors for 3D isotropy — targets P1 (big novelty swing)

A mature lineage exists; the self-supervised, no-HR-GT, materials-modality corner is the gap.

- **TPDM [V]** (Lee et al., ICCV 2023; arXiv 2303.08440): models 3D as a product of two
  perpendicular 2D diffusion priors, alternating denoising between planes. MRI Z-SR 5mm→1mm:
  PSNR 35.97 / SSIM ~0.964–0.970, beating DPS (34.77) and MCG (32.72). **But** it trains its 2D
  priors on slices from *already-isotropic* volumes (presumes HR perpendicular slices — unlike
  us), and inference is **2000 PC steps ≈ 24–36 h/volume** — the motivation for few-step
  distilled refiners.
- **DiffuseIR [V]** (MICCAI 2023; arXiv 2306.12109) and **Reference-Free EM** (Lee & Jeong,
  MICCAI 2024; arXiv 2308.01594): fully unsupervised isotropic reconstruction of 3D microscopy
  from **lateral slices alone**, no HR-GT, no known degradation — the closest prior art to our
  premise. Biological EM, not materials.
- **D2R [V]** (arXiv 2411.16792; MICCAI 2025): three-stage self-supervised VSR — 2D diffusion
  prior on XY (lightweight IRSDE) → cross-plane pseudo-HR → distill into a 3D VSR net. On
  FIB-SEM at **8× axial** (matches our fuse dsp 8), D2R-AENet 27.64/27.83/27.83 vs
  self-sup baseline 23.05/22.69/22.69, approaching supervised. (One fine-grained pipeline
  sub-claim was **refuted 1-2** — cite the framework and headline numbers, not the exact
  averaging mechanics.)
- **CRIS [X]** (arXiv 2606.15967, 2026-06): self-supervised isotropic restoration as 2D stripe
  completion on orthogonal reformats + multi-view fusion — *not* diffusion/GAN. MRI 32.9 dB /
  0.963, vEM 29.1 dB / 0.830 at 4× anisotropy. **Direct prior art for our orthogonal-slice +
  multi-view-fuse regime** — cite as related work, and a competitive-differentiation risk.

**Integration**: heaviest — either a new model class (score-based refinement of `XupX`) or a
distilled few-step refiner as a post-hoc stage. Diffusion tooling is a real dependency add on
py38pl16; a distilled 1–4-step refiner keeps inference tractable vs TPDM's 24–36 h.

**Claimable**: "diffusion-prior isotropy transferred to the degradation-free, no-HR-GT,
**materials-science** regime" (modality gap is open); "a few-step distilled refiner recovers HF
texture (P1) at practical inference cost." The fuse dataset (real 8× anisotropy, 3 registered
views) is a testbed the medical/EM versions lack.

## 3. Neural compression: FSQ/BSQ + next-scale entropy model — targets P4

The strongest *newly surfaced* direction — a recent ICLR paper does almost exactly the
next-scale-entropy-model idea, in 2D.

- **FSQ [X]** (Mentzer et al., ICLR 2024; arXiv 2309.15505): scalar-quantize a <10-dim
  projection; implicit product codebook, **no codebook collapse, no commitment loss / reseeding
  / entropy penalty**. Parity with VQ in MaskGIT/UViM. Would let us drop the VQ codebook-loss
  term in `backward_g`. 2D-only — no 3D/volumetric validation.
- **RFSQ [X]** (arXiv 2508.15860, 2025-08): **directly warns** that naive residual FSQ fails
  (residual magnitudes decay across stages so later stages get un-quantizable signals) — the
  exact failure mode a drop-in FSQ swap into our VAR-style residual stack would hit. Fix:
  per-stage learnable scaling or pre-stage invertible LayerNorm; +45% perceptual / −28.7% L1.
  **Read this before touching `encode()`.**
- **VAR [X]** (Tian et al., NeurIPS 2024; arXiv 2404.02905): the next-scale-prediction paradigm
  our multi-scale residual VQ is derived from; FID 18.65→1.73 on ImageNet, ~20× faster than
  raster AR. The natural entropy-model prior over our scale-0→3 codes.
- **ARPC [X]** (ICLR 2026; tianweiz07.github.io/Papers/26-iclr-3.pdf): **first to use VAR
  next-scale prediction for image compression** — transmit first k of K scale bitstreams, AR-
  generate the rest at the decoder; **using the VAR transformer as the arithmetic-coding
  probability model cuts bitrate ~30% at no quality cost.** Replaces VQ with a BSQ-style bitwise
  residual quantizer. 2D natural images only — **no 3D/scientific/microscopy application**, so
  the volumetric-bits-per-voxel story is open headroom.
- **Direct competitor [X]**: a VQ-VAE isotropy+compression method (arXiv/bioRxiv 704755 +
  J. Imaging Inform. Med. 2026; Springer s10278-026-01911-5) reports **128× slice compression +
  8× axial SR, ~1000× storage reduction, 2D-encoder/3D-decoder** — the same decomposition as
  ours — **but reports compression only as a storage ratio, no entropy-coded bits-per-voxel.**
  This both validates the framing and defines the gap we should fill (an actual RD curve).

**Integration**: FSQ/BSQ = contained swap of `VectorQuantizer2` in `encode()` (heed RFSQ's
conditioning fix). Entropy model = a new small AR head over the existing scale token grids;
larger but self-contained, no exotic deps.

**Claimable**: "first **entropy-coded bits-per-voxel rate–distortion** for self-supervised
isotropic 3D volume compression" (the storage-ratio competitor leaves this open); "FSQ removes
VQ codebook-loss tuning at parity." Turns P4 from a tagline into a measured second contribution.

## 4. Uncertainty + uncertainty-gated self-labeled segmentation — targets P5 (and P2!)

- **Self-supervised uncertainty SR [X]** (arXiv 2603.14074, 2026-03): a **Gaussian-NLL loss
  that jointly estimates per-voxel mean + variance with no HR-GT**, provably equivalent to
  supervised NLL for random-shift subsampling degradation — *exactly our dsp Z-subsample
  setting*. Self-sup 54.05 dB vs supervised 54.24 dB (0.2 dB gap), calibrated coverage.
  **Critically for P2**: the paper reports its uncertainty maps show a **period-two
  checkerboard from the 2× subsampling** (the four sub-grids carry different error) — a
  published mechanistic analog of our period-2 Z stripe. This is the single most useful hit for
  diagnosing P2: the stripe may be a subsampling-phase artifact, not a generator alias.
- **REHRSeg [X]** (arXiv 2410.10097, 2024-10): self-SR pseudo-supervision + an
  **uncertainty-aware SR head** for HR 3D MRI segmentation from LR input, uncertainty focused
  at ROI boundaries; SR↔seg feature alignment via structural KD. The template for the
  uncertainty-gated self-labeled segmentation route.

**Integration**: 4a (aleatoric head) = one extra decoder output channel + swap L1 projection
for Gaussian-NLL projection — small, py38-safe. Epistemic via existing EMA/checkpoint ensemble
(we already observe checkpoint disagreement — that *is* the signal). 4b (seg head) = larger add.

**Claimable**: "per-voxel trust maps for hallucinated Z structure (P5), self-supervised";
"inter-checkpoint variance quantifies P3"; and a **novel diagnostic** — attribute the p2z stripe
(P2) to subsampling-phase uncertainty via the 2603.14074 mechanism.

## 5. Alias-free generator design — targets P2 (and P1)

- **StyleGAN3 [X]** (Karras et al., NeurIPS 2021; arXiv 2106.12423): pins coordinate-dependent
  generator artifacts on **aliasing from careless resampling/pointwise ops inside the
  generator** and fixes them with low-pass-filtered nonlinearities/resampling — a **netG-level**
  change that matches StyleGAN2 FID (i.e., anti-aliasing need not cost fidelity — bears on the
  skipUB HF deficit). Foundational prior art: our novelty must be the 3D/self-supervised
  transfer. Directly supports diagnosing P2 as a generator resampling alias.
- **DeStripe [X]** (Liu et al., MICCAI 2022): **self-supervised (Self2Self) Fourier-domain
  stripe removal, no stripe-free GT** — exploits that unidirectional stripes concentrate in
  specific Fourier coefficients while foreground is isotropic; a GNN repairs only those, with
  unfolded-Hessian regularization. **The single most on-point method for P2** (periodic
  directional stripes in anisotropic 3D microscopy), and it needs no ground truth.
- **Wavelet-domain SR losses [X]** (CVPR 2024; arXiv 2402.19215): train the D **only on HF
  wavelet sub-bands** so it separates genuine HF detail from artifacts better than RGB/Fourier
  losses. **SWAGAN [X]** (Gal et al., SIGGRAPH 2021): wavelet transforms throughout G and D;
  attributes HF loss to spectral bias — the same failure as P1 — and improves HF realism.

**Integration**: notch loss on the known p2z frequency or a DeStripe-style Fourier repair =
loss/module add (contained). A full StyleGAN3-style alias-free trunk or wavelet D = new
`--netG`/`--netD` (larger, but the `ems→emsfpn→emsfpnu` progression is the established path).

**Claimable**: "the p2z stripe eliminated by a self-supervised spectral repair / notch, closing
the last standing artifact skipUB left"; potentially "wavelet-band D recovers HF (P1) without
the alias lattice" — a P1/P2 twofer feeding direction 1's ablation table.

## 6. Lightweight / factorized 3D decoder — motivated by skipUB cost (1.7× time, +25% mem)

The Mamba-in-GAN question is answered: it exists, but only in 2D so far.

- **SegMamba [X]** (Xing et al., MICCAI 2024; papers.miccai.org/.../Paper0663): Mamba backbone
  with a **Tri-Orientated Mamba (ToM)** module — Mamba over three orthogonal sequences —
  whole-volume long-range context, efficient at 64³ (~260k seq len). The ToM tri-scan matches
  our three-axis geometry (Z-scan = long-range structure to invent; X/Y-scan = observed
  texture). Segmentation, not SR/GAN.
- **I2I-Mamba [X]** (arXiv 2405.14022): **working Mamba-in-a-GAN-generator** (patch-D, dual-
  domain SSM in a CNN bottleneck) for medical synthesis — but **2D**. Its spiral-scan SSM gives
  higher **angular isotropy** in the receptive field than raster scans (relevant to an
  isotropy-oriented generator); 11 ms / 2.5 GB / 105 M params.
- **AiM [X]** (arXiv 2408.12245): Mamba as a plain next-token AR image generator, FID 2.21,
  2–10× faster than AR baselines — supports a Mamba **entropy model** for direction 3.
- **UD-Mamba [X]** (arXiv 2502.02024): folds uncertainty into the scan order (ties to
  direction 4) — but 2D segmentation only.

**Integration / dependency risk — the real blocker**: `mamba-ssm`'s selective-scan CUDA kernels
want recent PyTorch/CUDA/Triton and would very likely **force an environment fork** from the
legacy py38pl16 box — a cost none of directions 1, 3, 4, 5 carry. Also: **no evidence of Mamba
in a 3D GAN generator** (I2I-Mamba is 2D; SegMamba is 3D but not a GAN) — highest research risk.

**Claimable**: "linear-complexity global 3D Z-context recovers long-range structure a local
U-Net can't, at lower cost than skipUB" — but this is the most speculative claim and the only
direction adding a genuinely new *capability* rather than repairing/measuring what exists.

---

## Revised ranking (novelty-vs-prior-art headroom × integration cost)

| # | Direction | Headroom | Integ. cost | Dep. risk | Confidence | Verdict |
|---|---|---|---|---|---|---|
| 1 | R3GAN loss + feature-D | Med (2D-only prior art) | **Low** (loss swap) / Med (netD) | Low | **[V]** | **Do first** — cheapest, verified, hits P1+P3 |
| 3 | FSQ/BSQ + entropy model | **High** (no 3D bits-per-voxel exists) | Med | Low | [X] | **Best 2nd contribution** — makes P4 real |
| 4 | Uncertainty (+seg) | High (self-sup NLL new to our regime) | Low (4a) / High (4b) | Low | [X] | 4a is a cheap, high-value add; also diagnoses P2 |
| 5 | Alias-free / DeStripe / wavelet | Med (StyleGAN3 foundational; 3D transfer open) | Low (notch) / Med (netG) | Low | [X] | DeStripe is the on-point P2 fix; contained |
| 2 | Diffusion priors | Med (mature lineage; materials + few-step open) | **High** (new class) | Med | **[V]** | Big swing / 2nd paper; CRIS & D2R are close prior art |
| 6 | Lightweight/Mamba 3D | Med (no 3D-GAN Mamba) | High | **High** (env fork) | [X] | Last — highest risk, only new *capability* |

**One-paper path from the current campaign**: **1 + 3** — modernized objective/feature-D on the
existing lattice-vs-texture Pareto framework, plus FSQ + next-scale entropy model to give a real
bits-per-voxel RD curve. Both reuse everything built; both have strong recent anchors (R3GAN [V],
ARPC/VAR [X]).

**Cheap high-value add to either**: **4a** (Gaussian-NLL uncertainty head) — one output channel,
py38-safe, gives P5 *and* a published mechanism to diagnose the p2z stripe (P2).

**Bigger swing / 2nd paper**: **2** (diffusion-prior isotropy on fuse data) — but note **CRIS
(2606.15967) and D2R (2411.16792) are close prior art**; the defensible novelty is the
materials-science modality + a few-step distilled refiner at practical cost.

## Provenance & caveats

- Track A (dirs 1, 2) findings are **3-vote adversarially verified**; dirs 3–6 rest on
  **single-agent quote-backed extraction** (outside the 25-claim verify budget) — citations are
  real, author-reported numbers are unaudited. Two claims were formally refuted and excluded
  (Projected-GAN "40× faster" framing; D2R's exact averaging pipeline).
- All performance numbers are authors' self-reported benchmarks in **medical/EM/natural-image**
  domains; **none on materials-science volumes**. That modality gap is itself the recurring
  novelty hook.
- Knowledge current to ~mid-2026 (sources through 2026-06); no CVPR 2026 main-proceedings sweep.
- Before committing a direction: re-verify its [X] citations, and land the two pending
  current-campaign items first (skipU-only decoupling run; float blur-KID redo from
  checkpoints) since direction 1's ablation builds on them.
- Full research artifacts: workflow run `wf_4f60991e-8dd`, journal at
  `…/subagents/workflows/wf_4f60991e-8dd/journal.jsonl`.
