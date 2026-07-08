# Suggestions: fixing the diagonal lattice / structure-beading artifact

Action plan distilled from `doc/research_artifact_directions.md` (deep-research
report + FFT diagnostics on val_epoch_110). The symptom: structures in the enhanced
output bead along diagonal chains, following a stride-2 alias lattice (period-2/4
diagonal peaks up to 94×/21× background in the lateral planes; confirmed in both
X–Z and Y–Z orientations).

## Baseline to build on

**`vqcleanM0aMSskipE`** — per the baseline declaration in `doc/experiments_MS.md`
(2026-07-07). All iterations below chain from `models/vqcleanM0aMSskipE.py` and are
measured against its numbers, using its run line:

```
--lamb 5 --lr 0.002 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --l1how max
--netG ed023emsfpn  (num_scales 4)
```

Rationale: the artifact lives in the 3D netG trunk (ConvTranspose3d) and the
discriminator's stride-2 blindness — both orthogonal to skipE's validated
ingredients (full-sum trunk + zero-init skips, EMA-at-eval, coarse L1). The two
open knobs (`--lamb_coarse`, `--ema_decay`) stay CLI-switchable and unaffected.
Caveat: the Z3 iteration cannot start step-0-equal to skipE (trunk weight shapes
change), so compare at matched optimizer steps, not epochs.

## Ranked fixes

### 0. Lattice-peak validation metric (do immediately, before any fix)

**IMPLEMENTED (2026-07-07):** `utils/metrics_spectral.py` + hooks in
`models/base.py` (`validation_step` accumulates mean power spectra of both
lateral plane orientations across batches — Welch averaging; ratios computed in
`validation_epoch_end`). Logs `val_lat_{p2diag,p2a,p2z,p4diag}` to MLflow/TB and
the console line; measures EMA weights like LPIPS/KID. ~1 = clean; the diagnosed
lattice showed p2diag up to ~90×. Unit-tested against planted checkerboards
(clean noise → ~1; planted p2 diag → >300×; axis-selective).

Rationale: LPIPS/KID see the artifact only indirectly; this measures it directly,
giving every iteration a clean verdict ("lattice gone?" separate from "quality
better?") and settling whether Z3 alone suffices.

### 1. Z3 — resize-conv trunk (`--netG ed023emsfpnu`)

**IMPLEMENTED (2026-07-07):** registry-only — `ed023emsfpnu` (and non-fpn twin
`ed023emsu`) registered in `networks/registry.py`; membership in
`UPSAMPLE_GENERATORS` makes the `ed` branch pass `use_upsample=(2,2,2)` (never
`True`: the value doubles as `nn.Upsample`'s scale_factor). All three up stages
become nearest-Upsample + Conv3d(k3,s1,p1) via the existing `deconv3d_bn_block`;
zero ConvTranspose3d remain (smoke-tested; decode output shapes match baseline,
3.45M vs 5.04M generator params). `ed023eMSfpn.py` itself is untouched.

Removes the lattice SOURCE (Distill 2016; period-4 diagonal proves at least two
stages imprint, so all stages are replaced). Nearest (not trilinear) upsampling:
Distill's default, the k3 conv can learn the tent filter, and trilinear's
smoothing is the exact YZ-texture pressure the skipE post-mortem warns about.
Fresh trunk weights — no step-0 equivalence; compare at matched optimizer steps.

### 2. BlurPool the discriminator (own iteration, after Z3)

**IMPLEMENTED (2026-07-07):** `--netD patchblur_16` — `BlurPool2d` (fixed
[1,2,1]⊗[1,2,1]/16 binomial, non-persistent buffer, zero learned params) in
`networks/cyclegan/models.py`; `Discriminator(blur=True)` converts every one of
the 4 stride-2 blocks to Conv(k4,s1) → InstanceNorm → LeakyReLU → BlurPool(s2)
(blur last, after the nonlinearity, per Zhang). Output patch maps identical to
plain patch_16 (16×16 @256, 12×12 @192, smoke-tested); parameter shapes/order
identical (warm-start from a plain-D checkpoint needs a key remap — Sequential
indices shift). Registry branch checked BEFORE the `patch` prefix; patchblur_4/8
come free. In skipE, `--netD` drives net_d + net_d_128 + net_d_64 → all three Ds
upgraded; the LDM VQ-loss internal D is untouched (scores lattice-clean XY recon).

The X–Y plane analysis showed period-2 aliases at the D's first layer and
period-4 at its second — each depth is blind at its own Nyquist, so all layers
are blurred. This is the other half of the mechanism: a resize-conv G can drift
back to grid-anchored solutions if the D stays blind (Schwarz: the training
signal, not the architecture, decides what persists). Separate iteration for
attribution; expect the PAIR (1+2), not either alone, to fully kill the pattern.
Empirical caution: blur also attenuates what the D must detect — watch the
lattice metric.

### 3. Spectral (FFT-magnitude) discriminator alongside spatial + focal frequency loss

Targets content-follows-the-grid directly: diagonal orientation-energy excess is
invisible to patch_16's receptive field but first-class in the Fourier magnitude
plane (Luo ICCV 2023: spectral D wins at high freq, spatial at low — use both).
FFL as a cheap loss-only companion (D2R weights it λ=100). Loss-side, single pass.
Also the prerequisite for ever retesting `--l1how band` (its null space filled with
exactly this lattice).

### 4. Z2 — zero-init noise injection (only if beading persists after 1–3)

Learned per-channel Gaussian noise scales (zero-init, step-0 == parent) after each
up stage. Removes the INCENTIVE to anchor on the grid: the deterministic per-slice
latent supplies no other high-frequency seed for texture nucleation. Mechanism-
argued, not literature-verified — stays behind the evidence-backed fixes.

### Deferred / not recommended now

- StyleGAN3-style filtered nonlinearities: the deep fix, heavy 3D engineering;
  reserve for residue after 1–4.
- Quantizer changes (FSQ etc.): the artifact is in the 3D trunk, not the VQ — the
  trilinear-upsampled coarse panels were lattice-clean; FSQ reportedly degrades in
  residual stacks anyway.
- Retesting `--l1how band`: only after 1–3 land.

## Sequence (0–2 implemented 2026-07-07; run lines ready in run.sh)

1. ~~FFT lattice metric into `base.py`~~ DONE (`val_lat_*`, active for every run).
2. **MSskipU** = skipE + `--netG ed023emsfpnu` (resize-conv trunk) — RUNNING on
   fuse. Interim @ step ~10.4k (2026-07-08): lattice ELIMINATED (p2diag 1.3 /
   p4diag 1.1 vs 17–94× before; no beading in GIFs), LPIPS parity-to-better,
   but KID regressed 5.4 vs 3.1 at matched steps — resize-conv under-produces
   high frequency and the stride-2 D doesn't demand it (Schwarz's predicted
   trade). Full verdict in doc/experiments_MS.md.
3. **MSskipUB** = MSskipU + `--netD patchblur_16` — run line in run.sh; now the
   PRIORITY run: BlurPool is the mechanism that would make the D see (and
   demand) the missing high frequencies. The UB-vs-U delta isolates it.
4. Spectral D / FFL if orientation bias outlives the lattice peaks.
5. Z2 noise injection if YZ texture still lags.

Open question to carry along: the measured X↔Y asymmetry (pure-X period-2 strong,
pure-Y near-clean despite a symmetric trunk) — check on raw float tensors (not the
8-bit GIF) and look for an axis-asymmetric op (downbranch/resizebranch latent
reshaping, non-cubic val crop) once the metric is in place.
