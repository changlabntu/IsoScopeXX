# MS_refine2: further model-search directions for the multi-scale models

Constraints: (a) **one decode pass** producing all scale outputs (out0 192³ / out128 96³ /
out64 48³); (b) **~unchanged parameter count**; objective = best **KID + isotropy-LPIPS**.
Continues `MS_refine.md` (v1: directions 1–5, of which 1+2 → `vqcleanM0aMSskip`, now training).

## Mechanism analysis

KID punishes unrealistic slice texture; the isotropy LPIPS (XY vs YZ of the prediction)
punishes directional asymmetry. Current architectural weaknesses against them:

1. Slices are encoded independently by the 2D VQ encoder and **nothing mixes Z until `up3`** —
   the trunk must invent all cross-slice coherence three layers deep.
2. ConvTranspose3d (k4, s2) upsampling is **checkerboard-prone** — a classic KID artifact source.
3. The generator has **no stochastic source** for high-frequency texture; the deterministic
   per-slice latent can't supply Z detail, so YZ planes stay smoother than XY.
4. The coarse heads are supervised only against the model's **own pooled output**
   (pyramid loss), never against real data.
5. Validation/inference use **raw weights, not an EMA**.

## Direction catalog

### Tier F — free: flags or loss-only, zero params, zero extra passes

- **F1. Generator EMA for eval/checkpoints.** `--use_ema`/`LitEma`/`ema_scope()` already exist
  in the model files, but the validation loop never enters `ema_scope`, so EMA currently
  changes nothing even when enabled. Wire `validation_step` + checkpoint saving to EMA
  weights. Canonical, near-universal KID improvement; the weight copy is buffers, not
  trainable params. **Do this first.**
- **F2. Real-data supervision of the coarse heads.** Add the L1 Z-projection loss per scale:
  project out128/out64 (same `get_projection` logic, scaled `uprate`) against XY-avg-pooled
  `oriX` (÷2, ÷4). Grounds the heads in observed data; complements `--adv_ms`.
- **F3. Structural projection losses.** Existing `--lbm_ms_ssim` (currently 0), or a
  Laplacian-pyramid L1 on the Z-projection instead of plain L1.
- **F4. Discriminator feature matching.** L1 between D intermediate features of real XY
  slices and generated slices. Classic perceptual/stability win; needs `patch_16` to expose
  features (check `networks/cyclegan/models.py`; `add_loss_adv`'s `net_d(a)[0]` indexing
  suggests a tuple already exists).

### Tier Z — tiny params, aimed squarely at the isotropy metric

- **Z1. Zero-init Z-mixing latent stage** (~768 params). Right after `decoder.conv_in`:
  `x = x + zconv(x)` with a depthwise Conv3d, kernel (1,1,3), zero-init (same ControlNet
  philosophy as MSskip → step-0 identical to MSskip). Directly attacks weakness 1 — the
  cheapest possible cross-slice communication, applied at latent resolution where it costs
  almost nothing. **Top architectural pick.**
- **Z2. Stochastic noise injection (StyleGAN-style)** (~224 params). Additive Gaussian noise
  with learned per-channel scales (zero-init) after each up stage. Gives the GAN a source of
  high-frequency Z texture the latent cannot supply — targets KID and the YZ-smoothness half
  of the LPIPS gap. Eval uses noise=0 (or a fixed seed) for deterministic outputs.
- **Z3. Anti-checkerboard upsampling.** The generator ctor already has `use_upsample`
  (Upsample+Conv3d instead of ConvTranspose3d) but the registry never passes it → new netG
  file with it defaulted on. Slightly FEWER params (27·Cin·Cout vs 64·Cin·Cout per stage).
  Removes weakness 2. Caveat: all trunk weight shapes change → fresh training, no step-0
  equivalence with any prior run.

### Tier S — structural, ~1% params

- **S1. Residual-over-trilinear output.** Heads predict a delta over the trilinear upsample
  (`XupX = Xup + delta`; head final activation → none; pooled `Xup` for the coarse heads).
  Guarantees the data-fidelity floor, focuses capacity on detail, typically faster
  convergence.
- **S2. Bottleneck attention at 24³.** One vanilla attention block on the trunk input,
  mirroring the 2D VQGAN's bottleneck (~0.26M params ≈ +0.9%, ~49 GMAC). Global 3D context
  for long-range structure; the only entry with non-trivial param/compute cost.

### Explicitly out (violates constraints or already resolved)

- Multi-pass schemes (dsp-phase consistency, partial-sum decodes) — break the 1-pass budget.
- Bigger adapters / post-injection blocks — remains MS_refine v1 direction 4, the designated
  fallback if MSskip still trails MS.
- Extra discriminators per orientation — D params are training-only, but the six-way D
  already dominates step cost; deferred.

## Recommended sequencing (one model file per iteration, per convention)

1. **`vqcleanM0aMSskipE`** = MSskip + F1 (EMA in val/checkpoints) + F2 (real-data coarse L1).
   Zero params, step-0 == MSskip. Safe to bundle: F1 only changes *eval* weights, so it never
   confounds training dynamics; EMA alone often moves KID more than architecture tweaks.
2. **`vqcleanM0aMSskipZ`** = previous + Z1 (zero-init Z-mixing conv). The most
   mechanism-aligned change for the isotropy metric.
3. **`vqcleanM0aMSskipN`** = previous + Z2 (noise injection), if the GIFs still show YZ
   texture lagging XY.
4. Z3 / S1 / S2 afterward as independent probes, chosen by which gap persists:
   KID gap → Z3; L1/fidelity gap → S1; long-range structure errors in GIFs → S2.

Every addition after F1 is one-at-a-time per iteration for attribution. Trigger point for
iteration 1: once MSskip has ~50 epochs to compare against MS at matched windows.

## Experiment log (as of 2026-07-06)

All runs on THX10SDM20xw/roiD (192×192×24 → 192³ isotropic, num_scales 4), tracked in
`$LOGS/thx-MS.db`. Window means at matched epochs; ↓ better.

| run | model change | val_lpips_pred | val_kid | verdict |
|---|---|---|---|---|
| MS | baseline vqcleanM0aMS (summed latent, heads as taps; lr 5e-4, adv_ms 0) | 0.589 (best 0.569) | ~5.6 (best 2.6) | reference |
| MSfpn | strict FPN: trunk gets coarse codes only, fine scales injected after taps | 0.628 | 7.5 | NEGATIVE (stopped @95ep) |
| MSskip | full-sum trunk + zero-init skips; pyr_detach, adv_ms 0.5, lr 2e-3 | **0.578** (best 0.549) | ~5.4–6.0 | WIN on both vs MS |
| MSskipE | + EMA at eval + real-data coarse L1 (l1c) | 0.616 (plateau) | **4.96** (still falling) | SPLIT: KID champion, LPIPS regression |
| MSskipP band | + band-limited dense projection L1 | 0.625 | 6.95 | NEGATIVE (stopped @139ep) |
| MSskipP lse | + soft-max projection, random phase | queued (run.sh ready) | | |

Current best: MSskip for isotropy-LPIPS (0.578), MSskipE for KID (4.96). No run holds both.

Lessons per iteration:

1. **MSfpn** — strict scale→head visibility costs quality: the fine scales (576/765
   tokens/slice) got one 3D stage of decode depth instead of eight, and the Z-smeared late
   injection showed directly in the XY-vs-YZ metric (barely beat the 0.634 trilinear
   baseline). Converged glacially toward at-best parity.
2. **MSskip** — the same skips as REINFORCEMENT (full-sum trunk, zero-init adapters,
   step-0 == MS) + loss fixes overtook MS on both metrics by ~epoch 70. Caveat: lr and
   loss config changed together with the architecture; the MS+pyr_detach/adv_ms control
   was never run.
3. **MSskipE** — EMA is real but double-edged: best KID of any run (was silently dead code
   before — the LitEma import path didn't exist), but LPIPS plateaued 0.04 above MSskip.
   Suspects: the ~65-epoch EMA horizon at decay 0.9999 (weight averaging smooths the
   high-frequency YZ texture the isotropy metric wants) and/or l1c drag. Untested probes:
   --ema_decay 0.999, --lamb_coarse ablation.
4. **MSskipP band** — the old 'max' projection was secretly load-bearing: band anchored
   the measured Z-band as designed, but the freed high-frequency band filled with
   axis-aligned CHECKERBOARD — ConvTranspose wants to emit it, the stride-2 patch D
   aliases it into invisibility, and max-projection had been the only damping. Spectrally
   stable over 130 epochs (not a transient). band is only viable after an
   anti-checkerboard generator.

Standing conclusions: latent skips help; visibility restriction hurts; zero-init is the
right way to add anything (three iterations started exactly at their parent's quality).
The **Z3 anti-checkerboard generator is now the most-motivated next architectural
iteration** — implicated in both skipE's LPIPS ceiling and band's failure, and the
prerequisite for retesting band. Next: MSskipP-lse, then Z3.

## Status (2026-07-05)

- MSskip result at matched windows (epochs 96–121): val_lpips_pred 0.580 vs MS 0.598,
  val_kid 4.73 vs 5.75 — beats the baseline on both target metrics (and beats MS's final
  131–156 window too). Early-epoch handicap (KID 11.1 at 2–10) from lr 0.002 + fresh coarse
  discriminators resolved by ~epoch 45.
- **Iteration 3 implemented (queued after skipE):** `models/vqcleanM0aMSskipP.py` — projection
  forward-model variants for the L1 anchor. `--l1how band` (primary probe: Gaussian-low-pass
  both prediction and trilinear-upsampled observation along Z, dense L1 — full agreement in
  the measured band, zero constraint on the high-Z frequencies the GAN must invent),
  `--l1how lse` (+`--lse_tau`; bias-corrected soft-max projection, tau→0 = max), and
  `--l1phase random` (per-step window phase for mean/max/lse, shared across main and coarse
  losses). Coarse `l1c` follows the same forward model at halved/quartered rates. Run line
  in `run.sh` changes only `--l1how` vs the skipE run.
- **Iteration 1 implemented:** `models/vqcleanM0aMSskipE.py` (F1+F2). EMA always on
  (`--ema_decay`, LitEma from `ldm/modules/ema.py` with num_updates warmup); validation
  loop and epoch checkpoints run under EMA weights via `on_validation_start/end` and a
  `training_epoch_end` wrapper. Real-data coarse L1 (`--lamb_coarse`, logged `l1c`):
  out128/out64 Z-projections vs XY-avg-pooled `oriX`, same `skipl1`/`--l1how`, halved/
  quartered uprate (requires uprate % 4 == 0). Run line in `run.sh` (commented, lr 0.002
  matching the MSskip run).
