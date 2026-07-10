# CRISIS: audit of MS-campaign conclusions under the precrop-blur confound

Independent adversarial audit (2026-07-09) of every scientific claim in the MS campaign,
re-examined under the `--precrop` default bug fixed in commit 1c11827: until then, all
THX10SDM20xw/roiD training patches (384² in-plane) ran `CenterCrop(256)+Resize(→384)`,
leaving ~63% of native mid-band and ~19% of native high-frequency energy in the training
data. 256²-in-plane datasets (roiAdsp4, E2507218cube) were untouched.

## Ground truth established first (from code and logs, not docs)

- **Train blurred, val native — confirmed in code.** `dataloader/data_multi.py:
  get_transforms()` puts `A.CenterCrop(precrop)+A.Resize` only in the `'train'` pipeline;
  the `'test'` pipeline is `Resize` only (no-op at native size), and `train.py:111-113`
  builds the val split with `mode='test'`. So every roiD run trained on blurred patches
  while **all validation inputs, metric references, and inference-study inputs were
  native-sharp**. `cfg/aisr.yaml` does not set `precrop`; the argparse default 256 hit
  every roiD run.
- **The fuse corpus is clean.** E2507218cube volumes are 256³ (verified via tifffile);
  CenterCrop(256)+Resize(256) were no-ops.
- **The lattice diagnostic was on fuse, not roiD.** The `val_epoch_110` GIF matching the
  diagnostic geometry in `doc/research_artifact_directions.md` (128 frames, 4×256²
  panels) is `logs/fuse-MS/d2890b19.../artifacts/images/val_epoch_110.gif` (2026-07-07)
  — the fuse-skipE run. The 17–94× lattice ratios were measured on clean-data training.
- **Two post-diagnosis runs are themselves contaminated** (snapshot `config.json`s):
  **MSskipUB thx10** (20260708_183437) and **vqcleanMH Scale4** (20260709_134948) both
  trained with `precrop: 256` on roiD — blurred. Only the 20260709_2011xx vqclean
  relaunch has `precrop: 0`.
- **Recurring second-order mechanism:** the six-way D defined "real texture" from blurred
  slices (~19% native HF), so **no training signal ever demanded native high frequency**;
  the HF band was adversarially unconstrained for every roiD run, while every val metric
  scored outputs against native-sharp references and fed the blurred-trained (now OOD)
  2D VQ encoder native input.

## Per-claim verdicts

| # | Claim | Verdict |
|---|---|---|
| a | MSfpn NEGATIVE (starved full-res head) | **SURVIVES** |
| b | MSskip WIN, LPIPS 0.577 champion | **WEAKENED** (mostly pre-existing) |
| c | skipE KID champion via EMA; ema999 "genuine trade" | **WEAKENED** |
| d | band NEGATIVE; "sparse max was the only damping" | **WEAKENED** |
| e | lse NEGATIVE (mild) | **SURVIVES** (with caveat) |
| f | Alias-lattice diagnosis; MSskipU kills it, KID regresses | **SURVIVES** |
| g | val_spec gap (vqclean 0.84 vs MS 0.46 vs skipE 0.25) | **INVALIDATED** |
| h | Sharpness ranking vqclean > MS > MSskipE | **INVALIDATED** |
| i | vqcleanMH pivot premise | **INVALIDATED as evidenced** (decision salvageable) |

### a) MSfpn NEGATIVE — SURVIVES

MSfpn vs MS is like-for-like: identical roiD blur, identical val handling, matched-epoch
windows, decisive margin (LPIPS 0.637 barely above the 0.634 do-nothing baseline; KID gap
not closing). The mechanism (finest scale gets one 3D stage; injection is Z-smooth) is
architecture-level. If anything the blur *understates* the penalty: the 24×24 scale
carries most in-plane detail, and blur removed exactly that detail — on native data the
starvation should be worse. No plausible way the confound flips this sign.

### b) MSskip WIN — WEAKENED (by its own admitted confound, not the bug)

MSskip vs MS is internally valid under the blur (both trained and scored identically).
What does not survive is what the docs concede: lr (5e-4→2e-3), `pyr_detach`, and
`adv_ms 0.5` changed together with the architecture; the `MS + loss-config` control was
never run — "add paths, never remove them" is attributed, not isolated. The *absolute*
0.577 is now meaningless: a blur-trained model with an OOD-native input, judged against
native-sharp XY. Champion status holds only inside the blurred thx-MS family.

### c) skipE / EMA — WEAKENED

The internal ordering (skipE best KID; ema999 monotone between parents) survives as a
qualitative statement *in this regime*. But "EMA mutes the HF YZ texture the isotropy
metric rewards; genuine trade" is confounded twice. (1) Under blurred training, the
raw-weight HF that EMA averages away is mostly *unconstrained hallucination/lattice*
(the D never saw native HF to anchor it); the sharp-native val reference rewards any HF,
so raw weights get LPIPS credit for noise and EMA is penalized for removing it. On fixed
data the LPIPS cost of EMA could shrink substantially — the 0.04 magnitude is untrusted.
(2) skipE changed EMA *and* `lamb_coarse` together (A2 control still pending — and the
coarse L1 target was *blurred* `oriX`). Independently shaky:
`KernelInceptionDistance(feature=64, subset_size=16)` (`models/base.py:139`) uses the
lowest Inception block and tiny subsets — every KID margin in the campaign is
low-level-statistics with high variance.

### d) band NEGATIVE — WEAKENED

Single-variable attribution is clean, and the ConvTranspose-lattice mechanism was
independently confirmed on *clean* fuse data — "the trunk wants to emit the lattice"
stands. But "the freed Z-band *inevitably* fills with checkerboard; sparse max was
secretly the only damping; projection line CLOSED" was measured in the worst-case regime:
training data had ~19% of native HF, so the freed band had almost **no real content the
six-way D could demand instead** — the lattice won by default, not necessarily by
necessity. On native data the D's YZ-vs-XY comparison would push real texture into that
band. Band retest now double-gated: anti-checkerboard netG (as already planned) AND
fixed data. "Band NEGATIVE as run" survives; "max stays, line closed" does not.

### e) lse NEGATIVE — SURVIVES, weakly

Clean single-variable design, blurred regime on both sides; the mechanism (soft-max adds
diffuse smoothing pressure) is data-independent and directionally plausible on native
data. But the effect was "mild" atop an already-smoothed regime scored against a sharp
reference that amplifies small smoothing deltas — effect *size* untrusted. Never
load-bearing; do not spend a rerun on it unless band is retested anyway.

### f) Alias-lattice line — SURVIVES (fully insulated from the bug)

Verified: the FFT diagnostic GIF is from fuse-skipE (clean 256³ data), and
MSskipU-vs-fuse-skipE (p2diag 94→1.3; KID 5.42 vs 3.13 at matched steps) is a
single-variable netG swap, both runs on the untouched dataset, same `lamb_coarse 0`.
The lattice is architectural and resize-conv removing it is directly measured. Ordinary
caveats remain (interim single seed, KID feature=64 fragility; "resize-conv
under-produces HF" is an inference from KID, not a spectral measurement). BUT: the
**thx MSskipUB run meant to complete the pair trained on blurred roiD (precrop=256)**,
and its success thresholds (KID ≤4.97, LPIPS ≤0.577) are blurred-family numbers — it can
answer "does BlurPool help vs blurred-thx-skipE" internally, nothing absolute.

### g) val_spec gap — INVALIDATED

`models/base.py:448-469` computes `val_spec_recon` as the 2D VQ recon's spectral
retention **vs the val input `oriX` — native-sharp**, while every MS model's
encoder/codebook/decoder trained exclusively on blurred slices and *never saw the band it
is asked to retain*. vqclean trained on roiAdsp4 (256² — clean, no-op transform). So
"0.84 vs 0.46 vs 0.25" measures **train/val domain match, not architecture or recipe**;
the fix commit itself calls the bug "almost certainly the dominant cause of the roiD
recon_mid deficit". Already shaky pre-bug: cross-dataset/resolution (roiAdsp4 128/16 vs
roiD 192/24), unmatched training age (ep600 vs ep100), train-mode MC-dropout draws, and
vqclean's own trajectory (1.07→0.94→0.84 over ep100→600) shows the metric drifts with
training age alone (values >1 credit hallucinated energy). Survives only
vqclean-internally: recon_mid 0.84 vs xy_mid 0.39 on the same model/volume ("the trunk
renders less than the latent carries") — itself on a roiAdsp4→roiD domain shift.

### h) Sharpness ranking — INVALIDATED as a model ranking

Compares a clean-trained model (vqclean/roiAdsp4) against blur-trained models (roiD) on
a **native-sharp input volume**: the MS models are softer both because their texture
prior was blurred *and* because their 2D encoder is OOD on native slices — on top of the
doc's own caveats (one MC-dropout draw in train mode, unmatched epochs, different
training direction). "MS lowers peak XY sharpness" cannot be attributed to MS at all.
Survives: vqclean's YZ ripple/lattice observation (architectural, later quantified on
clean fuse data) and the study's useful output — it motivated `val_lat_*`.

### i) vqcleanMH pivot — premise INVALIDATED as evidenced; decision premature, not disproven

The premise — "the vqclean recipe preserves texture that the MS recipe destroys; keep the
recipe, add read-only heads" — rests on (g) and (h), both dominated by the data bug. The
decisive un-run control is **skipE/MS on FIXED roiD**: if their recon_mid/xy_mid recover
toward vqclean's on clean data, the recipe was never the problem and MH solves a phantom.
Aggravating: the first MH run (Scale4, 20260709_1349xx) trained blurred — discard it. The
team's mitigation (thx-MS-384 store + vqclean-on-roiD benchmark, precrop 0) re-pins the
*reference* but still omits the MS-recipe-on-clean-data control. In MH's favor: read-only
detached heads are near-free by design, and lr 2e-4 / no-pyramid was never shown
*harmful* — the pivot may survive, but currently on priors, not evidence.

## Most dangerous now-wrong (or unsupported) beliefs, ranked

1. **"The MS recipe/architecture destroys mid-band texture (recon 0.25–0.46 vs vqclean
   0.84)."** Mostly train/val domain mismatch. Acting on it (the MH pivot, abandoning the
   skipE lineage) redirects the whole campaign around a data bug.
2. **"MS trades XY sharpness for morphology vs vqclean."** Same root cause; risks a
   permanent belief that the multi-scale line has an inherent fidelity ceiling.
3. **"The projection line is closed; `max` is the only viable forward model; band's null
   space inevitably fills with checkerboard."** Measured when the freed band had almost
   no real HF for the D to demand; band may be viable on fixed data + resize-conv.
4. **"EMA 0.9999 costs ~0.04 LPIPS — a confirmed genuine trade."** The smoothing penalty
   is inflated by hallucinated-HF-vs-sharp-reference scoring; retune decay on clean data
   before baking dual-eval complexity in.
5. **All absolute thresholds — LPIPS 0.577 / KID 4.97 as success gates (run.sh UB
   comment) and the Kubeflow "KID 3.9" long-run — are blurred-family numbers.** Any
   comparison spanning the fix boundary inside thx-MS is invalid; the blurred MSskipUB
   and Scale4-MH runs must not be scored against thx-MS-384 runs.

## Minimal rerun set to re-establish the load-bearing conclusions

1. **(Already queued, run.sh)** vqclean on fixed roiD (thx-MS-384) — pins the clean
   reference for val_spec/LPIPS/KID at this geometry.
2. **skipE (declared baseline config) on fixed roiD, matched optimizer steps to (1)** —
   the single decisive run. If val_spec_recon/xy recover, claims (g)(h)(i) collapse and
   the MS lineage is rehabilitated; if not, the recipe blame regains support. Everything
   else ranks below this.
3. **Relaunch MSskipUB on fixed roiD** (current one is blurred) — the anti-alias pair
   conclusion (f) needs its second half on data that will remain the standard.
4. **EMA-decay ablation on clean data** — cheapest as fuse-skipE with `--ema_decay
   0.999`, or fold into (2) via dual-eval (raw + EMA checkpoints) — re-sizes the (c) trade.
5. **band retest** only after (3) lands and only on fixed data (double-gated).
6. **Redo the sharpness study**: all models in `--eval`, matched steps, checkpoints from
   (1)+(2), scored with `utils/metrics_spectral.py` — inference-only, hours not days.
7. Optional hygiene (pre-existing debt): the `MS + pyr_detach/adv_ms/lr 2e-3` control for
   claim (b); reconsider `KernelInceptionDistance(feature=64, subset_size=16)` — every
   KID margin flows through it.

Claims needing no rerun: (a) MSfpn stays dead (blur biased against finding this negative,
yet it was found decisively); (e) lse stays closed unless band reopens; (f)'s lattice
elimination stands as measured.
