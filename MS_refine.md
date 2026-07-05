# MS_refine: directions for improving KID/LPIPS of the multi-scale models

## Problem

`vqcleanM0aMSfpn` (FPN scale-to-depth injection: trunk gets only the coarse partial sum,
`quants[-2]`/`quants[-1]` injected laterally after the out64/out128 taps) shows **worse KID and
val_lpips** than the non-FPN `vqcleanM0aMS` (full sum into the trunk, heads as pure taps).

Diagnosed causes, in order of suspicion:

1. **Non-detached pyramid loss blurs out0.** The THX10 FPN run used `--lamb_pyr 1.0` without
   `--pyr_detach` and with `--adv_ms 0`. The coarse heads are architecturally unable to express
   fine content, so the two-way L1 pulls the pooled full-res output toward what coarse codes can
   express — direct blur pressure on `out0`, unopposed by any coarse adversarial loss.
2. **The finest scale gets almost no 3D processing.** The 24×24 scale carries 576 of 765
   tokens/slice (most in-plane detail). Non-FPN gives it the full trunk (~8 conv layers,
   3 learned Z-upsamples); FPN gives it only `up1` + output conv. Its injection is per-slice 2D
   content trilinearly stretched 24→96 in Z, so it arrives XY-sharp but Z-smooth, and one
   ConvTranspose3d must make it 3D-consistent → XY slices sharper than YZ slices → exactly what
   the XY-vs-YZ `val_lpips_pred` isotropy metric punishes (and KID sees too).
3. **Cold start.** Random-init `inject_convs` + a trunk relearning from a much coarser input
   (45 vs 765 tokens/slice) → FPN lags at equal epochs even if it would converge. Check whether
   the gap shrinks over training before concluding.

## Constraints for refinement

- (a) **one decode pass** producing all three outputs (out0 192³, out128 96³, out64 48³);
- (b) **~same parameter count**;
- objective: best KID / LPIPS.

Note: these constraints do NOT require the strict scale→head visibility contract — and the
contract is what costs quality. That opens the design spectrum below.

## Directions (ranked by expected quality-per-effort)

### 1. Loss-config fixes (zero cost — do first, before judging any architecture)

Run with `--pyr_detach --adv_ms 0.5` and consider `--lamb_pyr 0.5`. Removes the self-inflicted
blur drag (cause 1) and gives the coarse heads adversarial sharpening. No params, no compute,
no code change.

### 2. Full-sum trunk + keep injections as reinforcement skips  ← recommended next run

Change `generation()` trunk input from `sum(quants[:num_scales-2])` back to `sum(quants)`,
but KEEP the lateral injections of the fine scales at pre_up2/pre_up1.

- Every code gets full decoder depth → out0 quality floor = non-FPN baseline by construction.
- Fine codes additionally reach the late layers through a short path; the trunk has no skip
  connections at all today, so a high-frequency shortcut past ~8 layers plausibly IMPROVES
  sharpness over the baseline rather than just matching it.
- Same params (reuses existing `inject_convs`), one pass, three outputs.
- Sacrifice: the hard visibility contract (coarse heads still produce meaningful coarse
  outputs, just not provably-from-coarse-codes).

### 3. Keep a partial contract: move injections one stage earlier

Inject `quants[-1]` before `conv6` (after `up2`) instead of after `conv6`. The finest scale then
gets `conv6` + `up1` (two 3D stages incl. a learned Z-upsample instead of one) — directly
targets the Z-smearing mechanism (cause 2). out64's contract survives; out128's is sacrificed
(it then sees all scales).

### 4. Spend the parameter budget at the injection site, not the adapter

- A small 3×3×3 Conv3d block right after each injection point (e.g. one 2nf→2nf conv at the 96³
  stage ≈ 110K params, +0.4% of 28M) gives injected content local 3D context / Z-consistency.
- Or make the adapter Z-aware: apply it AFTER the trilinear resize, or use a ConvTranspose
  along Z so the 24→96 stretch is learned rather than trilinear.

### 5. Enforce the scale hierarchy in the latent instead of the decoder

If the coarse→fine code hierarchy is the actual goal: revert the 3D path to full-sum
(quality-optimal) and add a cheap 2D auxiliary loss — decode each partial sum with the existing
2D decoder and L1-match appropriately pooled input slices. Codes learn hierarchy; zero new 3D
compute; one pass unchanged.

## Recommendation

Combine **1 + 2** for the next run: a one-line change in `models/vqcleanM0aMSfpn.py`
(`generation()` trunk input → `sum(quants)`) plus `--pyr_detach --adv_ms 0.5` in the run line.
Cannot be worse than the non-FPN baseline by gradient-path construction, keeps everything built
so far. If coarse-output provenance matters later (preview/compression tiers), follow up with
3 or 5 to restore a softer hierarchy.

Whichever variant runs: **compare at matched epochs** — the FPN carries a cold-start handicap
from its random adapters, so early-epoch comparisons systematically flatter the incumbent.

## Status / decisions (2026-07-04)

Matched-epoch MLflow data (`thx-MS.db`, runs `MS` vs `MSfpn`) confirmed the diagnosis: at epochs
55–73, val_lpips_pred 0.637 vs 0.602 (MSfpn barely beats the 0.634 trilinear baseline), val_kid
7.67 vs 5.94, pyr plateau 50% higher, gap not closing.

- **Directions 1+2: implemented** as `models/vqcleanM0aMSskip.py` (reuses `--netG ed023emsfpn`
  unchanged). Added refinement: the `inject_convs` are **zero-initialized** (ControlNet-style),
  so with the full-sum trunk the model is numerically identical to `vqcleanM0aMS` at step 0 —
  no cold-start handicap; skips can only learn to help. Run line in `run.sh` with
  `--pyr_detach --adv_ms 0.5`.
- **Direction 3: dropped** — its rationale (more decode depth for the fine scale under a
  restricted trunk) is dissolved by the full-sum trunk of direction 2.
- **Direction 4: deferred** — in skip mode the trunk carries the fine codes anyway and can
  suppress a harmful Z-smeared skip, so 4's benefit is speculative and bundling it would
  confound attribution. Next iteration (own model file + netG with a small post-injection
  Conv3d block) only if MSskip still trails MS on LPIPS.
- Attribution caveat: the MSskip run changes loss config (1) AND architecture (2) vs the `MS`
  baseline. If attribution is needed, run `vqcleanM0aMS` + `--pyr_detach --adv_ms 0.5` as the
  control.
