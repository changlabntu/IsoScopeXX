# Wire the multi-scale VQ latent into the multi-scale 3D output (true coarse-to-fine)

## Context

`vqcleanM0aMS` currently carries **two unrelated** notions of "multi-scale":

1. **Latent pyramid** (`--num_scales`, VAR-style residual VQ in `encode()`): quantizes the 2D
   encoder latent at coarse→fine in-plane grids (for num_scales=4: 3²/6²/12²/24²), but **sums
   every scale back into one isotropic 24³ latent** before it ever reaches the 3D generator.
2. **Output pyramid** (`--netG ed023ems`): the generator emits the volume at 48³/96³/192³
   (out64 / out128 / out0), where the two coarse heads are just **decoder-depth taps of that
   single summed latent** — supervised toward `avg_pool3d(out0)` (pyramid consistency) and by
   per-scale discriminators.

So the coarse latent codes do **not** drive the coarse outputs — the mapping the user wants
("latent scale 4/2/1 → output 1/4, 1/2, full") does not exist. Verified shapes: the latent
pyramid is downsampled **in-plane only** (the 24 Z-slices are the 2D stack/batch dim and are
never touched), and after `sum(quants)` the latent fed to `net_g` is a single `(24,4,24,24)`
cube. Goal: make **each output resolution be decoded from the cumulative latent up to its
matching scale**, so coarse codes genuinely reconstruct low-frequency structure and finer codes
add detail.

## Approach — Design A: cumulative-latent truncated shared decode

Decode each output head from the **cumulative partial sum** of latent scales up to that head's
level, running the *same* decoder weights truncated at the matching depth. All cumulative
latents are already `(24,4,24,24)` (each `quant_k_up` is upsampled to full 24² inside `encode()`
before summing), so no Z-anisotropy and no new resampling layers. Chosen over a
feature-reuse cascade (Design B) because A adds **zero new parameters / no optimizer or
checkpoint changes**, keeps `out0` byte-identical to today, and yields three clean, independent,
interpretable gradient paths. (B — injecting residual latents mid-decode — is faster per step
but needs new upsample/projection convs at 48³/96³ and entangles the heads; keep it as a
follow-up only if A's extra decode passes become a bottleneck.)

### Changes

**1. `networks/EncoderDecoder/ed023eMS.py` — new truncated multi-latent decode**
Add a `method='decode_ms'` branch to `Generator.forward` (`ed023eMS.py:196-223`) that takes three
already-`conv_in`'d latents and runs three truncated passes over the shared
`up3/conv5/up2/conv6/up1` weights, reusing the existing `conv7_64`/`conv7_128`/`conv7_k` heads:
```
lc, lm, lf = x['coarse'], x['mid'], x['full']          # each (1,256,24,24,24)
out64  = self.conv7_64 ( self.conv5(self.up3(lc)) )                                   # 48³
out128 = self.conv7_128( self.conv6(self.up2(self.conv5(self.up3(lm)))) )             # 96³
xf     = self.up1(self.conv6(self.up2(self.conv5(self.up3(lf)))))
out0, out1 = self.conv7_k(xf), self.conv7_g(xf)                                       # 192³
return {'out0': out0, 'out1': out1, 'out128': out128, 'out64': out64}
```
Leave the existing `'decode'`/`'encode'` paths untouched (still used by `vqcleanM0a.py` and
validation). **No new modules.**

**2. `models/vqcleanM0aMS.py` `encode()` (`:183-241`) — return cumulative sums**
`quants` (the per-scale `quant_k_up`, each `(24,4,24,24)`) is already built. Add
`cumulatives = list(itertools.accumulate(quants))` (last element == `quant`) and return it in the
currently-`None` 5th slot: `return quant, emb_loss, indices, h, cumulatives`. Thread it through
`forward()` (`:252-257`) via the unused 4th return slot: `return dec, diff, h, cumulatives, quant`.

**3. `models/vqcleanM0aMS.py` `__init__` — scale→output mapping**
Add `--ms_out_map` (default `'auto'`) in `add_model_specific_args` (`:153-181`). Auto default:
`self.ms_out_map = [num_scales-3, num_scales-2, num_scales-1]` → indices for `[out64, out128, out0]`.
- `num_scales=3` (6²/12²/24²) → `[0,1,2]`, clean 1:1.
- `num_scales=4` (current run, 3²/6²/12²/24²) → `[1,2,3]`; the extra coarsest scale folds into
  out64's cumulative — exactly the requested behavior.
Guard `num_scales >= 3` with graceful fallback to the old single-latent decode for 1/2.

**4. `models/vqcleanM0aMS.py` `generation()` (`:314-338`) — decode each cumulative at its depth**
Unpack `cumulatives`, select `[coarse, mid, full] = [cumulatives[i] for i in self.ms_out_map]`.
Apply the existing `downbranch`/`resizebranch` Z-transform + `decoder.conv_in` + permute
(`:319-330`, factor into a local helper) to **each** of the three → three `(1,256,24,24,24)` cubes.
Call `out = self.net_g({'coarse':lc,'mid':lm,'full':lf}, method='decode_ms')`. Keep the same
attribute names (`self.XupX / XupX128 / XupX64 / gif_scales`) so validation, GIF logging
(`base.py:353-355`), and metrics work unchanged. `full == quant`, so `out0` stays identical to today.

**5. `models/vqcleanM0aMS.py` `backward_g()` (`:355-417`) — retune pyramid consistency**
Keep the six-way GAN on `XupX` (`:359`), the L1 projection (`:361-365`), and the per-scale GANs
on `XupX128/XupX64` (`:396-400`) — they now score genuinely coarse-latent-driven outputs (the
point of the change). Only adjust the pyramid-consistency block (`:384-391`): default
`--pyr_detach` to **True** and lower `--lamb_pyr` default (1.0 → ~0.1), so it acts as a one-way
full→coarse stabilizer instead of dragging `out0` toward its own blur. Optional additive
`--lamb_pyr_proj` (default 0): reuse `get_projection` to give each coarse head an independent
projection-to-input anchor. No change to the VQ `aeloss`/`qloss` block or `backward_d`.

**6. Optimizer / checkpoint wiring — none.** `net_d_128/net_d_64` are already in `opt_disc` and
`netd_names`; `net_g` already in `opt_ae`. Design A adds no modules.

### Gradient-flow note (verified by planning pass)
`out0`→`full`(all scales), `out128`→`mid`(scales 0..k-1), `out64`→`coarse`(coarsest). Coarse
scales appear in every cumulative sum, so the coarse codebook gets strong multi-resolution
supervision — desired, not double-counting. The 2D reconstruction path (`self.reconstructions =
decode(quant)` via the separate ldm `Decoder`, driving `aeloss`/`qloss`) is untouched; `net_g`
only borrows `decoder.conv_in`. Watch loss balance (coarse GAN vs `qloss`/`aeloss`) since the
coarse `quant_convs`/`quantizers` now receive extra gradient.

### Risk
Two extra partial decode passes ≈ **+50–70% net_g decode activation/time** at batch 1. Likely
fine; if OOM, wrap net_g decode stages in `torch.utils.checkpoint`. Flag before scaling
batch/crop.

## Verification

No test suite exists; verify by training a short run and inspecting TensorBoard/MLflow:
```bash
CUDA_VISIBLE_DEVICES=2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr \
  --prj msout/vqcleanM0aMS/Scale3/latent2out --env brcb \
  --dataset E2507218fuse/E2507218cube/ --direction zcube --models vqcleanM0aMS \
  --netG ed023ems --num_scales 3 --lamb 5 --tracking_uri sqlite://///home/gary/workspace/logs/mlflow.db \
  --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005
```
Checks:
1. **Constructs & shapes:** `decode_ms` returns out64=48³, out128=96³, out0=192³ (add a one-off
   assert or a `python -c` smoke test of `Generator(...).forward(dict_of_cubes, 'decode_ms')`).
2. **Coarse-to-fine actually engaged:** in the MLflow GIF panels, the 1/4 and 1/2 heads should now
   show progressively coarser but *structurally faithful* volumes (low-freq first), not just
   blurred copies of out0. Confirm `axx64/axx128` and `pyr` loss keys log and move.
3. **Ablation sanity:** set `--ms_out_map` to feed `full` to all three (old behavior) and confirm
   metrics/GIF match the current model — isolates the change.
4. **Regression:** `val_lpips_pred` / `val_kid` should not degrade vs the current `--adv_ms 0`
   baseline once trained a few epochs.

## Critical files
- `networks/EncoderDecoder/ed023eMS.py` — `decode_ms` branch (only additive code)
- `models/vqcleanM0aMS.py` — `encode`/`forward`/`__init__`/`generation`/`backward_g`
- `models/base.py` — reference only (loss helpers, validation/GIF hooks; no edits)
- `run.sh` — add the new focused experiment line
