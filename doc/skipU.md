# skipU(B): model innovations relative to vqclean

Status as of **2026-07-14**. Summarizes what the skipU/skipUB configuration adds on top of the
`vqclean` baseline, layer by layer. Head-to-head numbers come from the clean (post-precrop-fix)
roiD192gf campaign — see `doc/experiments_MS_skipE_skipU_gamma_2026-07.md` for the full ledger.

## Naming

skipU is **not a separate model class**. It is `vqcleanM0aMSskipE` run with
`--netG ed023emsfpnu`; skipUB additionally swaps `--netD patch_16 → patchblur_16`. The MLflow
`models` tag therefore still reads `vqcleanM0aMSskipE` — filter runs by `prj`, not the tag.

| name | model class | netG | netD |
|---|---|---|---|
| vqclean | `vqclean` | `ed023e` | `patch_16` |
| skipE | `vqcleanM0aMSskipE` | `ed023emsfpn` | `patch_16` |
| skipU | `vqcleanM0aMSskipE` | `ed023emsfpnu` | `patch_16` |
| skipUB | `vqcleanM0aMSskipE` | `ed023emsfpnu` | `patchblur_16` |

## Baseline: vqclean

2D VQ encoder → a single `VectorQuantizer2` at one scale → latent into a 3D ConvTranspose
generator, trained with the shared self-supervised recipe (six-way PatchGAN + L1 projection).
One quantization pass, one output resolution, one discriminator scale.

## The four innovation layers

1. **Multi-scale residual VQ** (`vqcleanM0a`). Instead of one quantization pass, VAR-style
   coarse-to-fine residual quantization: `--num_scales` passes, each quantizing what the
   previous scales failed to represent and subtracting its upsampled contribution
   (`--shared_codebook` toggles one codebook vs. per-scale). The latent becomes a pyramid of
   codes rather than one grid.

2. **Multi-scale progressive outputs** (`vqcleanM0aMS`). The 3D generator emits coarse 1/4- and
   1/2-scale volumes (out64/out128) alongside the full-resolution output, each judged by its
   own discriminator. `--adv_ms` weights the coarse adversarial terms (`--adv_ms 0` skips the
   coarse discriminators); `--pyr_detach` stops coarse gradients from re-entering the trunk.

3. **Skip trunk + coarse real supervision** (`vqcleanM0aMSskip` → `skipE`).
   - **Full-sum trunk**: every VQ scale gets full decoder depth.
   - **Zero-init lateral skips** (ControlNet-style): the two finest VQ scales are re-injected
     *after* the out64/out128 taps, so at step 0 the model is numerically identical to
     `vqcleanM0aMS` and can only learn skips that help; the coarse heads see only coarse codes
     (fine detail cannot leak down the pyramid).
   - **EMA-at-eval**: validation and epoch checkpoints use EMA weights.
   - **Real-data coarse supervision** (`--lamb_coarse`): out128/out64 are Z-projected (same
     `--skipl1`/`--l1how` as the main L1, at halved/quartered uprate) and L1-matched to the
     real input — the coarse pyramid is anchored to data, not just the GAN.

4. **Anti-aliasing pair — the "U" and "B"** (skipU / skipUB).
   - **skipU (generator)**: every `ConvTranspose3d(k4, s2)` up-stage is replaced with
     nearest-neighbor `Upsample((2,2,2))` + `Conv3d(k3, s1)` ("resize-conv"), removing the
     kernel-overlap mechanism that produces checkerboard artifacts. Switched purely by netG
     name via `UPSAMPLE_GENERATORS` in `networks/registry.py` (same class as `ed023emsfpn`).
   - **skipUB (discriminator)**: BlurPool anti-aliased downsampling (Zhang 2019), so the D
     cannot reward the generator for matching aliasing phase patterns.

## Why it matters (roiD192gf head-to-head, skipE vs skipUB)

- The resize-conv + BlurPool pair is what killed the diagonal lattice artifact:
  `val_lat_p2diag` 7.91 (skipE) → **1.11** (skipUB, clean floor).
- skipE's beaded/dotted blob boundaries (checkerboard riding on edges) become smooth.
- Costs: 1.7× slower epochs, +25% GPU memory, a residual period-2 Z stripe
  (`val_lat_p2z`), and a drifting `val_kid` that makes late-checkpoint selection riskier.
