# fuse MS
#CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/vqcleanM0aMSadv0/Scale4/max5skip4 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --lamb 5 --models vqcleanM0aMS --num_scales 4 --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005 --netG ed023ems --adv_ms 0 --tracking_uri MS


# fuse MS
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSadv0/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMS --num_scales 4 --cropsize 192 --cropz 24 --lr 0.0005 --netG ed023ems --adv_ms 0 --tracking_uri thx-MS

# fuse MS + FPN scale-to-depth injection (out64<-scales 0-1, out128<-0-2, out0<-all)

#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSadv0/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSfpn --num_scales 4 --cropsize 192 --cropz 24 --lr 0.0005 --netG ed023emsfpn --adv_ms 0 --tracking_uri thx-MS

# thx MSskip: full-sum trunk + zero-init lateral skips (step-0 == MS baseline) + pyr_detach + coarse GANs (doc/experiments_MS.md)
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskip/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskip --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --tracking_uri thx-MS

# thx MSskipE: MSskip + EMA-at-eval + real-data coarse L1 (doc/experiments_MS.md)
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipE/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri thx-MS

# thx MSskipP band: skipE + band projection (--l1how band)
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipP/Scale4/band5 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskipP --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --l1how band --tracking_uri thx-MS

# thx MSskipP lse: skipE + soft-max projection (--l1how lse)
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipP/Scale4/lse5 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskipP --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --l1how lse --l1phase random --tracking_uri thx-MS

# thx MSskipE-ema999: A1 ablation — skipE with EMA horizon ~6.5 epochs instead of ~65 (only --ema_decay changes).
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipE/Scale4/ema999 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --ema_decay 0.999 --tracking_uri thx-MS

# thx MSskipE coarse0: skipE with --lamb_coarse 0
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipEcoarse0/Scale4/max5skip4coarse0 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --cropsize 192 --cropz 24 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 0 --tracking_uri https://mlflow.ntugarylab.dpdns.org/ #thx-MS


#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipE/Scale4/max5skip4coarse0 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --cropsize 192 --cropz 192 --dsp 8 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 0 --tracking_uri fuse-MS

# fuse MSskipU: skipE + Z3 resize-conv trunk (--netG ed023emsfpnu) — removes the ConvTranspose alias-lattice
# SOURCE (diagonal beading, doc/research_artifact_suggestions.md). Fresh trunk weights: no step-0 equivalence,
# compare vs the fuse skipE line above at matched optimizer steps. Verdict: val_lat_* (expect p2diag/p4diag -> ~1) + LPIPS/KID.

#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipU/Scale4/max5skip4coarse0 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --cropsize 192 --cropz 192 --dsp 8 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.002 --netG ed023emsfpnu --pyr_detach --adv_ms 0.5 --lamb_coarse 0 --tracking_uri fuse-MS

#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipUB/Scale4/max5skip4coarse0 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.002 --netG ed023emsfpnu --netD patchblur_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 0 --tracking_uri thx-MS

# --models options (vqclean line -> MS line):
#   vqclean          (single VQ autoencoder + 3D net_g; the texture-reference recipe: lr 2e-4, no EMA,
#                     six-way adv + max-projection L1; --netG ed023e)
#   vqcleanM0a       (vqclean + VAR-style multi-scale residual VQ latent: --num_scales coarse-to-fine
#                     passes, --shared_codebook; single full-res output)
#   vqcleanM0aSup0   (M0a + REAL projection supervision from multi-view fuse data: XupX mean-pooled
#                     along X/Y is L1-matched to xcube/ycube; --direction zcube_xcube_ycube --lamb_xy --aniso 8)
#   vqcleanM0aMS     (M0a + multi-scale progressive OUTPUTS with per-scale discriminators;
#                     --netG ed023ems, --adv_ms weights the 1/2- and 1/4-scale adv losses)
#   vqcleanM0aMSfpn  (MS + FPN scale-to-depth injection, out64<-scales 0-1 etc.; --netG ed023emsfpn.
#                     NEGATIVE — starves the full-res head; verdict survives the precrop audit)
#   vqcleanM0aMSskip (MS + full-sum trunk with zero-init lateral skips, step-0 == MS baseline;
#                     + --pyr_detach + coarse GANs at lr 2e-3)
#   vqcleanM0aMSskipE(MSskip + EMA-at-eval (--ema_decay) + real-data coarse L1 (--lamb_coarse);
#                     the declared MS baseline config. skipU/UB are NOT separate models — this model
#                     with --netG ed023emsfpnu (resize-conv trunk, kills the alias lattice) and
#                     --netD patchblur_16 (BlurPool D))
#   vqcleanM0aMSskipP(skipE + alternative Z projections: --l1how band | lse instead of max)
#   vqcleanMH        (vqclean recipe trunk + multi-scale VQ latent + READ-ONLY coarse heads:
#                     --netG ed023emsdet detaches the head taps, heads train by distillation only,
#                     trunk gradients identical to headless vqclean by construction; lr 2e-4)

# thx vqcleanMH
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanMH/Scale1/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqcleanMH --num_scales 1 --lr 0.0002 --netG ed023emsdet --netD patch_16 --tracking_uri thx-MS-384

# thx vqclean benchmark
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqclean/roiD192/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqclean --lr 0.0002 --netG ed023e --netD patch_16 --tracking_uri thx-MS-384


# --- clean-data controls (fixed precrop, thx-MS-384 store; precrop audit 2026-07-09, doc retired) ---
# Fresh roiD192 prj paths — never reuse the blurred thx-MS prj dirs. Compare vs the vqclean
# roiD192 benchmark above at matched optimizer steps.

# thx MS baseline on fixed roiD: exact config of the original thx MS run (adv_ms 0, lr 5e-4)
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMS/roiD192/max5skip4adv0 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqcleanM0aMS --num_scales 4 --lr 0.0005 --netG ed023ems --adv_ms 0 --tracking_uri thx-MS-384

# thx MSskipE on fixed roiD: THE decisive control — exact declared-baseline config (skip trunk +
# EMA-at-eval + coarse L1, lr 2e-3). If val_spec recovers toward the vqclean benchmark, the MS
# recipe was never the problem; if not, the recipe blame regains support.
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipE/roiD192/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri thx-MS-384

# --- foreground-gamma intensity (nm 11g + floor: --gamma_lo clips the noise band to -1,
# --gamma expands the faint foreground) ---
# Lesson from the roiD192g gamma-0.25 run (mlflow 0606f574, ep84): the dark bulk of roiD is
# acquisition noise, not texture — plain gamma amplified it into the dominant image content and
# the GAN regressed below its do-nothing baseline. Floor at the noise ceiling (-0.80 by visual sweep; med+4MAD -0.70 clipped real dim-volume structure)
# instead: background -> flat -1 (~90% of voxels), foreground gets the range (median -0.40).
# Input must be pre-normalized to [-1,1]. THIRD metric-scale regime: fresh roiD192gf prjs,
# thx-MS-384gf store — never score against nm=00 or gamma-0.25 runs.

# skipE foreground-gamma baseline
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipE/roiD192gf/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --nm 11g --gamma 0.5 --gamma_lo -0.8 --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.0005 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri thx-MS-384gf

# skipE foreground-gamma on fused zcube: same recipe, fuse geometry (cropz 192 dsp 8);
# gamma_lo -0.9 = fuse-tuned floor (med+4MAD ~ -0.93; roiD's -0.8 would clip dim structure).
# Fresh gf prj + fuse-MS-gf store — new metric scale, never score against fuse-MS runs.
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/vqcleanM0aMSskipE/zcube192gf/max5skip4 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --nm 11g --gamma 0.5 --gamma_lo -0.9 --cropsize 192 --cropz 192 --dsp 8 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.0005 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri fuse-MS-gf

# skipUB foreground-gamma — anti-alias stack (resize-conv netG + BlurPool netD), only deltas vs the line above
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipUB/roiD192gf/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --nm 11g --gamma 0.5 --gamma_lo -0.8 --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patchblur_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri thx-MS-384gf

# MSclean baseline. b=4
CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj MSclean/b6 --env b200 --dataset THX10SDM20xw/ --direction roiD/ --nm 11g --gamma 0.7 --gamma_lo -0.8 --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models MSclean --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1  --tracking_uri https://mlflow.ntugarylab.dpdns.org/ -b 6

# remote tracking: --tracking_uri https://mlflow.ntugarylab.dpdns.org/




