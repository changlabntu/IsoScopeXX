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
#   MScleanSup0      (MSclean + REAL X/Y projection supervision for the fused multi-view data, ported
#                     from vqcleanM0aSup0: --direction zcube_xcube_ycube --lamb_xy --aniso 8 --l1how_xy mean;
#                     no new params, MSclean ckpts load directly. See the fused-data section below)
#   MScleanSup0a     (Sup0 with the side loss as a fine-Z SPECTRUM match, --xy_mode spec (default) |
#                     hp | l1; voxel L1 to the views is minimised by blur — see the file header)
#   MScleanSup1      (fused data with the MEASURED forward model: Gaussian axial PSF --psf_sigma 12 in the
#                     main l1 (--l1how psf, samples the input's planes) and the side losses, + per-sample
#                     gain fit --side_gain and +-2 vox shift tolerance --side_shift. Box/max pooling is
#                     the wrong forward model for these views (PSF FWHM ~28 vox, not 8) — see file header)

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

# MSclean baseline. b=6
#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj MSclean/b6 --env b200 --dataset THX10SDM20xw/ --direction roiD/ --nm 11g --gamma 0.7 --gamma_lo -0.8 --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models MSclean --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1  --tracking_uri https://mlflow.ntugarylab.dpdns.org/ -b 6

# ExM
#CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj MSclean/b6 --env b200 --dataset iUExM/ --direction roiA/ --nm 11g --gamma 0.7 --gamma_lo -0.8 --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models MSclean --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1  --tracking_uri https://mlflow.ntugarylab.dpdns.org/ -b 4

# nanotube BEST — recovered from mlflow run 66ca7a2addc64ed1a3a4ac4a94778fbd (exp 16),
#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj MSclean/maxL1_b4Lb5 --env b200 --dataset nanotubeA/ --direction SA635/ --nm 11g --gamma 0.7 --gamma_lo -0.9 --cropsize 192 --cropz 48 --dsp 1 --lamb 5 --l1how max --skipl1 1 --models MSclean --num_scales 4 --lr 0.0005 --netG ed023emsfpn --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 -b 2 --resizebranch 0.5 --vq_restart --n_epochs 301 --epoch_save 25 --tracking_uri https://mlflow.ntugarylab.dpdns.org/ --n_embed_scales 128,128,256,512

#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj MSclean/maxL1_b4Lb10_skip4 --env b200 --dataset nanotubeA/ --direction SA635/ --nm 11g --gamma 0.7 --gamma_lo -0.9 --cropsize 192 --cropz 48 --dsp 1 --lamb 10 --l1how max --skipl1 4 --models MSclean --num_scales 4 --lr 0.0005 --netG ed023emsfpn --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 -b 2 --resizebranch 0.5 --vq_restart --n_epochs 301 --epoch_save 25 --tracking_uri https://mlflow.ntugarylab.dpdns.org/ --n_embed_scales 128,128,256,512


# iUExM
#CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj MSclean/b6 --env b200 --dataset iUExM/ --direction roiA/ --nm 11g --gamma 0.7 --gamma_lo -0.8 --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models MSclean --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1  --tracking_uri https://mlflow.ntugarylab.dpdns.org/ -b 4

# filopodia
#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj MSclean/b4Lb10 --env b200 --dataset filopodia/ --direction SA635/ --nm 11g --gamma 0.3 --gamma_lo -0.9 --cropsize 192 --cropz 48 --dsp 1 --lamb 10 --models MSclean --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1  --tracking_uri https://mlflow.ntugarylab.dpdns.org/ -b 2 --downbranch 2 --vq_normalize --vq_restart --tracking_uri https://mlflow.ntugarylab.dpdns.org/

# TP0727
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj MSclean/b4Lb10 --env brcb --dataset TP0727/ --direction TP0727/ --nm 11g --gamma 0.4 --gamma_lo -0.9 --cropsize 192 --cropz 48 --dsp 1 --lamb 10 --models MSclean --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 -b 1 --downbranch 2 --vq_normalize --vq_restart --tracking_uri MS0728 --n_epochs 301

#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj MSclean/b4Lb10 --env brcb --dataset TP0727/ --direction TP0727/ --nm 11g --gamma 0.7 --gamma_lo -0.9 --cropsize 192 --cropz 48 --dsp 1 --lamb 10 --models MSclean --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 -b 1 --downbranch 2 --vq_normalize --vq_restart --tracking_uri MS0728 --n_epochs 301

# skipU baseline
#python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipU/roiD192gfC/max5skip4 --env b200 --dataset THX10SDM20xw/ --direction roiD/ --nm 11g --gamma 0.7 --gamma_lo -0.8 --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models MSclean --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri https://mlflow.ntugarylab.dpdns.org/ -b 4

# skipU baseline
#python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipU/roiD192gfC/max5skip4 --env b200 --dataset THX10SDM20xw/ --direction roiD/ --nm 11g --gamma 0.7 --gamma_lo -0.8 --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models MSclean --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri https://mlflow.ntugarylab.dpdns.org/ -b 4

# QD baseline
#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/MScleanQD/roiD192gfC/max5skip4 --env b200 --dataset thx10/ --direction roiD/ --nm 11g --gamma 0.7 --gamma_lo -0.8 --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models MScleanQD --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri https://mlflow.ntugarylab.dpdns.org/ -b 3 --vq_normalize --vq_restart --n_embed_scales 128,128,256,512

# QD 
CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/MScleanQD/roiD192gfC/max5skip4 --env b200 --dataset symmetricbead/ --direction CamA/ --nm 11g --gamma 1.0 --gamma_lo -1.0 --cropsize 192 --cropz 48 --dsp 2 --lamb 1 --models MScleanQD --num_scales 4 --lr 0.0002 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri https://mlflow.ntugarylab.dpdns.org/ -b 2 --vq_normalize --vq_restart --n_embed_scales 128,128,256,512 --n_epochs 501 --l1how dsp --ema_decay 0

#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqclean/roiD192gfC/max5skip4 \
#      --env b200 --dataset thx10/ --direction roiD/ \
#      --nm 11g --gamma 0.7 --gamma_lo -0.8 \
#      --cropsize 192 --cropz 24 --dsp 1 --lamb 5 \
#      --models vqclean --lr 0.0005 \
#      --netG ed023e --netD patch_16 \
#      --tracking_uri https://mlflow.ntugarylab.dpdns.org/ -b 4


# ---------------------------------------------------------------------------
# MScleanSup0 on the fused multi-view data (real X/Y projection supervision).
#
# Geometry is NOT free here: backward_g asserts XupX and the xcube/ycube views share
# the same grid, so cropz == cropsize (isotropic cube) and dsp == aniso == 8
# (net_g upsamples 8x in every axis: 8 * cropz/dsp == cropz only when dsp == 8).
# Keep --downbranch 1 — downbranch 2 halves the output Z and trips the assert.
# uprate = 8, divisible by 4 as --lamb_coarse requires. cropsize 192 % aniso 8 == 0.
#
# --nm 11g, not the 11p of the old Sup0 recipe: E2507218fuse/ has no norm_stats.json
# (the one under E2507218cube/ is at the wrong level and that train/ is now empty), and
# 11p measured as a near no-op on these cubes anyway — topatch.py already min-max'd each
# view at patch creation, so the per-view affine anchors are pre-aligned. Gamma is
# monotonic and applied identically to all three views, and MIP commutes with a monotone
# map, so the default --l1how_xy max stays consistent under 11g.
#
# --lamb_xy 1 (below the model default 2) on purpose: measured on this data, only ~5% of
# l1_x/l1_y is genuine HR-Z signal — the rest is a floor from 1-4 voxel per-file
# misregistration and per-view PSF/intensity differences, so l1_x/l1_y will plateau high.
# Run the lamb_xy 0 control before believing any improvement is from the supervision.
# Pick free GPUs — the TP0727 lines above use 2..7.

# control: identical crops/geometry, supervision off (should reproduce plain MSclean)
#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/MScleanSup0/zxy192gf/lambxy0 --env brcb --dataset E2507218fuse/ --direction zcube_xcube_ycube --nm 11g --gamma 0.5 --gamma_lo -0.9 --cropsize 192 --cropz 192 --dsp 8 --lamb 5 --models MScleanSup0 --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --lamb_xy 0 --aniso 8 --l1how_xy max -b 1 --vq_normalize --vq_restart --tracking_uri MS0728 --n_epochs 301

# stronger supervision, only if lamb_xy 1 shows the loss actually moving off its floor
#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/MScleanSup0/zxy192gf/lambxy3 --env brcb --dataset E2507218fuse/ --direction zcube_xcube_ycube --nm 11g --gamma 0.5 --gamma_lo -0.9 --cropsize 192 --cropz 192 --dsp 8 --lamb 5 --models MScleanSup0 --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --lamb_xy 3 --aniso 8 --l1how_xy max -b 1 --vq_normalize --vq_restart --tracking_uri MS0728 --n_epochs 301

# MScleanSup1: PSF forward model (sigma 12) for main l1 + side losses, gain fit, +-2 shift. Controls: same with --lamb_xy 0
# (PSF main l1 only) and the Sup0 lambxy10 run 5a1258ce (box model). Smoke first: --n_epochs 2.
#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/MScleanSup1/zxy192gf/psf12_lambxy5 --env b200 --dataset E2507218zxy/ --direction zcube_xcube_ycube --nm 11g --gamma 0.5 --gamma_lo -0.9 --cropsize 192 --cropz 192 --dsp 8 --lamb 5 --models MScleanSup1 --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --l1how psf --psf_sigma 12 --lamb_xy 5 --side_gain 1 --side_shift 2 --aniso 8 -b 1 --vq_normalize --vq_restart --tracking_uri https://mlflow.ntugarylab.dpdns.org/ --n_epochs 501
#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/MScleanSup1/zxy192gf/psf12_lambxy0 --env b200 --dataset E2507218zxy/ --direction zcube_xcube_ycube --nm 11g --gamma 0.5 --gamma_lo -0.9 --cropsize 192 --cropz 192 --dsp 8 --lamb 5 --models MScleanSup1 --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --l1how psf --psf_sigma 12 --lamb_xy 0 --aniso 8 -b 1 --vq_normalize --vq_restart --tracking_uri https://mlflow.ntugarylab.dpdns.org/ --n_epochs 501

# MScleanSup0a: side loss = fine-Z spectrum match (lamb_xy re-tuned for log-power units)
#CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/MScleanSup0a/zxy192gf/spec1 --env b200 --dataset E2507218zxy/ --direction zcube_xcube_ycube --nm 11g --gamma 0.5 --gamma_lo -0.9 --cropsize 192 --cropz 192 --dsp 8 --lamb 5 --models MScleanSup0a --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --lamb_xy 1 --aniso 8 --xy_mode spec -b 1 --vq_normalize --vq_restart --tracking_uri https://mlflow.ntugarylab.dpdns.org/ --n_epochs 501

# MScleanSup0 fused-data baseline
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/MScleanSup0/zxy192gf/lambxy10 --env brcb --dataset E2507218fuse/ --direction zcube_xcube_ycube --nm 11g --gamma 0.5 --gamma_lo -0.9 --cropsize 192 --cropz 192 --dsp 8 --lamb 5 --models MScleanSup0 --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patch_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --lamb_xy 10 --aniso 8 --l1how_xy mean -b 1 --vq_normalize --vq_restart --tracking_uri MS0728 --n_epochs 501


# ---------------------------------------------------------------------------
# HISTORICAL — the original vqcleanM0aSup0 launches, recovered verbatim from git
# (run 1: 08f4500, 2026-06-28; run 2 below: 76c8cd0, 2026-06-29, which also added the
# held-out val_l1_x/val_l1_y metrics). Both logged to sqlite:////.../logs/mlflow.db,
# which was trashed 2026-07-03 and is now empty (0 rows) — no metrics, checkpoints or
# artifacts of these two runs survive anywhere, local or remote.
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuseSup0/vqcleanM0aSup0/Scale4/max1_10_skip4_run2 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube_xcube_ycube --lamb 1 --lamb_xy 5 --aniso 8 --nm 11p --models vqcleanM0aSup0 --num_scales 4 --tracking_uri sqlite://///home/gary/workspace/logs/mlflow.db --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005
#
# Three things break if this is re-run verbatim:
#   1. --dataset E2507218fuse/E2507218cube/ — that train/ is now EMPTY; the cubes moved
#      up to E2507218fuse/train/{zcube,xcube,ycube}, so use --dataset E2507218fuse/.
#   2. --tracking_uri sqlite://///... predates 5679913 (bare NAME -> $LOGS/NAME.db), so
#      today it would be taken as a NAME and make a garbage filename. Use MS0728.
#   3. --nm 11p needs $DATASET/E2507218fuse/norm_stats.json, which is absent (the one
#      under E2507218cube/ is a level too deep). Measured a near no-op on these cubes
#      anyway — topatch.py already min-max'd each view — so prefer the 11g line above.
# Also note vqcleanM0aSup0's --l1how_xy still defaults to 'max'; only MScleanSup0 was
# switched to 'mean'. See that model's header for why max biases toward blurred X/Y.

# QD CUT smoke (MScleanQD2 + PatchNCE, q=generated: NCE gradient flows into net_g; --nocut recovers QD2)
CUDA_VISIBLE_DEVICES=0,1,2,3 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj scratch/QDCUTsmoke --env b200 --dataset symmetricbead/ --direction CamA/ --nm 11g --gamma 1.0 --gamma_lo -1.0 --cropsize 192 --cropz 48 --dsp 2 --models MScleanQDCUT --num_scales 4 --netG ed023emsfpnu --netD patch_16 --l1how dsp --lamb 1 --pyr_detach --adv_ms 0.5 --n_embed_scales 128,128,256,512 --vq_restart -b 2 --lr 0.0002 --n_epochs 2 --tracking_uri https://mlflow.ntugarylab.dpdns.org/
