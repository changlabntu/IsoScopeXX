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
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqclean/roiD192/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqclean --lr 0.0002 --netG ed023e --netD patch_16 --tracking_uri thx-MS-384


# --- clean-data controls (fixed precrop, thx-MS-384 store; audit doc/audit_precrop_2026-07-09.md) ---
# Fresh roiD192 prj paths — never reuse the blurred thx-MS prj dirs. Compare vs the vqclean
# roiD192 benchmark above at matched optimizer steps.

# thx MS baseline on fixed roiD: exact config of the original thx MS run (adv_ms 0, lr 5e-4)
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMS/roiD192/max5skip4adv0 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqcleanM0aMS --num_scales 4 --lr 0.0005 --netG ed023ems --adv_ms 0 --tracking_uri thx-MS-384

# thx MSskipE on fixed roiD: THE decisive control — exact declared-baseline config (skip trunk +
# EMA-at-eval + coarse L1, lr 2e-3). If val_spec recovers toward the vqclean benchmark, the MS
# recipe was never the problem; if not, the recipe blame regains support.
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipE/roiD192/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri thx-MS-384

# --- gamma-remapped intensity (nm 11g: [-1,1] input, gamma 0.25; no sidecar needed) ---
# roiD is extremely left-piled (median -0.9, deep in tanh saturation); 11g recenters the bulk
# to ~0 (skew +4.7 -> +0.2). Input must be pre-normalized to [-1,1]. NEW comparability boundary:
# metric scales shift, so these runs get fresh roiD192g prjs + the thx-MS-384g store — never
# score them against nm=00 runs.

# skipE on remapped data — new baseline/comparator (lr 5e-4 kept from the running clean skipE)
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipE/roiD192g/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --nm 11g --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.0005 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri thx-MS-384g

# skipUB on remapped data — anti-alias stack (resize-conv netG + BlurPool netD), only deltas vs the line above
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipUB/roiD192g/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --nm 11g --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.0005 --netG ed023emsfpnu --netD patchblur_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri thx-MS-384g

# remote tracking: --tracking_uri https://mlflow.ntugarylab.dpdns.org/



