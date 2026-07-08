# fuse MS
#CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/vqcleanM0aMSadv0/Scale4/max5skip4 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --lamb 5 --models vqcleanM0aMS --num_scales 4 --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005 --netG ed023ems --adv_ms 0 --tracking_uri MS


# fuse MS
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSadv0/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMS --num_scales 4 --cropsize 192 --cropz 24 --lr 0.0005 --netG ed023ems --adv_ms 0 --tracking_uri thx-MS

# fuse MS + FPN scale-to-depth injection (out64<-scales 0-1, out128<-0-2, out0<-all)

#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSadv0/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSfpn --num_scales 4 --cropsize 192 --cropz 24 --lr 0.0005 --netG ed023emsfpn --adv_ms 0 --tracking_uri thx-MS

# thx MSskip: full-sum trunk + zero-init lateral skips (step-0 == MS baseline) + pyr_detach + coarse GANs (doc/experiments_MS.md)
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskip/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskip --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --tracking_uri thx-MS

# thx MSskipE: MSskip + EMA-at-eval + real-data coarse L1 (doc/experiments_MS.md)
# result @141ep: best-in-class val_kid 4.96 (EMA win), but val_lpips_pred plateaued 0.616 vs MSskip 0.578
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipE/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri thx-MS

# thx MSskipP band: NEGATIVE @139ep — checkerboard fills the freed high-Z band (D aliases it away);
# val_kid 6.95 / val_lpips 0.625, worst of skip family. Retest band only after the Z3 anti-checkerboard netG.
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipP/Scale4/band5 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskipP --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --l1how band --tracking_uri thx-MS

# thx MSskipP lse: NEGATIVE @132ep — LPIPS flat at 0.636, KID stalled 5.75 (vs skipE 0.615/4.95 same age).
# Projection line closed: 'max' stays. (band also negative — see above.)
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipP/Scale4/lse5 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskipP --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --l1how lse --l1phase random --tracking_uri thx-MS

# thx MSskipE-ema999: A1 ablation — skipE with EMA horizon ~6.5 epochs instead of ~65 (only --ema_decay changes).
# Hypothesis: recovers MSskip's LPIPS (~0.578) while keeping skipE's KID (~5.0) -> first run holding both crowns.
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipE/Scale4/ema999 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --ema_decay 0.999 --tracking_uri thx-MS

# thx MSskipE: MSskip + EMA-at-eval + real-data coarse L1 (doc/experiments_MS.md)
# result @141ep: best-in-class val_kid 4.96 (EMA win), but val_lpips_pred plateaued 0.616 vs MSskip 0.578
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipEcoarse0/Scale4/max5skip4coarse0 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --cropsize 192 --cropz 24 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 0 --tracking_uri https://mlflow.ntugarylab.dpdns.org/ #thx-MS


#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipE/Scale4/max5skip4coarse0 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --cropsize 192 --cropz 192 --dsp 8 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 0 --tracking_uri fuse-MS

# fuse MSskipU: skipE + Z3 resize-conv trunk (--netG ed023emsfpnu) — removes the ConvTranspose alias-lattice
# SOURCE (diagonal beading, doc/research_artifact_suggestions.md). Fresh trunk weights: no step-0 equivalence,
# compare vs the fuse skipE line above at matched optimizer steps. Verdict: val_lat_* (expect p2diag/p4diag -> ~1) + LPIPS/KID.
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipU/Scale4/max5skip4coarse0 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --cropsize 192 --cropz 192 --dsp 8 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.002 --netG ed023emsfpnu --pyr_detach --adv_ms 0.5 --lamb_coarse 0 --tracking_uri fuse-MS

# thx MSskipUB: the full anti-alias stack on thx10 — skipE + Z3 resize-conv trunk (--netG ed023emsfpnu, removes
# the ConvTranspose lattice SOURCE; fuse verdict: p2diag 94x -> 1.3) + BlurPool patch D (--netD patchblur_16,
# gives net_d/net_d_128/net_d_64 sight above their stride-2 Nyquist so they can DEMAND the high frequency
# resize-conv under-produces — the fuse-U KID regression, 5.4 vs 3.1). Deliberate knob choices:
#   --lamb_coarse 0  : drops the coarse-L1 smoothing pressure (sharpness-aligned; matches fuse U/UB config)
#   ema_decay 0.9999 : default kept for comparability with the thx skipE family (dual-eval still the follow-up)
#   l1how max / lamb 5 / skipl1 4 : from cfg/aisr.yaml — held constant across the whole lineage; band retest
#   is gated on this run's verdict, not folded in.
# Compare vs thx skipE/skipEcoarse0 at matched optimizer steps (the clean UB-vs-U BlurPool isolation lives on
# fuse). Success = val_lat_* stays ~1 AND val_kid recovers to <=skipE (4.97) AND val_lpips_pred <= MSskip (0.577).

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipUB/Scale4/max5skip4coarse0 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --cropsize 192 --cropz 24 --dsp 1 --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --lr 0.002 --netG ed023emsfpnu --netD patchblur_16 --pyr_detach --adv_ms 0.5 --lamb_coarse 0 --tracking_uri thx-MS


# remote tracking: --tracking_uri https://mlflow.ntugarylab.dpdns.org/



