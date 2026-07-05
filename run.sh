# fuse MS
#CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/vqcleanM0aMSadv0/Scale4/max5skip4 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --lamb 5 --models vqcleanM0aMS --num_scales 4 --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005 --netG ed023ems --adv_ms 0 --tracking_uri MS


# fuse MS
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSadv0/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMS --num_scales 4 --cropsize 192 --cropz 24 --lr 0.0005 --netG ed023ems --adv_ms 0 --tracking_uri thx-MS

# fuse MS + FPN scale-to-depth injection (out64<-scales 0-1, out128<-0-2, out0<-all)

#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSadv0/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSfpn --num_scales 4 --cropsize 192 --cropz 24 --lr 0.0005 --netG ed023emsfpn --adv_ms 0 --tracking_uri thx-MS

# thx MSskip: full-sum trunk + zero-init lateral skips (step-0 == MS baseline) + pyr_detach + coarse GANs (MS_refine 1+2)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskip/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskip --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --tracking_uri thx-MS

# thx MSskipE: MSskip + EMA-at-eval + real-data coarse L1 (MS_refine2 F1+F2)
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj thx10/vqcleanM0aMSskipE/Scale4/max5skip4 --env brcb --dataset THX10SDM20xw/ --direction roiD/ --lamb 5 --models vqcleanM0aMSskipE --num_scales 4 --cropsize 192 --cropz 24 --lr 0.002 --netG ed023emsfpn --pyr_detach --adv_ms 0.5 --lamb_coarse 1 --tracking_uri thx-MS







