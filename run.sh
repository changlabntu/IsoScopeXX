# default 8X enhancement
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --nocut
# default 4X enhancement
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --nocut --downbranch 2 --cropz 32 # (128 // 4)
# 8X enhancement with contrastive loss lbNCE=1
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --lbNCE 1


# filopodia on GHCL01
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml filopodia_GHCL01 --env GHCL01 --nocut --downbranch 2 --cropz 32

# filopodia on GHCL03
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml filopodia_GHCL03 --prj default/max10skip4 --env GHCL03 --nocut --downbranch 2 --cropz 32

# filopodia on NTHU_Gary3
#NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml filopodia_NTHU_Gary3 --env NTHU_Gary3 --nocut --downbranch 2 --cropz 32

#export LOGS=/home/gary/workspace/logs/THX10SDM20xw
# multi scale vq
#CUDA_VISIBLE_DEVICES=4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj vqcleanM0a/S4/max5skip4 --env brcb --dataset THX10SDM20xw --direction roiAdsp4 --lamb 5 --models vqcleanM0a --num_scales 4 --tracking_uri sqlite://///home/gary/workspace/logs/THX10SDM20xw/mlflow.db

# VAE
#CUDA_VISIBLE_DEVICES=4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj vqcleanVAE/max5skip4 --env brcb --dataset THX10SDM20xw --direction roiAdsp4 --lamb 5 --models vqclean --tracking_uri sqlite://///home/gary/workspace/logs/THX10SDM20xw/mlflow.db --vae --ldmyaml ldmaex2

# VQ
#CUDA_VISIBLE_DEVICES=4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj vqcleanVQ/max5skip4 --env brcb --dataset THX10SDM20xw --direction roiAdsp4 --lamb 5 --models vqclean --tracking_uri sqlite://///home/gary/workspace/logs/THX10SDM20xw/mlflow.db  --ldmyaml vqgan

# VQ -256 
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj vqcleanVQ/192/max5skip4 --env brcb --dataset THX10SDM20xw --direction roiAdsp4 --lamb 5 --models vqclean --tracking_uri sqlite://///home/gary/workspace/logs/THX10SDM20xw/mlflow.db  --ldmyaml vqgan --cropz 24 --cropsize 192 --precrop 256 #--epoch_save 20

# flow
#CUDA_VISIBLE_DEVICES=4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj flowV1/max5skip4 --env brcb --dataset THX10SDM20xw --direction roiAdsp4 --lamb 5 --models flowV1 --tracking_uri sqlite://///home/gary/workspace/logs/THX10SDM20xw/mlflow.db  --ldmyaml vqgan --flow_steps_train 4 --flow_steps 20 --flow_lambda 1.0

# VQ
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj vqcleanVQ/vqganB/max3skip4 --env brcb --dataset HumanGolgi/E2507218/ --direction E2507218z --lamb 3 --models vqclean --tracking_uri sqlite://///home/gary/workspace/logs/mlflow.db  --ldmyaml vqganB --ngf 64

# flow
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj flowV1s20/max3skip4ngf64 --env brcb --dataset HumanGolgi/E2507218/ --direction E2507218z --lamb 3 --models flowV1 --tracking_uri sqlite://///home/gary/workspace/logs/mlflow.db --ldmyaml vqganB --ngf 64 --flow_steps_train 30 --flow_steps 20 --flow_lambda 1.0

# VQ 
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj vqcleanVQ/vqganB/max10skip4 --env brcb --dataset E2507218fuse --direction roiA --lamb 10 --models vqclean --tracking_uri sqlite://///home/gary/workspace/logs/mlflow.db  --ldmyaml vqgan --ngf 32 --downbranch 8

# M0a
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj vqcleanM0a/Scale4/max5skip4 --env brcb --dataset E2507218fuse --direction E2507218z8x --lamb 5 --models vqcleanM0a --num_scales 4 --tracking_uri sqlite://///home/gary/workspace/logs/mlflow.db --cropsize 192 --cropz 24

# fuse
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/vqcleanM0a/Scale4/max5skip4 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --lamb 5 --models vqcleanM0a --num_scales 4 --tracking_uri sqlite://///home/gary/workspace/logs/mlflow.db --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005


# fuse - supervised 0 (views loaded + multi-panel GIF only, no X/Y loss)
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuseSup/vqcleanM0a/Scale4/max5skip4 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/_xcube/_ycube/ --lamb 5 --models vqcleanM0a --num_scales 4 --tracking_uri sqlite://///home/gary/workspace/logs/mlflow.db --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005 --nm 11p

# fuse - supervised 0 (vqcleanM0aSup0: xcube/ycube projection supervision, --lamb_xy)
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuseSup0/vqcleanM0aSup0/Scale4/max3_max0_skip4_run2 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube_xcube_ycube --lamb 3 --lamb_xy 0 --aniso 8 --nm 11p --models vqcleanM0aSup0 --num_scales 4 --tracking_uri sqlite://///home/gary/workspace/logs/mlflow.db --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005 --l1how_xy max

# multi-scale progressive output (vqcleanM0aMS + ed023ems): full/1/2/1/4 heads; pyramid consistency (--lamb_pyr) + per-scale discriminators
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj msout/vqcleanM0aMS/Scale4/pyr1 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube --models vqcleanM0aMS --netG ed023ems --num_scales 4 --lamb 5 --lamb_pyr 1 --tracking_uri sqlite://///home/gary/workspace/logs/mlflow.db --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005

# fuse MS
CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/vqcleanM0aMSadv0/Scale4/max5skip4 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --lamb 5 --models vqcleanM0aMS --num_scales 4 --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005 --netG ed023ems --adv_ms 0 --tracking_uri MS





