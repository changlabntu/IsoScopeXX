# fuse
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/vqcleanM0a/Scale4/max5skip4 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --lamb 5 --models vqcleanM0a --num_scales 4 --tracking_uri sqlite://///home/gary/workspace/logs/mlflow.db --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005

# fuse
CUDA_VISIBLE_DEVICES=2 NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj fuse/vqcleanM0a/Scale4/max5skip4 --env brcb --dataset E2507218fuse/E2507218cube/ --direction zcube/ --lamb 5 --models vqcleanM0a --num_scales 4 --tracking_uri mlflow --cropsize 192 --cropz 192 --dsp 8 --lr 0.0005

