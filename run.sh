# default 8X enhancement
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --nocut
# default 4X enhancement
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --nocut --downbranch 2 ---cropz 32 # (128 // 4)
# 8X enhancement with contrastive loss lbNCE=1
NO_ALBUMENTATIONS_UPDATE=1 python train.py --yaml aisr --prj default/max10skip4 --env t09 --lbNCE 1

