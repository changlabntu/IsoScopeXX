#!/usr/bin/env bash
# Inference on val volumes with test/inference.py — run from the repo root.
# Same convention as run.sh: the last uncommented line is the current focus.

# MSskipE max5skip4, epoch 100 (best val_kid 3.99 in thx-MS)
#CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python test/inference.py --checkpoint /home/gary/workspace/logs/THX10SDM20xw/thx10/vqcleanM0aMSskipE/Scale4/max5skip4 --epoch 100 --source /home/gary/workspace/Data/THX10SDM20xw/val/roiD --destination test/out/MSskipE_max5skip4_ep100

# MSskipE ema999, epoch 200 (best val_lpips_pred 0.597 in thx-MS; EMA weights) — full val set, eval mode
#CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python test/inference.py --checkpoint /home/gary/workspace/logs/THX10SDM20xw/thx10/vqcleanM0aMSskipE/Scale4/ema999 --epoch 200 --source /home/gary/workspace/Data/THX10SDM20xw/val/roiD --destination test/out/MSskipE_ema999_ep200

# MSskipE ema999 ep200, first val volume, MC-dropout train mode (now the default; add --eval for deterministic)
#CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python test/inference.py --checkpoint /home/gary/workspace/logs/THX10SDM20xw/thx10/vqcleanM0aMSskipE/Scale4/ema999 --epoch 200 --source /home/gary/workspace/Data/THX10SDM20xw/val/roiD/th000008003.tif --destination test/out/MSskipE_ema999_ep200_train

# Vanilla MS (vqcleanM0aMS) ep100 — note: the MSadv0/max5skip4 experiment holds two runs; this timestamped
# dir is the vanilla-MS one (the newer 20260704_15xxxx dirs are the MSfpn restart).
#CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python test/inference.py --checkpoint /home/gary/workspace/logs/THX10SDM20xw/thx10/vqcleanM0aMSadv0/Scale4/max5skip4/checkpoints/20260704_011354 --epoch 100 --source /home/gary/workspace/Data/THX10SDM20xw/val/roiD/th000008003.tif --destination test/out/MS_max5skip4_ep100_train

# Non-MS baselines: vqclean single-VQ, max5skip4 recipe (logs0; trained on direction roiAdsp4,
# unlike the MS runs' roiD). Their generation() applies cropz unconditionally — inference.py
# zeroes hparams.cropz after loading so the full Z depth is processed.
#CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python test/inference.py --checkpoint /home/gary/workspace/logs0/THX10SDM20xw/vqcleanVQ/192/max5skip4/checkpoints/20260406_004228 --epoch 700 --source /home/gary/workspace/Data/THX10SDM20xw/val/roiD/th000008003.tif --destination test/out/vqclean192_max5skip4_ep700_train
CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python test/inference.py --checkpoint /home/gary/workspace/logs0/THX10SDM20xw/vqcleanVQ/max5skip4 --epoch 600 --source /home/gary/workspace/Data/THX10SDM20xw/val/roiD/th000008003.tif --destination test/out/vqclean_max5skip4_ep600_train
