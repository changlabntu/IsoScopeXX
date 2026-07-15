#!/usr/bin/env bash
# 3D isotropic inference on val volumes with test/inference.py — run from the
# repo root. (2D VQ-head reconstructions: test/inference2d.sh.)
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
#CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python test/inference.py --checkpoint /home/gary/workspace/logs0/THX10SDM20xw/vqcleanVQ/max5skip4 --epoch 600 --source /home/gary/workspace/Data/THX10SDM20xw/val/roiD/th000008003.tif --destination test/out/vqclean_max5skip4_ep600_train

# ---------------------------------------------------------------------------
# The thx10-071226 clean gamma-foreground set (EXPERIMENTS.md §1) + vqclean
# baseline on the first NVOL val volumes, train mode (MC dropout, the default
# — one stochastic draw each). Outputs live under the dataset's out/ dir (also
# the default --destination base in test/inference.py):
#   out/{tag}/{stem}.tif        raw isotropic output per model (384^3), saved
#                               with --save_zx: ZX page order (page y=k, rows
#                               Z, cols X — the synthesized axis in-plane)
#   out/input/{stem}.tif        the trilinear-upsampled inputs, same ZX order
#                               (model dirs hold outputs only)
#   out/summary/{tag}.tif       [ZX page y=k | XY page z=k] concat per model —
#                               FIRST volume only, even when NVOL > 1
#   out/summary/input.tif       same concat of the first trilinear-upsampled input
# (concat_views.py pairs each page with its transpose, so ZX-ordered volumes
# come out ZX in the left panel, XY in the right — swapped vs. XY-ordered.)
# The three roiD192gf runs train with nm=11g (gamma=0.5, gamma_lo=-0.8 from
# config.json): inference.py applies the same floor+gamma to the input and by
# default inverts XupX (and the saved Xup input) back to the pre-gamma [-1,1]
# scale (--no_invert keeps gamma space) — same logic as test/inference2d.sh.
set -e
OUT=/home/gary/workspace/Data/THX10SDM20xw/out
SRC=/home/gary/workspace/Data/THX10SDM20xw/val/roiD
NVOL=20
THX=/home/gary/workspace/logs/thx10-071226

# Stem of the first val volume — the only one written into out/summary/.
STEM=$(basename "$(ls "$SRC"/*.tif | head -1)" .tif)

# vqclean caveat: the bundle's vqclean/roiD192 has no .pth (EXPERIMENTS.md §4);
# the only vqclean with weights is the old logs0 run — nm=00, precrop-bug era,
# trained on roiAdsp4, not roiD. Provenance-grade comparison only. Its
# generation() applies cropz unconditionally — inference.py zeroes it on load.
TAGS=(skipE   skipU   skipUB  vqclean)
EPOCHS=(300   300     300     600)
CKPTS=(
  "$THX/vqcleanM0aMSskipE/roiD192gf/max5skip4"
  "$THX/vqcleanM0aMSskipU/roiD192gf/max5skip4"
  "$THX/vqcleanM0aMSskipUB/roiD192gf/max5skip4"
  "/home/gary/workspace/logs0/THX10SDM20xw/vqcleanVQ/max5skip4"
)

for i in "${!TAGS[@]}"; do
  # --save_input on the first model: the trilinear-upsampled inputs (identical
  # across models) are written once to out/input/ as the shared reference.
  EXTRA=""; [ "$i" -eq 0 ] && EXTRA="--save_input"
  CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python test/inference.py \
    --checkpoint "${CKPTS[i]}" --epoch "${EPOCHS[i]}" \
    --source "$SRC" --limit "$NVOL" --destination "$OUT/${TAGS[i]}" --save_zx $EXTRA
done

python test/concat_views.py --base "$OUT" --stem "$STEM" --tags "${TAGS[@]}" \
  --input "$OUT/input/$STEM.tif"
