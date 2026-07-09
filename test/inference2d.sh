#!/usr/bin/env bash
# 2D VQ-autoencoder reconstructions (the per-slice encoder->quantizer->decoder
# head every model runs before the 3D generator) of the first val volume, for
# ALL thx10 MS models + the non-MS vqclean baseline. Output size == input size
# (48x384x384) — this isolates how much detail survives the VQ bottleneck,
# separately from what the 3D trunk does with it (see test/inference.sh).
#   out/summary2d/{tag}.tif    slice-wise 2D reconstruction per model
#   out/summary2d/input.tif    the normalized raw input, same size
# Run from the repo root. Same checkpoint pins/caveats as test/inference.sh.
set -e
OUT=/home/gary/workspace/Data/THX10SDM20xw/out
SRC=/home/gary/workspace/Data/THX10SDM20xw/val/roiD/th000008003.tif
THX=/home/gary/workspace/logs/THX10SDM20xw/thx10

TAGS=(MS            MSskip        MSskipE       MSskipEema999 MSskipPband  MSskipPlse   MSskipUB      vqclean)
EPOCHS=(100         100           100           200           100          100          100           600)
CKPTS=(
  "$THX/vqcleanM0aMSadv0/Scale4/max5skip4/checkpoints/20260704_011354"
  "$THX/vqcleanM0aMSskip/Scale4/max5skip4"
  "$THX/vqcleanM0aMSskipE/Scale4/max5skip4"
  "$THX/vqcleanM0aMSskipE/Scale4/ema999"
  "$THX/vqcleanM0aMSskipP/Scale4/band5"
  "$THX/vqcleanM0aMSskipP/Scale4/lse5"
  "$THX/vqcleanM0aMSskipUB/Scale4/max5skip4coarse0"
  "/home/gary/workspace/logs0/THX10SDM20xw/vqcleanVQ/max5skip4"
)

for i in "${!TAGS[@]}"; do
  EXTRA=""; [ "$i" -eq 0 ] && EXTRA="--save_input $OUT/summary2d/input.tif"
  CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python test/inference2d.py \
    --checkpoint "${CKPTS[i]}" --epoch "${EPOCHS[i]}" \
    --source "$SRC" --out "$OUT/summary2d/${TAGS[i]}.tif" $EXTRA
done
