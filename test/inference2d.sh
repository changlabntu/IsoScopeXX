#!/usr/bin/env bash
# 2D VQ-autoencoder reconstructions (the per-slice encoder->quantizer->decoder
# head every model runs before the 3D generator) of the first val volume, for
# the thx10-071226 clean gamma-foreground set (EXPERIMENTS.md §1) + the vqclean
# baseline. Output size == input size (48x384x384) — this isolates how much
# detail survives the VQ bottleneck, separately from what the 3D trunk does.
#
# The three roiD192gf runs train with nm=11g (gamma=0.5, gamma_lo=-0.8 from
# their config.json): inference2d.py applies the same floor+gamma to the input
# and by default INVERTS the reconstruction back to the pre-gamma [-1,1] scale
# (pass --no_invert to keep gamma space). input.tif is saved through the same
# round trip, so all outputs and the input are directly comparable — up to the
# noise band below gamma_lo, which the forward clip flattens to -0.8.
#
# vqclean caveat: the bundle's vqclean/roiD192 has no .pth (EXPERIMENTS.md §4);
# the only vqclean with weights is the old logs0 run — nm=00, precrop-bug era,
# trained on roiAdsp4, not roiD. Provenance-grade comparison only.
#
#   out/summary2d/{tag}.tif    slice-wise 2D reconstruction per model
#   out/summary2d/input.tif    the round-tripped raw input, same size
# Run from the repo root.
set -e
OUT=/home/gary/workspace/Data/THX10SDM20xw/out
SRC=/home/gary/workspace/Data/THX10SDM20xw/val/roiD/th000008003.tif
THX=/home/gary/workspace/logs/thx10-071226

TAGS=(skipE   skipU   skipUB  vqclean)
EPOCHS=(300   300     300     600)
CKPTS=(
  "$THX/vqcleanM0aMSskipE/roiD192gf/max5skip4"
  "$THX/vqcleanM0aMSskipU/roiD192gf/max5skip4"
  "$THX/vqcleanM0aMSskipUB/roiD192gf/max5skip4"
  "/home/gary/workspace/logs0/THX10SDM20xw/vqcleanVQ/max5skip4"
)

for i in "${!TAGS[@]}"; do
  EXTRA=""; [ "$i" -eq 0 ] && EXTRA="--save_input $OUT/summary2d/input.tif"
  CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python test/inference2d.py \
    --checkpoint "${CKPTS[i]}" --epoch "${EPOCHS[i]}" \
    --source "$SRC" --out "$OUT/summary2d/${TAGS[i]}.tif" $EXTRA
done
