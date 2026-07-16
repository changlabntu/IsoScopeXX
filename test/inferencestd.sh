#!/usr/bin/env bash
# Boundary-uncertainty (maskstd) summary sweep — run from the repo root.
# For every model in the thx10-071226 bundle, runs the FIRST val volume with
# TTA + MC dropout (--tta --mc 4 = 8 pooled passes, ~17 s/model + load) and a
# foreground threshold --std_trd: every pass is binarized at it and the mask's
# across-pass std sqrt(p(1-p)) — nonzero only where passes disagree, i.e. the
# uncertain foreground boundary — lands in out/output_std/{tag}/{stem}_maskstd.tif.
# concat_views.py --std then builds the same two-panel view concats as the
# mean-output summary into:
#   out/summary_std/{tag}.tif   [ZX page y=k | XY page z=k] of the maskstd map
# The mean outputs this run also produces overwrite out/output_3d/{tag}/{stem}.tif
# with the 8-pass average (same convention as inference3d.sh with --tta).
# STD_TRD is in the SAVED output intensity scale (the gf runs invert back to
# pre-gamma [-1, 1]; inference.py maps it through the forward gamma itself).
# vqclean is NOT in this sweep: its only weights lived under logs0 (see
# inference3d.sh caveat) and logs0 no longer exists on this box (2026-07-16) —
# out/vqclean/ holds stale July-13 outputs that can't be regenerated here.
set -e
OUT=/home/gary/workspace/Data/THX10SDM20xw/out
SRC=/home/gary/workspace/Data/THX10SDM20xw/val/roiD
STD_TRD=-0.7
MC=4
THX=/home/gary/workspace/logs/thx10-071226

STEM=$(basename "$(ls "$SRC"/*.tif | head -1)" .tif)

TAGS=(skipE   skipU   skipUB)
EPOCHS=(300   300     300)
CKPTS=(
  "$THX/vqcleanM0aMSskipE/roiD192gf/max5skip4"
  "$THX/vqcleanM0aMSskipU/roiD192gf/max5skip4"
  "$THX/vqcleanM0aMSskipUB/roiD192gf/max5skip4"
)

for i in "${!TAGS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python test/inference.py \
    --checkpoint "${CKPTS[i]}" --epoch "${EPOCHS[i]}" \
    --source "$SRC/$STEM.tif" --destination "$OUT" --tag "${TAGS[i]}" \
    --save_zx --tta --mc "$MC" --std_trd "$STD_TRD"
done

python test/concat_views.py --base "$OUT" --stem "$STEM" --tags "${TAGS[@]}" --std
