#!/usr/bin/env bash
# Zarr-native encode -> codec (npz) -> decode round trip.
# Copy-paste one block at a time. EVERY block here needs the py38zarr env
# (py38pl16 clone + tensorstore): `conda activate py38zarr` — encode streams
# the zarr source, and the decode blocks all use --orig or --zarr.
#
# Paths for sample0 (THX10) — edit ZARR/EXP for another sample.
ZARR=/media/cheese/Ghc_data3/THX10/sample0_crop_ome.zarr   # OME-Zarr (z,x,y) source
EXP=sample0zarrval                                         # -> $OUT/$EXP/codec/
OUT=/home/cheese/workspace/Output


# 1. ENCODE the whole ome.zarr -> per-patch npz codecs.
#    Streams every z-chunk in ONE run (no per-chunk --window handoff): the exact
#    global intensity window is computed once over all 160 slices via a one-pass
#    uint16 histogram, then shared by every chunk. Output: $OUT/$EXP/codec/z{a}-{b}/{rrr}{ccc}.npz
#    (35x57 grid x 5 z-chunks of 32) + a norm_params.json per chunk.
CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python inference/encode_stack.py \
    --model skipU --exp $EXP --zarr $ZARR --half --batch 8


# 2. RECONSTRUCTION vs ORIGINAL, one row (visual check of the VQ round trip).
#    --what reconstruction = the slice-wise VQ-head output at INPUT resolution
#    (no Z super-res), written as XY pages. --orig also dumps the raw source
#    patches for the same row straight from the codec's recorded zarr, at input
#    resolution and identical (Z, Y, X) shape, so the two TIFFs overlay 1:1.
#    Row 17 of the 35-row grid -> both strips are (32, 256, 14592) uint16.
CUDA_VISIBLE_DEVICES=1 python inference/decode_stack.py \
    --codec $OUT/$EXP/codec --tiff $OUT/$EXP/tifs \
    --what reconstruction --chunk z0000-0032 --row 17 --orig --batch 2
#    -> $OUT/$EXP/tifs/reconstruction/z0000-0032_row017.tif  (VQ recon)
#       $OUT/$EXP/tifs/original/z0000-0032_row017.tif         (raw source, same units)


# 3. ENHANCEMENT vs ORIGINAL, a 5x4 cell block of the SECOND chunk, ZX view.
#    --what decode = the isotropic super-resolved volume (Z x uprate=8), written
#    as ZX pages (synthesized Z in-plane). --cells 15 20 27 31 = grid rows 15-19
#    x cols 27-30 (half-open) = 20 patches. --orig here dumps each source patch
#    TRILINEAR Z-upsampled to the same 256 Z and as ZX pages too, so enhancement
#    (smooth Z) and original (blocky Z) sit in the identical (256, 256, 256) view.
#    Per-patch files (cell mode does not stitch): decode_* and original_* per cell.
CUDA_VISIBLE_DEVICES=1 python inference/decode_stack.py \
    --codec $OUT/$EXP/codec --tiff $OUT/$EXP/tifs_enh \
    --what decode --chunk z0032-0064 --cells 15 20 27 31 --orig --batch 2
#    -> $OUT/$EXP/tifs_enh/decode/z0032-0064_r{015..019}c{027..030}.tif   (SR, ZX)
#       $OUT/$EXP/tifs_enh/original/z0032-0064_r{015..019}c{027..030}.tif (trilinear, ZX)


# 4. ENHANCEMENT to ZARR for a selected ROI, with the interpolated original too.
#    Pick the ROI: --chunk = which z-chunk (32 input slices -> 256 SR slices),
#    --cells R0 R1 C0 C1 = the X/Y patch block (half-open, patch 256). --crop
#    sizes the store to the ROI bounding box (origin at the ROI corner), fully
#    populated -> a small, napari-friendly store (WITHOUT --crop the store is the
#    whole-volume 334 GB shape written sparsely: single-scale + mostly empty, so
#    napari crawls). --orig writes the source, TRILINEAR Z-upsampled to the SR
#    size, into a sibling <zarr>_original.zarr of identical shape, so enhancement
#    and its interpolated original overlay 1:1. Both are native (z,x,y) OME 0.5.
ROI_CHUNK=z0064-0096        # <- select z-chunk
ROI_CELLS="18 21 28 32"     # <- select R0 R1 C0 C1 (rows x cols of the 35x57 grid)
CUDA_VISIBLE_DEVICES=1 python inference/decode_stack.py \
    --codec $OUT/$EXP/codec --zarr $OUT/$EXP/enh_roi.zarr \
    --what decode --chunk $ROI_CHUNK --cells $ROI_CELLS --crop --orig --batch 2
#    -> $OUT/$EXP/enh_roi.zarr           (isotropic SR enhancement, uint16 z,x,y)
#       $OUT/$EXP/enh_roi_original.zarr  (trilinear-interpolated original, same shape)


# ============================================================================
# THX10 FULL VOLUME (thx10cropzarr) — encoded 2026-07-19, patch 384.
# Geometry differs from sample0: (z, x, y) = (816, 14208, 8448), store chunks
# (48, 384, 384), grid 22 rows x 37 cols of 384x384 patches, 17 z-chunks of 48
# (z0000-0048 .. z0768-0816). Codecs live OUTSIDE $OUT, next to the raw store.
FZARR=/media/cheese/Ghc_data3/THX10/thx10cropzarr
FBASE=/media/cheese/Ghc_data3/THX10          # -> $FBASE/thx10codec/codec/
FCODEC=$FBASE/thx10codec/codec

# FIXED intensity window for THX10 — ALWAYS pass this, NEVER let encode_stack
# scan. It is roiAdsp4.tif's OWN p0.1/p99.9 (the training patches' exact
# `percentile=True` mapping in Data/thx10/to_patch.py). A fresh full-store scan
# instead returns [0, 522] because the store's p1 lands on the ~3% zero padding
# (p1=0); that pushes the material median to gamma -0.56, OUT of the model's
# training regime, and is what caused the patch-by-patch contrast/offset drift.
# [113, 588] pins the matrix to the gamma floor (median -1.0) as in training:
# VQ-recon gain 0.96 / offset ~0 (vs 1.8 / -35 under [0, 522]).
FWIN="113 588"                # <- the only window to use; passing it skips the scan

# F1. ENCODE THE FULL VOLUME. fp16 batch 2 peaks ~9 GiB; ~230 s/chunk, ~35 min,
#     ~1.9 GB codecs. 17 z-chunks z0000-0048 .. z0768-0816, 814 npz each.
CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python inference/encode_stack.py \
    --model skipU --exp thx10codec --zarr $FZARR --out_base $FBASE \
    --patch 384 --half --batch 2 --window $FWIN

# F1b. ENCODE ONE MIDDLE CHUNK (testing) — z0384-0432 is the middle of the 17.
#      --range BEGIN END selects those input slices; still --window $FWIN (no
#      scan). ~5 min -> $FCODEC/z0384-0432/ (814 npz + norm_params.json).
CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 python inference/encode_stack.py \
    --model skipU --exp thx10codec --zarr $FZARR --out_base $FBASE \
    --patch 384 --half --batch 2 --window $FWIN --range 384 432

# F2. Decode blocks: SAME flags as steps 2-4 above with $FCODEC and the 384
#     geometry, but --half is REQUIRED — a 384-patch '--what decode' (output
#     384^3) OOMs in fp32 on a 24 GB card (fp16 peaks ~9 GiB; the codecs were
#     fp16-encoded anyway). patch/window/uprate are read from norm_params.json.
#     One z-chunk decodes 48 input slices -> 384 SR slices; the FULL volume is
#     (6528, 14208, 8448) uint16 = ~1.57 TB logical, so keep --crop for ROIs.
#     ROI: central 5x6-patch block (rows 9-13 x cols 16-21 of the 22x37 grid)
#     of the middle z-chunk. Enhancement + trilinear original land as sibling
#     OME-Zarr stores under $FBASE/thx10out/.
CUDA_VISIBLE_DEVICES=1 python inference/decode_stack.py \
    --codec $FCODEC --zarr $FBASE/thx10out/enh_z0384-0432.zarr \
    --what decode --chunk z0384-0432 --cells 9 14 16 22 \
    --crop --orig --half --batch 1
#    -> $FBASE/thx10out/enh_z0384-0432.zarr           (5x6 patches: (384, 2304, 1920) SR)
#       $FBASE/thx10out/enh_z0384-0432_original.zarr  (trilinear original, same shape)
