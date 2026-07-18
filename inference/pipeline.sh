#!/usr/bin/env bash
# Zarr-native encode -> codec (npz) -> decode round trip.
# Copy-paste one block at a time. Zarr steps need the py38zarr env
# (py38pl16 clone + tensorstore): `conda activate py38zarr`.
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
    --what decode --chunk z0096-00128 --cells 10 20 22 31 --orig --batch 2
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
