"""Concatenate two orthogonal views of inferred volumes for side-by-side inspection.

For each model tag, reads {base}/output_3d/{tag}/{stem}.tif — a float32 cube
from test/inference.py — and writes {base}/summary_3d/{tag}.tif where every
page pairs the volume's own page k with its page/row-transposed page k,
concatenated horizontally. For an XY-ordered (Z, Y, X) cube that is
[XY at z=k | ZX at y=k]; for a ZX-ordered cube (inference.py --save_zx) it is
[ZX at y=k | XY at z=k]. --input adds the trilinear-upsampled input volume
(saved by inference.py --save_input) as {base}/summary_3d/input.tif.

--std switches to the uncertainty maps: reads
{base}/output_std/{tag}/{stem}_maskstd.tif (inference.py --std_trd) and writes
the same two-panel concats to {base}/summary_std/{tag}.tif (no input panel —
there is no input maskstd).

Usage (see test/inference3d.sh and test/inferencestd.sh):
    python test/concat_views.py --base /home/gary/workspace/Data/THX10SDM20xw/out \
        --stem th000008003 --tags skipE skipU skipUB vqclean \
        --input /home/gary/workspace/Data/THX10SDM20xw/out/input/th000008003.tif
"""

import argparse
import os

import numpy as np
import tifffile as tiff


def xy_zx(vol):
    """(Z, Y, X) cube -> (pages, rows, 2*cols): [XY at z=k | ZX at y=k]."""
    assert vol.shape[0] == vol.shape[1], (
        f'need Z == Y to pair XY pages with ZX pages, got {vol.shape}')
    zx = vol.transpose(1, 0, 2)          # (Y, Z, X): page y, rows Z, cols X
    return np.concatenate([vol, zx], axis=2)


def write_concat(src_path, dst_path):
    vol = tiff.imread(src_path).astype(np.float32)
    cat = xy_zx(vol)
    tiff.imwrite(dst_path, cat)
    print(f'{os.path.basename(dst_path)}  pages x rows x cols = {cat.shape}')


def main():
    parser = argparse.ArgumentParser(description='Build [XY | ZX] view concats')
    parser.add_argument('--base', required=True,
                        help='Output base dir holding output_3d/{tag}/{stem}.tif')
    parser.add_argument('--stem', required=True, help='Volume filename stem (no .tif)')
    parser.add_argument('--tags', nargs='+', required=True, help='Model tags, in display order')
    parser.add_argument('--input', default=None,
                        help='Trilinear-upsampled input .tif -> {base}/summary/input.tif')
    parser.add_argument('--std', action='store_true',
                        help='Concat {base}/output_std/{tag}/{stem}_maskstd.tif into '
                             '{base}/summary_std/{tag}.tif instead of the mean outputs')
    args = parser.parse_args()

    dest = os.path.join(args.base, 'summary_std' if args.std else 'summary_3d')
    os.makedirs(dest, exist_ok=True)

    if args.input and not args.std:
        write_concat(args.input, os.path.join(dest, 'input.tif'))
    for tag in args.tags:
        src = (os.path.join(args.base, 'output_std', tag, args.stem + '_maskstd.tif') if args.std
               else os.path.join(args.base, 'output_3d', tag, args.stem + '.tif'))
        write_concat(src, os.path.join(dest, tag + '.tif'))


if __name__ == '__main__':
    main()
