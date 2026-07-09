"""Concatenate XY and ZX views of inferred volumes for side-by-side inspection.

For each model tag, reads {base}/{tag}/{stem}.tif — a (Z, Y, X) float32 cube
from test/inference.py — and writes {base}/summary/{tag}.tif where every page
is [XY plane at z=k | ZX plane at y=k] concatenated horizontally. --input adds
the trilinear-upsampled input volume (saved by inference.py --save_input) as
{base}/summary/input.tif for quick comparison.

Usage (see test/inference.sh):
    python test/concat_views.py --base /home/gary/workspace/Data/THX10SDM20xw/out \
        --stem th000008003 --tags MS MSskip MSskipE ... \
        --input /home/gary/workspace/Data/THX10SDM20xw/out/MS/th000008003_input.tif
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
    parser.add_argument('--base', required=True, help='Output dir holding {tag}/{stem}.tif')
    parser.add_argument('--stem', required=True, help='Volume filename stem (no .tif)')
    parser.add_argument('--tags', nargs='+', required=True, help='Model tags, in display order')
    parser.add_argument('--input', default=None,
                        help='Trilinear-upsampled input .tif -> {base}/summary/input.tif')
    args = parser.parse_args()

    dest = os.path.join(args.base, 'summary')
    os.makedirs(dest, exist_ok=True)

    if args.input:
        write_concat(args.input, os.path.join(dest, 'input.tif'))
    for tag in args.tags:
        write_concat(os.path.join(args.base, tag, args.stem + '.tif'),
                     os.path.join(dest, tag + '.tif'))


if __name__ == '__main__':
    main()
