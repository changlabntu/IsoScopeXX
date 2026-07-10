#!/usr/bin/env python
"""Precompute per-view normalization constants for the '11p'/'11pg' dataloader methods.

For each source folder under `<dataset>/train/` it computes robust percentiles
(`--lo`/`--hi`) over a sample of cubes and writes them to
`<dataset>/norm_stats.json`. The dataloader (`nm='11p'`) then maps
`[lo, hi] -> [-1, 1]` per view, harmonizing the per-view brightness/gain gap
while leaving the irreducible cross-view structural difference untouched.

The constants are computed on `train/` only and shared by both train and val,
so validation is normalized with the exact same mapping.

Examples
--------
    # resolve DATASET from cfg/env.json[brcb]
    python tools/compute_norm_stats.py --env brcb --dataset E2507218fuse/E2507218cube

    # only specific views, custom percentiles, more sampled cubes
    python tools/compute_norm_stats.py --env brcb --dataset E2507218fuse/E2507218cube \
        --direction zcube_xcube_ycube --lo 0.5 --hi 99.9 --n 16

    # or point straight at the dataset dir (the one containing train/)
    python tools/compute_norm_stats.py --dataroot /home/gary/workspace/Data/E2507218fuse/E2507218cube
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile as tiff


def resolve_dataroot(args) -> Path:
    if args.dataroot:
        return Path(args.dataroot)
    if not (args.env and args.dataset):
        sys.exit("Provide --dataroot, or both --env and --dataset.")
    env_file = Path(__file__).resolve().parent.parent / 'cfg' / 'env.json'
    with open(env_file) as f:
        envs = json.load(f)
    if args.env not in envs:
        sys.exit(f"--env '{args.env}' not in {env_file} (have {sorted(envs)}).")
    dataset_root = envs[args.env].get('DATASET')
    if not dataset_root:
        sys.exit(f"env '{args.env}' has no DATASET entry.")
    return Path(dataset_root) / args.dataset


def view_folders(train_dir: Path, direction: str):
    if direction:
        names = direction.split('_')
    else:
        names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    folders = [train_dir / n for n in names]
    for f in folders:
        if not f.is_dir():
            sys.exit(f"Not a directory: {f}")
    return folders


def folder_percentiles(folder: Path, n: int, stride: int, lo: float, hi: float):
    files = sorted(folder.glob('*.tif'))[:n]
    if not files:
        sys.exit(f"No .tif files in {folder}")
    vals = [tiff.imread(str(f)).astype(np.float32).ravel()[::stride] for f in files]
    vals = np.concatenate(vals)
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi)), len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--env', type=str, default=None, help='machine profile in cfg/env.json (gives DATASET)')
    ap.add_argument('--dataset', type=str, default=None, help='dataset subdir under DATASET')
    ap.add_argument('--dataroot', type=str, default=None, help='direct path to dataset dir (containing train/)')
    ap.add_argument('--direction', type=str, default=None, help="underscore-joined views, e.g. zcube_xcube_ycube; default = all train/ subdirs")
    ap.add_argument('--lo', type=float, default=0.5, help='low percentile -> -1 (default 0.5)')
    ap.add_argument('--hi', type=float, default=99.9, help='high percentile -> +1 (default 99.9)')
    ap.add_argument('--n', type=int, default=8, help='number of cubes sampled per view (default 8)')
    ap.add_argument('--stride', type=int, default=16, help='voxel subsample stride for speed (default 16)')
    ap.add_argument('--out', type=str, default=None, help='output path (default <dataset>/norm_stats.json)')
    args = ap.parse_args()

    dataroot = resolve_dataroot(args)
    train_dir = dataroot / 'train'
    if not train_dir.is_dir():
        sys.exit(f"No train/ under {dataroot}")
    folders = view_folders(train_dir, args.direction)

    stats = {}
    for folder in folders:
        plo, phi, used = folder_percentiles(folder, args.n, args.stride, args.lo, args.hi)
        stats[folder.name] = [plo, phi]
        print(f"  {folder.name:12s}  p{args.lo}={plo:+.4f}  p{args.hi}={phi:+.4f}  (n={used})")

    out = Path(args.out) if args.out else dataroot / 'norm_stats.json'
    payload = {'lo_pct': args.lo, 'hi_pct': args.hi, 'n_files': args.n,
               'stride': args.stride, 'stats': stats}
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
