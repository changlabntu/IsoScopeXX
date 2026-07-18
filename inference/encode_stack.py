"""Encode a stack of large 2D slices to codecs, patch by patch — no decoding.

    python inference/encode_stack.py --model skipU --exp sample0 \
        --source /media/cheese/Ghc_data3/THX10/sample0_crop \
        [--range 0 32] [--half --batch 8] [--tsne]

The source folder holds large 2D tif slices (sorted filename order = Z
order), all the same (H, W) with H and W divisible by --patch (see the
crop_params.json convention from the cropping step). --range BEGIN END picks
slices [BEGIN:END] of the sorted list (default: all) — for a folder of
hundreds of slices, run one 32-slice chunk at a time. The selected chunk is
loaded into RAM once (kept uint16), then every (patch, patch, chunk) volume
of the H/patch x W/patch grid is:

  1. mapped to [-1, 1] with a GLOBAL robust percentile window estimated from
     the full stack (--lo_pct/--hi_pct, default 1/99.9) — one shared intensity
     scale for all patches, like the training data's global map
  2. put through the model's own normalization (nm='11g' gamma for skipU)
  3. encoded to codebook indices, saved as
     {out_base}/{exp}/codec/z{BEGIN}-{END}/{rrr}{ccc}.npz
     (rrr = row / Y-grid index, ccc = col / X-grid index; the z chunk dir
     keeps successive --range chunks of the same experiment separate)

Codec only by default — decode later from the npz files
(inference/inference_latent.py shows the decode path); --tsne additionally
runs inference/latent_tsne.py over the chunk's codec dir (plots + anomaly csv
land in that dir). A norm_params.json in the chunk dir records the intensity
window and all parameters needed to reproduce or invert the encode. Chunks of
one sample should share ONE window: the first chunk estimates it, later
chunks pass it back via --window LO HI (printed and stored in the json).
"""

import argparse
import json
import os
import sys
import time
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import tifffile as tiff  # noqa: E402
import torch  # noqa: E402

from inference import Engine, available, get as registry_get  # noqa: E402

DEFAULT_OUT_BASE = '/home/cheese/workspace/Output'


def main():
    parser = argparse.ArgumentParser(
        description='Global-percentile normalize + encode a 2D-slice stack to codecs')
    parser.add_argument('--model', required=True,
                        help=f'Registry model name ({", ".join(available())})')
    parser.add_argument('--exp', required=True,
                        help='Experiment name: codecs go to {out_base}/{exp}/codec/')
    parser.add_argument('--source', required=True,
                        help='Folder of large 2D .tif slices (sorted name order = Z)')
    parser.add_argument('--range', type=int, nargs=2, default=None, metavar=('BEGIN', 'END'),
                        help='Slice selection: files[BEGIN:END] of the sorted list '
                             '(default: all slices)')
    parser.add_argument('--patch', type=int, default=256, help='Patch size in Y/X (default 256)')
    parser.add_argument('--lo_pct', type=float, default=1.0,
                        help='Lower robust percentile of the global intensity window (default 1)')
    parser.add_argument('--hi_pct', type=float, default=99.9,
                        help='Upper robust percentile (default 99.9)')
    parser.add_argument('--pct_stride', type=int, default=8,
                        help='Pixel stride when sampling the stack for percentiles (default 8)')
    parser.add_argument('--window', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                        help='Reuse a fixed intensity window instead of estimating one — '
                             'pass the first chunk\'s window so all chunks share one scale')
    parser.add_argument('--tsne', action='store_true',
                        help="Also run latent_tsne over this chunk's codec dir afterwards")
    parser.add_argument('--batch', type=int, default=8,
                        help='Patches per encode forward (default 8)')
    parser.add_argument('--epoch', type=int, default=None, help="Override the registry's epoch")
    parser.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')
    parser.add_argument('--half', action='store_true', help='Run model and inputs in fp16')
    parser.add_argument('--out_base', default=DEFAULT_OUT_BASE,
                        help=f'Output base dir (default: {DEFAULT_OUT_BASE})')
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.source, '*.tif'))
                   + glob(os.path.join(args.source, '*.tiff')))
    if not files:
        raise FileNotFoundError(f'No .tif slices in {args.source}')
    z0, z1 = args.range if args.range is not None else (0, len(files))
    files = files[z0:z1]
    if not files:
        raise ValueError(f'--range {z0} {z1} selects no slices')
    chunk = f'z{z0:04d}-{z1:04d}'
    print(f'chunk {chunk}: slices {os.path.basename(files[0])} .. '
          f'{os.path.basename(files[-1])}')

    print(f'loading {len(files)} slices into RAM ...', flush=True)
    stack = np.stack([tiff.imread(f) for f in files])           # (Z, H, W), source dtype
    Z, H, W = stack.shape
    if H % args.patch or W % args.patch:
        raise ValueError(f'slice size ({H}, {W}) not divisible by patch {args.patch} — '
                         'crop first (see the sample0_crop convention)')
    rows, cols = H // args.patch, W // args.patch
    print(f'stack (Z, H, W) = {stack.shape} {stack.dtype}, '
          f'{stack.nbytes / 2**30:.1f} GiB -> grid {rows} x {cols} = {rows * cols} patches')

    if args.window is not None:
        lo, hi = args.window
        print(f'global window (given): [{lo:.1f}, {hi:.1f}]')
    else:
        sub = stack[:, ::args.pct_stride, ::args.pct_stride]
        lo, hi = np.percentile(sub, [args.lo_pct, args.hi_pct])
        print(f'global window: p{args.lo_pct}={lo:.1f}, p{args.hi_pct}={hi:.1f} '
              f'(sampled {sub.size / 1e6:.0f}M px, stride {args.pct_stride}) — '
              f'reuse with --window {lo:.1f} {hi:.1f} for the other chunks')

    eng = Engine.from_registry(args.model, epoch=args.epoch, device=args.device)
    if args.half:
        eng.gan.half()
    epoch = args.epoch if args.epoch is not None else registry_get(args.model).get('epoch')

    root = os.path.join(args.out_base, args.exp)
    codec_dir = os.path.join(root, 'codec', chunk)
    os.makedirs(codec_dir, exist_ok=True)
    norm_params = dict(model=args.model, epoch=epoch, **eng.spec,
                       source=args.source, z_range=[z0, z1], n_slices=Z,
                       slice_hw=[H, W], patch=args.patch,
                       grid_rows_cols=[rows, cols],
                       lo_pct=args.lo_pct, hi_pct=args.hi_pct,
                       window_lo=float(lo), window_hi=float(hi))
    with open(os.path.join(codec_dir, 'norm_params.json'), 'w') as f:
        json.dump(norm_params, f, indent=2)
    meta_str = json.dumps(norm_params)

    def prep(r, c):
        """Grid cell (r, c) -> normalized (1, 1, patch, patch, Z) tensor."""
        p = stack[:, r * args.patch:(r + 1) * args.patch,
                  c * args.patch:(c + 1) * args.patch]           # (Z, y, x)
        v = np.transpose(p, (1, 2, 0)).astype(np.float32)        # (Y, X, Z)
        v = np.clip((v - lo) / (hi - lo), 0, 1) * 2 - 1          # global window -> [-1, 1]
        v = eng.normalize(v)                                     # model nm (11g gamma)
        return torch.from_numpy(v)[None, None]

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    t0 = time.perf_counter()
    n_scales = None
    for i in range(0, len(cells), args.batch):
        chunk = cells[i:i + args.batch]
        x = torch.cat([prep(r, c) for r, c in chunk]).to(eng.device)
        if args.half:
            x = x.half()
        _, indices = eng.encode(x)                               # per scale: (B, Z, hk, wk)
        n_scales = len(indices)
        for b, (r, c) in enumerate(chunk):
            np.savez_compressed(
                os.path.join(codec_dir, f'{r:03d}{c:03d}.npz'), meta=meta_str,
                **{f'scale_{k}': idx[b:b + 1].cpu().numpy().astype(np.int32)
                   for k, idx in enumerate(indices)})
        done = i + len(chunk)
        if done % (args.batch * 25) < args.batch or done == len(cells):
            rate = done / (time.perf_counter() - t0)
            print(f'  {done}/{len(cells)} patches  ({rate:.1f}/s, '
                  f'eta {(len(cells) - done) / rate:.0f}s)', flush=True)

    total = sum(os.path.getsize(os.path.join(codec_dir, f))
                for f in os.listdir(codec_dir) if f.endswith('.npz'))
    raw = stack.nbytes
    print(f'done: {len(cells)} codecs ({n_scales} scales) in {time.perf_counter() - t0:.0f}s '
          f'-> {codec_dir}\ncodec total {total / 2**20:.1f} MiB vs raw {raw / 2**30:.1f} GiB '
          f'({raw / total:.0f}x)')

    if args.tsne:
        del stack                                   # free the RAM before sklearn
        from inference.latent_tsne import run as tsne_run
        tsne_run(codec_dir, out_dir=codec_dir)


if __name__ == '__main__':
    main()
