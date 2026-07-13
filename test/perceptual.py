"""Isotropy perceptual metrics: FID / KID / LPIPS between the XY and XZ views
of generated 3D volumes.

For every 3D .tif in --source (e.g. .../out/vqclean/), the volume is sliced
into its XY planes (the native high-res orientation, used as the "real" set)
and its XZ planes (the view crossing the synthesized Z axis, the "fake" set),
and each selected metric is computed between the two sets of slices. The
per-volume values are then reported as mean +/- std over the whole folder.

Conventions match models/base.py's validation metrics so numbers are
comparable with the logged val_lpips_* / val_kid: taming LPIPS on [-1, 1]
floats, torchmetrics KID with feature=64 / subset_size=16, and the fixed
clamp(-1, 1) -> uint8 [0, 255] map for the Inception metrics. LPIPS pairs the
i-th XY slice with the i-th XZ slice (arbitrary pairing, as in base.py).

Volumes are read per --order: 'zx' (the test/inference3d.sh --save_zx layout:
page y=k, rows Z, cols X) or 'xy' (page z=k, rows Y, cols X).

Usage:
    python test/perceptual.py --source /home/gary/workspace/Data/THX10SDM20xw/out/vqclean \
        [--metrics fid kid lpips] [--order zx] [--stride 1] [--limit 0]
"""

import argparse
import os
import sys
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import tifffile as tiff  # noqa: E402
import torch  # noqa: E402


def load_views(path, order, stride):
    """Read one 3D .tif and return (xy, xz) slice stacks as (N, 1, H, W) tensors.

    xy: XY planes indexed by z (rows Y, cols X) — the native high-res view.
    xz: XZ planes indexed by y (rows Z, cols X) — crosses the synthesized axis.
    """
    vol = tiff.imread(path).astype(np.float32)
    if vol.ndim != 3:
        raise ValueError(f'{path}: expected a 3D volume, got shape {vol.shape}')
    if order == 'zx':                      # (Y, Z, X) -> (Z, Y, X)
        vol = np.transpose(vol, (1, 0, 2))
    xy = vol[::stride]                     # (Z', Y, X)
    xz = vol.transpose(1, 0, 2)[::stride]  # (Y', Z, X)
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[:, None]  # noqa: E731
    return to_t(xy), to_t(xz)


def to_rgb(t):
    return t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t


def to_uint8(t):
    """clamp(-1, 1) -> uint8 [0, 255] RGB — same map as models/base.py."""
    t = to_rgb(t).clamp(-1, 1)
    return ((t + 1) / 2 * 255).to(torch.uint8)


def batched(t, n):
    for i in range(0, t.shape[0], n):
        yield t[i:i + n]


@torch.no_grad()
def lpips_xy_xz(lpips_fn, xy, xz, device, batch):
    """Mean LPIPS over i-th XY slice vs i-th XZ slice pairs."""
    n = min(xy.shape[0], xz.shape[0])
    if xy.shape[-2:] != xz.shape[-2:]:
        raise ValueError(f'LPIPS needs equal slice sizes, got XY {tuple(xy.shape[-2:])} '
                         f'vs XZ {tuple(xz.shape[-2:])} (non-cubic volume)')
    vals = []
    for i in range(0, n, batch):
        a = to_rgb(xy[i:i + batch].clamp(-1, 1)).to(device)
        b = to_rgb(xz[i:i + batch].clamp(-1, 1)).to(device)
        vals.append(lpips_fn(a, b).flatten())
    return float(torch.cat(vals).mean())


@torch.no_grad()
def inception_distance(metric, xy, xz, device, batch):
    """Update a torchmetrics FID/KID with XY as real, XZ as fake; return scalar."""
    metric.reset()
    for chunk in batched(xy, batch):
        metric.update(to_uint8(chunk).to(device), real=True)
    for chunk in batched(xz, batch):
        metric.update(to_uint8(chunk).to(device), real=False)
    out = metric.compute()
    return float(out[0] if isinstance(out, tuple) else out)  # KID -> (mean, std)


def main():
    parser = argparse.ArgumentParser(
        description='FID/KID/LPIPS between XY and XZ views of generated volumes')
    parser.add_argument('--source', required=True, help='Directory of 3D .tif volumes')
    parser.add_argument('--metrics', nargs='+', default=['fid', 'kid', 'lpips'],
                        choices=['fid', 'kid', 'lpips'])
    parser.add_argument('--order', default='zx', choices=['zx', 'xy'],
                        help="Saved page order: 'zx' = inference3d.sh --save_zx layout "
                             "(page y, rows Z), 'xy' = page z, rows Y (default: zx)")
    parser.add_argument('--stride', type=int, default=1,
                        help='Use every k-th slice of each view (default: 1 = all)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Only evaluate the first N files, sorted (0 = all)')
    parser.add_argument('--batch', type=int, default=32, help='Slice batch size on GPU')
    parser.add_argument('--feature', type=int, default=64, choices=[64, 192, 768, 2048],
                        help='Inception feature dim for FID/KID (64 = models/base.py val_kid)')
    parser.add_argument('--kid_subset', type=int, default=16,
                        help='KID subset_size (16 = models/base.py val_kid)')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.source, '*.tif'))
                   + glob(os.path.join(args.source, '*.tiff')))
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f'No .tif files found in {args.source}')

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    lpips_fn = None
    if 'lpips' in args.metrics:
        from taming.modules.losses.lpips import LPIPS
        lpips_fn = LPIPS().eval().to(device)
        for p in lpips_fn.parameters():
            p.requires_grad = False
    fid_metric = None
    if 'fid' in args.metrics:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid_metric = FrechetInceptionDistance(feature=args.feature).to(device)
    kid_metric = None
    if 'kid' in args.metrics:
        from torchmetrics.image.kid import KernelInceptionDistance
        kid_metric = KernelInceptionDistance(feature=args.feature,
                                             subset_size=args.kid_subset).to(device)

    print(f'{len(files)} volume(s) in {args.source}  '
          f'(order={args.order}, stride={args.stride}, metrics={",".join(args.metrics)})')

    results = {m: [] for m in args.metrics}
    for path in files:
        xy, xz = load_views(path, args.order, args.stride)
        row = {}
        if fid_metric is not None:
            row['fid'] = inception_distance(fid_metric, xy, xz, device, args.batch)
        if kid_metric is not None:
            row['kid'] = inception_distance(kid_metric, xy, xz, device, args.batch)
        if lpips_fn is not None:
            row['lpips'] = lpips_xy_xz(lpips_fn, xy, xz, device, args.batch)
        for m, v in row.items():
            results[m].append(v)
        print('  ' + os.path.basename(path) + '  '
              + '  '.join(f'{m} {v:.4f}' for m, v in row.items()))

    print(f'\n{os.path.basename(args.source.rstrip("/"))} (n={len(files)}):')
    for m in args.metrics:
        v = np.array(results[m])
        print(f'  {m:6s} {v.mean():.4f} +/- {v.std():.4f}')


if __name__ == '__main__':
    main()
