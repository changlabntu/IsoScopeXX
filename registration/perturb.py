"""Step 1 — build a ground-truth-corrupted stack for the latent registration
experiment.

    python registration/perturb.py --nz 120 --size 1024

Loads a registered ROI from the sample0 OME-Zarr, then simulates serial-section
misalignment: every slice z >= 1 gets a small random similarity transform
RELATIVE to the previous slice (rot ~U(+-rot), shift ~U(+-trans),
scale ~U(1+-scale)), composed cumulatively into an absolute per-slice
ground-truth transform (a random-walk drift, like real section-by-section
acquisition). Slice 0 is untouched, so absolute transforms are anchored
(M_0 = I) and recovered transforms are directly comparable to ground truth.

The ROI is loaded with a margin (--margin) around the analysis crop and warped
before cropping, so warp borders/fill never enter the analysis window (the run
aborts if the sampled drift could exceed the margin).

Outputs under {out_base}/{name}/ (default name reg{size}z{nz}s{seed}):
  original.npy    (Z, Y, X) float32 in [-1, 1] — the pristine crop
  corrupted.npy   (Z, Y, X) float32 in [-1, 1] — the per-slice-warped crop
  gt_transforms.json  relative + absolute params/matrices, ROI, window, seed
  perturb_preview.png XZ reslice of original vs corrupted (drift visible)

Intensity: raw uint16 -> [-1, 1] by a robust percentile window of the loaded
ROI (or a fixed --window LO HI, e.g. sample0's global 99 495). The model's
'11g' normalization is NOT applied here — features.py applies it; these
volumes stay model-free.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from registration import affine  # noqa: E402

DEFAULT_ZARR = '/media/cheese/Ghc_data3/THX10/sample0_crop_ome.zarr'
DEFAULT_OUT_BASE = '/home/cheese/workspace/Output/registration'


def sample_transforms(nz, rot, trans, scale, seed):
    """Relative per-slice similarity jitter + its cumulative absolute chain."""
    rng = np.random.default_rng(seed)
    rel = [(0.0, 0.0, 0.0, 1.0)]
    for _ in range(1, nz):
        rel.append((float(rng.uniform(-rot, rot)),
                    float(rng.uniform(-trans, trans)),
                    float(rng.uniform(-trans, trans)),
                    float(1.0 + rng.uniform(-scale, scale))))
    abs_mats = [affine.identity()]
    for p in rel[1:]:
        abs_mats.append(affine.compose(affine.params_to_mat(*p), abs_mats[-1]))
    return rel, abs_mats


def max_corner_displacement(abs_mats, size):
    """Largest content displacement (px) at the analysis-crop corners."""
    h = size / 2.0
    corners = np.array([[-h, -h], [-h, h], [h, -h], [h, h]]).T   # (2, 4) as (x, y)
    worst = 0.0
    for m in abs_mats:
        d = m[:, :2] @ corners + m[:, 2:] - corners
        worst = max(worst, float(np.abs(d).max()))
    return worst


def main():
    parser = argparse.ArgumentParser(description='Synthetic per-slice misalignment + GT')
    parser.add_argument('--zarr', default=DEFAULT_ZARR,
                        help=f'OME-Zarr store, axes (z, x, y) (default: {DEFAULT_ZARR})')
    parser.add_argument('--level', default='0', help='Zarr pyramid level (default 0)')
    parser.add_argument('--z0', type=int, default=20, help='First slice (default 20)')
    parser.add_argument('--nz', type=int, default=120, help='Number of slices (default 120)')
    parser.add_argument('--x0', type=int, default=None,
                        help='Left edge (X) of the analysis crop (default: centered)')
    parser.add_argument('--y0', type=int, default=None,
                        help='Top edge (Y) of the analysis crop (default: centered)')
    parser.add_argument('--size', type=int, default=1024,
                        help='Analysis crop size in Y and X (default 1024)')
    parser.add_argument('--margin', type=int, default=160,
                        help='Extra px loaded+warped around the crop (default 160)')
    parser.add_argument('--rot', type=float, default=0.5,
                        help='Max RELATIVE per-slice rotation, deg (default 0.5)')
    parser.add_argument('--trans', type=float, default=3.0,
                        help='Max RELATIVE per-slice shift, px (default 3)')
    parser.add_argument('--scale', type=float, default=0.005,
                        help='Max RELATIVE per-slice scale jitter (default 0.005)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--window', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                        help='Fixed intensity window (default: ROI percentiles)')
    parser.add_argument('--lo_pct', type=float, default=1.0)
    parser.add_argument('--hi_pct', type=float, default=99.9)
    parser.add_argument('--name', default=None,
                        help='Experiment dir name (default reg{size}z{nz}s{seed})')
    parser.add_argument('--out_base', default=DEFAULT_OUT_BASE)
    args = parser.parse_args()

    from inference.zarr_io import open_zarr
    arr = open_zarr(args.zarr, args.level)
    Z, X, Y = arr.shape                                   # axes (z, x, y)
    pad = args.size + 2 * args.margin
    x0 = (X - args.size) // 2 if args.x0 is None else args.x0
    y0 = (Y - args.size) // 2 if args.y0 is None else args.y0
    xa, ya = x0 - args.margin, y0 - args.margin
    if not (0 <= args.z0 < args.z0 + args.nz <= Z and 0 <= xa and xa + pad <= X
            and 0 <= ya and ya + pad <= Y):
        raise ValueError(f'ROI z{args.z0}+{args.nz}, x{xa}+{pad}, y{ya}+{pad} outside '
                         f'store (Z, X, Y) = {(Z, X, Y)} — margin included')

    rel, abs_mats = sample_transforms(args.nz, args.rot, args.trans, args.scale, args.seed)
    worst = max_corner_displacement(abs_mats, args.size)
    print(f'max corner displacement over the chain: {worst:.1f} px (margin {args.margin})')
    if worst > args.margin - 4:
        raise ValueError('drift exceeds the margin — raise --margin or lower '
                         '--rot/--trans/--scale (or change --seed)')

    print(f'loading ({args.nz}, {pad}, {pad}) from {args.zarr} ...', flush=True)
    block = arr[args.z0:args.z0 + args.nz, xa:xa + pad, ya:ya + pad].read().result()
    block = np.transpose(block, (0, 2, 1)).astype(np.float32)     # (Z, Y, X)

    if args.window is not None:
        lo, hi = args.window
    else:
        lo, hi = np.percentile(block[:, ::4, ::4], [args.lo_pct, args.hi_pct])
    print(f'intensity window: [{lo:.1f}, {hi:.1f}]')
    block = np.clip((block - lo) / (hi - lo), 0, 1) * 2 - 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = torch.from_numpy(block)[:, None].to(device)               # (Z, 1, P, P)
    mats = np.stack(abs_mats)
    with torch.no_grad():
        warped = affine.warp_slices(t, mats, mode='bicubic', fill=-1.0)
    m = args.margin
    original = block[:, m:m + args.size, m:m + args.size].copy()
    corrupted = warped[:, 0, m:m + args.size, m:m + args.size].cpu().numpy()

    name = args.name or f'reg{args.size}z{args.nz}s{args.seed}'
    out_dir = os.path.join(args.out_base, name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'original.npy'), original)
    np.save(os.path.join(out_dir, 'corrupted.npy'), corrupted)
    with open(os.path.join(out_dir, 'gt_transforms.json'), 'w') as f:
        json.dump(dict(zarr=args.zarr, level=args.level, z0=args.z0, nz=args.nz,
                       x0=x0, y0=y0, size=args.size, margin=args.margin,
                       rot=args.rot, trans=args.trans, scale=args.scale,
                       seed=args.seed, window=[float(lo), float(hi)],
                       rel_params=rel,
                       abs_params=[affine.mat_to_params(mv) for mv in abs_mats],
                       abs_mats=[mv.tolist() for mv in abs_mats]), f, indent=2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    w0 = max(0, args.size // 2 - 256)
    for ax, vol, title in zip(axes, (original, corrupted), ('original', 'corrupted')):
        ax.imshow(vol[:, args.size // 2, w0:w0 + 512], cmap='gray', aspect='auto',
                  vmin=-1, vmax=1)
        ax.set_title(f'{title} — XZ reslice at Y={args.size // 2}')
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'perturb_preview.png'), dpi=150)
    plt.close(fig)

    print(f'wrote original/corrupted ({args.nz}, {args.size}, {args.size}) '
          f'+ gt_transforms.json -> {out_dir}')


if __name__ == '__main__':
    main()
