"""Estimate the stitch-tile flat-field of a codec's source zarr -> flatfield.npz.

The THX10 acquisition is a stitched 20x montage: every tile boundary is a
broad dark vignetting band on a regular grid (pitch ~1890 raw px, see
regis2/stitch.md). The bands are static through z and multiplicative in the
signal above the dark offset — ~1-2% of total intensity but ~10-20% of the
offset-subtracted signal. A separable gain field G(x, y) = gx(x) * gy(y),
estimated once from a z-mean of the volume, divides them out:

    corrected = dark + (measured - dark) / G

Estimation is grid-constrained for robustness: a coarse band-scale gain
profile (foreground-weighted mean / wide median baseline) is comb-fitted for
the tile pitch+phase, the gain is kept only within a window around each grid
line (elsewhere it is exactly 1 — off-grid structure is specimen, not
vignetting), down-weighted where foreground coverage is thin (specimen
edges), refined with a second pass on the corrected mean, and clipped.
Validated on THX10: grid-line signal depth -10..-22% -> <1.2% residual at
well-covered lines; thin-coverage lines are under- rather than
over-corrected.

Usage (py38zarr env — tensorstore):

    python regis2/build_flatfield.py --codec /media/.../thx10codec/codec \
        [--level 2] [--step 10] [--png flatfield_qc.png]

Reads the source store from the codec's norm_params.json; writes
{codec_root}/flatfield.npz (gx, gy at --level resolution, the level scale,
dark, and the fitted grid). Consumed by inference/decode_stack.py
--flatfield.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
from scipy.ndimage import median_filter, uniform_filter1d  # noqa: E402

from inference.decode_stack import resolve_chunks  # noqa: E402
from inference.zarr_io import open_zarr  # noqa: E402


def raw_gain(mean_xy, fg, axis, dark, smooth, fine):
    """Band-scale gain profile along one in-plane axis, coverage-weighted.

    Foreground-masked mean profile, dark-subtracted, divided by its wide
    median baseline (keeps only band-scale dips). The gain is pulled toward 1
    where foreground coverage is thin (specimen edges: dips there are
    content, not vignetting). Returns (gain, coverage)."""
    other = 1 - axis
    cnt = fg.sum(other)
    cov = cnt / mean_xy.shape[other]
    p = (mean_xy * fg).sum(other) / np.maximum(cnt, 1)
    sig = np.where(cov > 0.05, p - dark, np.nan)
    valid = np.isfinite(sig)
    filled = np.where(valid, sig, np.nanmedian(sig))
    g = uniform_filter1d(filled / np.maximum(median_filter(filled, smooth), 1e-6), fine)
    g[~valid] = 1.0
    trust = np.clip((cov - 0.25) / 0.35, 0.0, 1.0)      # full trust above 60% coverage
    return 1.0 + trust * (g - 1.0), cov


def comb_fit(g, cov, pmin, pmax):
    """Fit the tile grid (pitch, phase) minimizing the mean gain over the comb,
    scored only at well-covered positions."""
    n = len(g)
    ok = cov > 0.4
    best = None
    for P in np.arange(pmin, pmax, 1.0):
        for ph in np.arange(0.0, P, 2.0):
            pos = np.arange(ph, n, P).astype(int)
            pos = pos[ok[pos]]
            if len(pos) < 3:
                continue
            sc = (g[pos] - 1.0).mean()
            if best is None or sc < best[0]:
                best = (sc, float(P), float(ph))
    if best is None:
        raise SystemExit('comb fit failed: not enough well-covered grid positions '
                         '(check --fg / --pitch range)')
    return best[1], best[2]


def grid_mask(n, pitch, phase, frac):
    """True within +-frac*pitch of every grid line."""
    x = np.arange(n)
    return np.abs((x - phase + pitch / 2) % pitch - pitch / 2) <= frac * pitch


def main():
    parser = argparse.ArgumentParser(
        description='Estimate a separable stitch-vignetting flat-field from a '
                    "codec's source zarr")
    parser.add_argument('--codec', required=True,
                        help='A codec/ tree of z chunk dirs, or one chunk dir '
                             '(source store read from norm_params.json)')
    parser.add_argument('--level', type=int, default=2,
                        help='Source pyramid level to estimate on (default 2 = 1/4 '
                             'scale; the bands are >100 raw px wide, so coarse is fine)')
    parser.add_argument('--step', type=int, default=10,
                        help='z stride of the mean blend (default: every 10th slice)')
    parser.add_argument('--dark', type=float, default=None,
                        help='Dark offset in source units (default: p5 of the mean '
                             "blend — NOT the codec's window_lo, which may be hand-set "
                             'above the true background)')
    parser.add_argument('--fg', type=float, default=0.1,
                        help='Foreground threshold: dark + FG * (p99 - dark) '
                             '(default 0.1)')
    parser.add_argument('--pitch', type=float, nargs=2, default=(1400, 2240),
                        metavar=('MIN', 'MAX'),
                        help='Tile-pitch search range in RAW px (default 1400 2240)')
    parser.add_argument('--win', type=float, default=0.2,
                        help='Grid-line window half-width as a fraction of the pitch; '
                             'gain is exactly 1 outside (default 0.2)')
    parser.add_argument('--gmin', type=float, default=0.75,
                        help='Gain floor (default 0.75; vignetting only dims)')
    parser.add_argument('--gmax', type=float, default=1.05,
                        help='Gain ceiling (default 1.05)')
    parser.add_argument('--out', default=None,
                        help='Output npz (default: {codec_root}/flatfield.npz)')
    parser.add_argument('--png', default=None,
                        help='Also write a before/after QC figure here')
    args = parser.parse_args()

    root, names, _ = resolve_chunks(args.codec, None)
    with open(os.path.join(root, names[0], 'norm_params.json')) as f:
        np0 = json.load(f)
    if 'zarr_level' not in np0:
        raise SystemExit(f'codec source is not a zarr store ({np0.get("source")})')
    src = open_zarr(np0['source'], args.level)
    nz = src.shape[0]
    scale = 2 ** args.level // 2 ** int(np0['zarr_level'])
    smooth = round(2400 / scale) | 1                     # baseline >> band width
    fine = max(3, round(124 / scale) | 1)
    print(f'source {np0["source"]} level {args.level}: {src.shape} '
          f'(scale x{scale} vs codec level {np0["zarr_level"]})')

    acc = np.zeros(src.shape[1:], dtype=np.float64)
    zs = range(0, nz, args.step)
    for zi in zs:
        acc += src[zi].read().result()
    mean = acc / len(zs)                                 # (x, y) in-plane
    dark = args.dark if args.dark is not None else float(np.percentile(mean, 5))
    p99 = float(np.percentile(mean, 99))
    fg = mean > dark + args.fg * (p99 - dark)
    print(f'mean of {len(zs)} slices (every {args.step}th of {nz}); dark {dark:.1f}, '
          f'foreground {fg.mean() * 100:.0f}%')

    # pass 1: gain -> comb fit -> keep only on-grid windows
    gains, masks, grids = [], [], []
    for axis, nm in ((0, 'x'), (1, 'y')):
        g1, cov = raw_gain(mean, fg, axis, dark, smooth, fine)
        pitch, phase = comb_fit(g1, cov, args.pitch[0] / scale, args.pitch[1] / scale)
        mask = grid_mask(len(g1), pitch, phase, args.win)
        gains.append(np.where(mask, g1, 1.0))
        masks.append(mask)
        grids.append((pitch, phase, cov))
        print(f'  {nm}: pitch {pitch * scale:.0f} raw px, phase {phase * scale:.0f} raw px')

    # pass 2: refine on the corrected mean, combine, clip
    gx0 = np.clip(gains[0], args.gmin, args.gmax)
    gy0 = np.clip(gains[1], args.gmin, args.gmax)
    corr1 = dark + (mean - dark) / (gx0[:, None] * gy0[None, :])
    final = []
    for axis, (g1, mask) in enumerate(zip(gains, masks)):
        g2, _ = raw_gain(corr1, fg, axis, dark, smooth, fine)
        final.append(np.clip(np.where(mask, g1 * g2, 1.0), args.gmin, args.gmax))
    gx, gy = final

    # QC: residual band depth at the fitted grid lines
    corr = dark + (mean - dark) / (gx[:, None] * gy[None, :])
    for axis, (nm, g) in enumerate((('x', gx), ('y', gy))):
        pitch, phase, cov = grids[axis]
        pos = np.arange(phase, len(g), pitch).astype(int)
        res, _ = raw_gain(corr, fg, axis, dark, smooth, fine)
        print(f'  {nm} grid lines: ' + ', '.join(
            f'{p} (cov {cov[p]:.2f}, gain {(g[p] - 1) * 100:+.0f}%, '
            f'residual {(res[p] - 1) * 100:+.1f}%)' for p in pos))

    out = args.out or os.path.join(root, 'flatfield.npz')
    np.savez(out, gx=gx.astype(np.float32), gy=gy.astype(np.float32),
             scale=scale, dark=dark, level=args.level, step=args.step,
             source=np0['source'],
             pitch_raw=[g[0] * scale for g in grids],
             phase_raw=[g[1] * scale for g in grids],
             win=args.win, gmin=args.gmin, gmax=args.gmax, fg=args.fg)
    print(f'wrote {out}: gx {gx.shape}, gy {gy.shape}, scale {scale}, dark {dark:.1f}')

    if args.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(14, 16))
        for ax, img, t in ((axs[0], mean, 'before'), (axs[1], corr, 'after')):
            im = img.T
            p1, p99v = np.percentile(im[im > dark], [1, 99])
            ax.imshow(np.clip((im - p1) / (p99v - p1), 0, 1), cmap='gray')
            ax.set_title(f'mean blend — {t}')
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(args.png, dpi=90)
        print(f'wrote {args.png}')


if __name__ == '__main__':
    main()
