"""Z-windowed 2-D latent-spectrum overlay (fixes the cluster_texture_all figs).

The original coarse_spectrum overlays colored whole-chunk Z-pooled features on
a single middle slice, gated by latent density, on a 1-D turbo ramp — three
compounding mismatches (see regis2/latent_cluster.md). This script fixes them:

1. Features pool only the central ``2*zwin+1`` latent Z planes
   (``load_latent_features(..., zwin=w)``), so the color describes the
   displayed slice's neighbourhood, not a 48-slice average.
2. The content gate is raw-pixel local contrast (patch std vs. a noise floor
   from the darkest-decile patches), so dim-but-structured regions are kept.
3. Color encodes TWO axes: PC1 -> hue (blue -> red), PC2 -> lightness, with a
   2-D legend inset carrying each PC's variance share.

Outputs, under ``--out`` (default Output/regis2/cluster_fable), for an
experiment name XXX (``--name``, default ``zw{zwin}_{blocks}``):
  XXX.npy   float32 (N, 6) — one row per gated content patch:
            (z, x, y, pc1, pc2, pc3); z = the chunk's displayed mid slice,
            x/y = patch-centre pixel coords at the codec's zarr level (same
            frame as thx10codec gaps.csv), pc* sign-aligned (pc1 with
            fine-scale energy, pc2/pc3 with raw patch std).
  XXX/      one overlay PNG per chunk.
  feat_zw{w}_{chunk}.npz  shared per-chunk feature caches (re-rendering and
            re-analysis are GPU-free once present).

Run (py38zarr env):
  CUDA_VISIBLE_DEVICES=0 python clustering/latent_overlay.py \
    [--chunks z0288-0336,z0624-0672] [--zwin 3] [--gate_k 2.0] [--blocks coarse]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, hsv_to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from inference.latent_tsne import balance_scales, load_latent_features  # noqa: E402
from inference.zarr_io import open_zarr  # noqa: E402

DEFAULT_CODEC = '/media/cheese/Ghc_data3/THX10/thx10codec/codec'
DEFAULT_OUT = '/home/cheese/workspace/Output/regis2/cluster_fable'

_T = np.linspace(0, 1, 256)
PC1_CMAP = ListedColormap(hsv_to_rgb(np.stack(         # same hue ramp as the
    [0.72 * (1 - _T), np.full_like(_T, 0.85),          # boxes style, fixed
     np.full_like(_T, 0.80)], -1)))                    # lightness


def chunk_features(codec, chunk, out, zwin, spatial, model, epoch, device):
    """Per-scale Z-windowed max-pooled features for one chunk dir, cached."""
    cache = os.path.join(out, f'feat_zw{zwin}_{chunk}.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return list(z['stems']), z['fps'], [tuple(b) for b in z['blocks']]
    stems, fps, blocks = load_latent_features(
        os.path.join(codec, chunk), zpool='max', spatial=spatial,
        per_scale=True, model=model, epoch=epoch, device=device, zwin=zwin)
    np.savez(cache, stems=np.array(stems), fps=fps, blocks=np.array(blocks))
    return stems, fps, blocks


def read_mid_slice(np0):
    """(slice_yx, zmid) — the chunk's middle slice, full resolution, (y, x)."""
    za, zb = np0['z_range']
    arr = open_zarr(np0['source'], np0['zarr_level'])
    zmid = (za + zb) // 2
    return np.asarray(arr[zmid].read().result()).T.astype(np.float32), zmid


def gate_stats(sl, stems, patch):
    """Per-patch (std, mean) of the mid slice; gate = std > k * noise floor."""
    stds, means = [], []
    for st in stems:
        r, c = int(st[:3]), int(st[3:6])
        reg = sl[r * patch:(r + 1) * patch, c * patch:(c + 1) * patch]
        stds.append(float(reg.std()))
        means.append(float(reg.mean()))
    return np.array(stds), np.array(means)


def select_blocks(which, blocks):
    if which == 'coarse':
        keep = blocks[:2]
    elif which == 'fine':
        keep = blocks[2:]
    else:
        keep = blocks
    cols = np.concatenate([np.arange(a, b) for a, b in keep])
    w = keep[0][1] - keep[0][0]
    return cols, [(k * w, (k + 1) * w) for k in range(len(keep))]


def legend_inset(fig, var1, var2):
    axl = fig.add_axes([0.905, 0.06, 0.075, 0.16])
    g = np.linspace(0, 1, 64)
    p1, p2 = np.meshgrid(g, g)                       # p2 rows: bottom -> top
    axl.imshow(hsv_to_rgb(np.dstack([0.72 * (1 - p1), np.full_like(p1, 0.85),
                                     0.35 + 0.6 * p2])),
               origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axl.set_xlabel(f'PC1 ({var1:.0%})', fontsize=8)
    axl.set_ylabel(f'PC2 ({var2:.0%})', fontsize=8)
    axl.set_xticks([]), axl.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--codec', default=DEFAULT_CODEC)
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help='base output dir; figures land in {out}/{name}/, the '
                         'patch matrix at {out}/{name}.npy, caches in {out}/')
    ap.add_argument('--name', default=None,
                    help='experiment name XXX (default zw{zwin}_{blocks}'
                         '[_k{gate_k}])')
    ap.add_argument('--chunks', default=None,
                    help='comma-separated chunk dir names (default: all z*/)')
    ap.add_argument('--zwin', type=int, default=3,
                    help='latent Z half-window around the displayed slice '
                         '(default 3 -> 7 planes); the value tags caches/figs')
    ap.add_argument('--gate_k', type=float, default=2.0,
                    help='keep patches with mid-slice std > k * noise floor '
                         '(floor = median std of the darkest 10%% of patches)')
    ap.add_argument('--blocks', choices=('coarse', 'all', 'fine'), default='coarse',
                    help='which residual scales feed the PCA (default coarse '
                         '= s0+s1, comparable to the old coarse_spectrum figs)')
    ap.add_argument('--style', choices=('boxes', 'contour'), default='boxes',
                    help='boxes = per-patch rectangles, PC1->hue + '
                         'PC2->lightness; contour = smoothed filled/line '
                         'contours of the PC1 field only (PC2 not encodable '
                         'in level sets), background stays visible. '
                         'contour appends _contour to the default name.')
    ap.add_argument('--spatial', type=int, default=8)
    ap.add_argument('--model', default=None)
    ap.add_argument('--epoch', type=int, default=None)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    name = args.name or (f'zw{args.zwin}_{args.blocks}'
                         + ('' if args.gate_k == 2.0 else f'_k{args.gate_k:g}')
                         + ('' if args.style == 'boxes' else '_contour'))
    fig_dir = os.path.join(args.out, name)
    os.makedirs(fig_dir, exist_ok=True)

    chunks = (args.chunks.split(',') if args.chunks else
              sorted(d for d in os.listdir(args.codec) if d.startswith('z') and
                     os.path.isdir(os.path.join(args.codec, d))))

    # ---- per-chunk features (cached) + mid-slice gate stats -----------------
    stems, chunk_of, fps, blocks = [], [], [], None
    slices, stds, means = {}, [], []
    for ci, ch in enumerate(chunks):
        st, fp, blocks = chunk_features(args.codec, ch, args.out, args.zwin,
                                        args.spatial, args.model, args.epoch,
                                        args.device)
        np0 = json.load(open(os.path.join(args.codec, ch, 'norm_params.json')))
        sl, zmid = read_mid_slice(np0)
        sd, mn = gate_stats(sl, st, np0['patch'])
        stems += st
        chunk_of += [ci] * len(st)
        fps.append(fp)
        stds.append(sd)
        means.append(mn)
        slices[ch] = (sl, zmid, np0['patch'])
        print(f'{ch}: {len(st)} patches, zmid={zmid}', flush=True)
    fps = np.concatenate(fps)
    stds, means = np.concatenate(stds), np.concatenate(means)
    chunk_of = np.array(chunk_of)

    # ---- raw-pixel content gate (per chunk noise floor) ---------------------
    is_c = np.zeros(len(stems), bool)
    for ci, ch in enumerate(chunks):
        m = chunk_of == ci
        dark = means[m] <= np.percentile(means[m], 10)
        floor = np.median(stds[m][dark])
        is_c[m] = stds[m] > args.gate_k * floor
        print(f'{ch}: gate kept {is_c[m].sum()}/{m.sum()} '
              f'(floor {floor:.1f})', flush=True)

    # ---- global PCA on the gated set ----------------------------------------
    cols, sub_blocks = select_blocks(args.blocks, blocks)
    F = balance_scales(fps[is_c][:, cols], sub_blocks)
    F = np.clip((F - F.mean(0)) / (F.std(0) + 1e-8), -4, 4)
    pca = PCA(20, random_state=0).fit(F)
    pcs = pca.transform(F)
    s3a, s3b = blocks[-1]
    fine = np.linalg.norm(fps[is_c][:, s3a:s3b], axis=1)
    pc1 = pcs[:, 0] * np.sign(np.corrcoef(pcs[:, 0], fine)[0, 1] or 1)
    pc2 = pcs[:, 1] * np.sign(np.corrcoef(pcs[:, 1], stds[is_c])[0, 1] or 1)
    pc3 = pcs[:, 2] * np.sign(np.corrcoef(pcs[:, 2], stds[is_c])[0, 1] or 1)
    v1, v2 = pca.explained_variance_ratio_[:2]
    print(f'{is_c.sum()} content patches | PC1 {v1:.0%} / PC2 {v2:.0%} of '
          f'variance | corr(PC1, fine)={np.corrcoef(pc1, fine)[0, 1]:+.2f}, '
          f'corr(PC1, raw std)={np.corrcoef(pc1, stds[is_c])[0, 1]:+.2f}',
          flush=True)

    def unit(p):
        lo, hi = np.percentile(p, 2), np.percentile(p, 98)
        return np.clip((p - lo) / (hi - lo + 1e-9), 0, 1)
    p1n, p2n = unit(pc1), unit(pc2)
    rgb = hsv_to_rgb(np.stack([0.72 * (1 - p1n), np.full_like(p1n, 0.85),
                               0.35 + 0.6 * p2n], -1))
    cst, cch = np.array(stems)[is_c], chunk_of[is_c]

    # ---- (z, x, y, pc1, pc2, pc3) patch matrix ------------------------------
    rows = []
    for i, st in enumerate(cst):
        _, zmid, patch = slices[chunks[cch[i]]]
        rows.append((zmid, (int(st[3:6]) + 0.5) * patch,
                     (int(st[:3]) + 0.5) * patch, pc1[i], pc2[i], pc3[i]))
    mat = np.array(rows, np.float32)
    np.save(os.path.join(args.out, f'{name}.npy'), mat)
    print(f'wrote {os.path.join(args.out, name)}.npy {mat.shape} '
          '(z, x, y, pc1, pc2, pc3)', flush=True)

    # ---- render -------------------------------------------------------------
    for ci, ch in enumerate(chunks):
        sl, zmid, patch = slices[ch]
        ds = max(1, max(sl.shape) // 3000)
        sld = sl[::ds, ::ds]
        p = patch / ds
        vlo, vhi = np.percentile(sld, 1), np.percentile(sld, 99.5)
        fig, ax = plt.subplots(figsize=(18, 18 * sld.shape[0] / sld.shape[1]))
        ax.imshow(sld, cmap='gray', vmin=vlo, vmax=vhi, interpolation='nearest')
        sel = np.where(cch == ci)[0]
        if args.style == 'boxes':
            for i in sel:
                st = cst[i]
                ax.add_patch(Rectangle((int(st[3:6]) * p, int(st[:3]) * p), p, p,
                                       facecolor=rgb[i], alpha=0.55,
                                       edgecolor='none', zorder=2))
            enc = 'PC1→hue, PC2→lightness'
        else:
            nr, nc = sl.shape[0] // patch, sl.shape[1] // patch
            grid = np.full((nr, nc), np.nan, np.float32)
            for i in sel:
                grid[int(cst[i][:3]), int(cst[i][3:6])] = p1n[i]
            w = np.isfinite(grid).astype(np.float32)
            g = gaussian_filter(np.nan_to_num(grid), 1.0)
            d = gaussian_filter(w, 1.0)
            field = np.where(d > 0.3, g / np.maximum(d, 1e-6), np.nan)
            gx = (np.arange(nc) + 0.5) * p
            gy = (np.arange(nr) + 0.5) * p
            lv = np.linspace(0, 1, 13)
            ax.contourf(gx, gy, field, levels=lv, cmap=PC1_CMAP, alpha=0.30,
                        zorder=2)
            ax.contour(gx, gy, field, levels=lv, cmap=PC1_CMAP,
                       linewidths=1.4, alpha=0.95, zorder=3)
            enc = 'PC1 contours (PC2 not shown)'
        ax.set_title(f'{ch} z={zmid} — {args.blocks} scales, zwin=±{args.zwin}, '
                     f'raw-std gate k={args.gate_k} | {enc}')
        ax.set_xlim(0, sld.shape[1]), ax.set_ylim(sld.shape[0], 0)
        ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        if args.style == 'boxes':
            legend_inset(fig, v1, v2)
        else:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            sm = ScalarMappable(Normalize(0, 1), PC1_CMAP)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=fig.add_axes([0.92, 0.06, 0.018, 0.2]))
            cb.set_label(f'PC1 ({v1:.0%} var), 2–98 pct normalised')
        png = os.path.join(fig_dir, f'{ch}.png')
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f'wrote {png}', flush=True)


if __name__ == '__main__':
    main()
