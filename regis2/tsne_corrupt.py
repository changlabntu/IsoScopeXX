"""t-SNE microstructure map with 10% synthetically misaligned patches.

    python regis2/tsne_corrupt.py

Repeats the skipU300 latent-t-SNE survey with the experiment's best settings
(--feat latent --zpool mean: the structure-aware embedding; the bag-of-codes
histogram is near-1D and background-dominated), except a random --frac of the
512 patches is corrupted by the current regis2 corruption model (walk_drift:
similarity random walk, rel rot ~U(+-0.5 deg) / shift ~U(+-3 px) / scale
~U(1+-0.005), composed with the smooth sinusoidal drift +-10 px / +-1 deg)
BEFORE encoding. Corrupted patches are re-encoded from the raw roiAdsp4 tifs
(same 11g normalization as the codec, per codec meta); clean patches use the
stored indices (Engine.latents_from_indices) — identical latent space.

The t-SNE plot marks corrupted patches GREEN (IsolationForest anomalies red,
as before) to show whether slice misalignment separates in the microstructure
embedding. Also reports precision@k: how many of the k corrupted patches rank
in the k most-anomalous scores.

Outputs under --out_dir: tsne_corrupt.png, corrupt_report.csv.
"""

import argparse
import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fnn  # noqa: E402
from glob import glob  # noqa: E402

from registration import affine, enhance, features  # noqa: E402
from registration.perturb import sample_transforms  # noqa: E402
from regis2.perturb_options import apply, chunk_transforms, drift_transforms  # noqa: E402

DEFAULT_CODEC = '/home/cheese/workspace/Output/skipU300/codec'
DEFAULT_RAW = '/home/cheese/workspace/Data/thx10/roiAdsp4'
DEFAULT_OUT = '/home/cheese/workspace/Output/regis2/tsne_corrupt10'


def pool_latent(vol, spatial, zpool='mean'):
    """(1, C, H, W, Z) latent -> mean/max over Z, adaptive-pool to spatial^2,
    flat. Mirrors inference/latent_tsne.py load_latent_features."""
    zp = vol.float().amax(-1) if zpool == 'max' else vol.float().mean(-1)
    p = Fnn.adaptive_avg_pool2d(zp[0], (spatial, spatial))
    return p.flatten().cpu().numpy()


def jump_feature(vol):
    """(1, C, H, W, Z) latent -> 6 slice-to-slice 'jump' statistics: the
    phase-correlation shift |t| between adjacent latent planes (register.py's
    coarse measurement, latent px) and the adjacent-plane NCC. Misalignment is
    an INTER-slice property, so unlike the Z-pooled content feature these
    stats carry it directly."""
    from registration.evaluate import ncc
    from registration.register import phase_corr_shift
    planes = vol[0].permute(3, 0, 1, 2).float()          # (Z, C, h, w)
    ts, nccs = [], []
    prev = planes[0].cpu().numpy().ravel()
    for z in range(planes.shape[0] - 1):
        dx, dy = phase_corr_shift(planes[z + 1], planes[z])
        ts.append(float(np.hypot(dx, dy)))
        cur = planes[z + 1].cpu().numpy().ravel()
        nccs.append(ncc(prev, cur))
        prev = cur
    ts, nccs = np.asarray(ts), np.asarray(nccs)
    return np.array([ts.mean(), np.median(ts), ts.max(), ts.std(),
                     nccs.mean(), nccs.min()])


def main():
    parser = argparse.ArgumentParser(description='latent t-SNE with corrupted patches')
    parser.add_argument('--codec', default=DEFAULT_CODEC)
    parser.add_argument('--raw', default=DEFAULT_RAW, help='Raw (Z, Y, X) [-1, 1] tifs')
    parser.add_argument('--out_dir', default=DEFAULT_OUT)
    parser.add_argument('--frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--corrupt', choices=('walk_drift', 'drift', 'chunk'),
                        default='walk_drift',
                        help='drift = smooth sinusoid only (+-10 px / +-1 deg '
                             'at --drift_scale 1, no random walk): tiny '
                             'adjacent-slice jumps (~2 px max) — the jump '
                             "feature's sensitivity stress test. Non-default "
                             'suffixes the outputs.')
    parser.add_argument('--drift_scale', type=float, default=1.0,
                        help='drift magnitude knob: amp = 10*scale px, rot = '
                             '1*scale deg (drift mode only; != 1 goes into the '
                             'output suffix)')
    parser.add_argument('--feat', choices=('latent_mean', 'jump', 'both'),
                        default='latent_mean',
                        help='latent_mean = Z-pooled content (alignment-'
                             'invariant); jump = adjacent-slice phase-corr |t| '
                             '+ NCC stats; both = concat (blocks variance-'
                             'balanced). Non-default suffixes the outputs.')
    parser.add_argument('--zpool', choices=('mean', 'max'), default='mean',
                        help='Z pooling of the latent_mean content feature '
                             '(latent_tsne --zpool). Non-default suffixes the '
                             'outputs.')
    parser.add_argument('--spatial', type=int, default=8)
    parser.add_argument('--pca', type=int, default=50)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--contamination', type=float, default=0.05)
    parser.add_argument('--model', default='skipU')
    parser.add_argument('--thumbs', action='store_true',
                        help='Also write a Z-MIP-thumbnail figure (latent_tsne '
                             'style; corrupted patches show their CORRUPTED '
                             'volume, green border; flagged = red border)')
    parser.add_argument('--n_thumbs', type=int, default=24)
    parser.add_argument('--thumb_px', type=int, default=56)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob(os.path.join(args.codec, '*.npz')))
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    n = len(stems)
    rng = np.random.default_rng(args.seed)
    corrupt_idx = np.sort(rng.choice(n, size=int(round(n * args.frac)), replace=False))
    corrupt = np.zeros(n, bool)
    corrupt[corrupt_idx] = True
    print(f'{n} patches, corrupting {corrupt.sum()} ({args.frac:.0%}) with {args.corrupt}')

    eng = features.get_engine(args.model)
    for nm in eng.gan.netg_names:
        getattr(eng.gan, nm).eval()
    device = eng.device

    def featurize(vol):
        parts = []
        if args.feat in ('latent_mean', 'both'):
            parts.append(pool_latent(vol, args.spatial, args.zpool))
        if args.feat in ('jump', 'both'):
            parts.append(jump_feature(vol))
        return np.concatenate(parts)

    feats, corrupt_mips = [], {}
    with torch.inference_mode():
        for i, (f, stem) in enumerate(zip(files, stems)):
            if corrupt[i]:
                vol = tifffile.imread(os.path.join(args.raw, stem + '.tif')).astype(np.float32)
                Z = vol.shape[0]
                if args.corrupt == 'drift':
                    gt = drift_transforms(Z, amp=10.0 * args.drift_scale,
                                          rot=1.0 * args.drift_scale)
                elif args.corrupt == 'chunk':
                    gt, _ = chunk_transforms(Z, seed=1000 + i)
                else:
                    _, walk = sample_transforms(Z, 0.5, 3.0, 0.005, 1000 + i)
                    gt = np.stack([affine.compose(d, w) for d, w
                                   in zip(drift_transforms(Z), np.stack(walk))])
                corrupted_vol = apply(vol, gt, device)
                if args.thumbs:
                    corrupt_mips[i] = corrupted_vol.max(axis=0)
                sl = enhance.encode_full(eng, corrupted_vol, zbatch=8)
                feats.append(featurize(sum(sl).to(device)))
            else:
                z = np.load(f)
                keys = sorted((k for k in z.files if k.startswith('scale_')),
                              key=lambda k: int(k.split('_')[1]))
                idx = [torch.from_numpy(z[k].astype(np.int64)).to(device) for k in keys]
                feats.append(featurize(sum(eng.latents_from_indices(idx))))
            if i % 64 == 0:
                print(f'  {i}/{n}')
    feats = np.stack(feats)
    if args.feat in ('jump', 'both'):                    # z-score the jump block
        s, e = feats.shape[1] - 6, feats.shape[1]
        feats[:, s:e] = ((feats[:, s:e] - feats[:, s:e].mean(0))
                         / (feats[:, s:e].std(0) + 1e-9))
        if args.feat == 'both':
            from inference.latent_tsne import balance_scales
            feats = balance_scales(feats, [(0, s), (s, e)])
    print(f'features: {feats.shape} ({args.feat}, spatial {args.spatial})')

    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.manifold import TSNE
    X = PCA(n_components=min(args.pca, n - 1, feats.shape[1]),
            random_state=args.seed).fit_transform(feats)
    iso = IsolationForest(contamination=args.contamination, random_state=args.seed)
    flags = iso.fit_predict(X) == -1
    scores = iso.score_samples(X)
    emb = TSNE(n_components=2, perplexity=min(args.perplexity, (n - 1) / 3),
               init='pca', learning_rate='auto', random_state=args.seed).fit_transform(X)

    k = int(corrupt.sum())
    topk = np.argsort(scores)[:k]
    hits = corrupt[topk].sum()
    print(f'IsolationForest flags {flags.sum()} at contamination {args.contamination}; '
          f'{corrupt[flags].sum()}/{k} corrupted flagged')
    print(f'precision@{k} (corrupted in {k} most-anomalous): {hits}/{k} = {hits / k:.0%}')

    fig, ax = plt.subplots(figsize=(9, 8))
    clean = ~corrupt
    ax.scatter(emb[clean & ~flags, 0], emb[clean & ~flags, 1], s=18, c='#4878cf',
               alpha=0.7, linewidths=0, label=f'clean ({(clean & ~flags).sum()})')
    ax.scatter(emb[clean & flags, 0], emb[clean & flags, 1], s=42, c='red',
               edgecolors='darkred', linewidths=0.5,
               label=f'clean, flagged ({(clean & flags).sum()})')
    ax.scatter(emb[corrupt, 0], emb[corrupt, 1], s=46, c='limegreen', marker='o',
               edgecolors='darkgreen', linewidths=0.8,
               label=f'corrupted ({corrupt.sum()})')
    ax.scatter(emb[corrupt & flags, 0], emb[corrupt & flags, 1], s=90,
               facecolors='none', edgecolors='red', linewidths=1.4,
               label=f'corrupted, flagged ({(corrupt & flags).sum()})')
    corrupt_label = args.corrupt
    if args.corrupt == 'drift' and args.drift_scale != 1.0:
        corrupt_label += f'{args.drift_scale:g}'
    feat_label = args.feat + ('' if args.zpool == 'mean' else f', z{args.zpool}')
    suffix = ('' if args.feat == 'latent_mean' else '_' + args.feat) \
             + ('' if args.zpool == 'mean' else '_z' + args.zpool) \
             + ('' if args.corrupt == 'walk_drift' else '_' + corrupt_label)
    ax.set_title(f't-SNE [{feat_label}] — skipU300, {args.frac:.0%} '
                 f'{corrupt_label}-corrupted (green)\nprecision@{k}: {hits}/{k}')
    ax.legend(loc='best')
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    png = os.path.join(args.out_dir, f'tsne_corrupt{suffix}.png')
    fig.savefig(png, dpi=200)
    print(f'-> {png}')

    if args.thumbs:
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        def thumb(i):
            """Z-MIP of the volume as embedded (corrupted stack for greens)."""
            m = corrupt_mips[i] if i in corrupt_mips else \
                tifffile.imread(os.path.join(args.raw, stems[i] + '.tif')).max(axis=0)
            m = m.astype(np.float32)
            step = max(1, m.shape[0] // args.thumb_px)
            m = m[::step, ::step]
            lo, hi = np.percentile(m, 1), np.percentile(m, 99.5)
            return np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)

        # farthest-point sampling seeded at the most anomalous point, then make
        # sure the corrupted population is represented
        n_th = min(args.n_thumbs, n)
        chosen = [int(np.argmin(scores))]
        d = np.linalg.norm(emb - emb[chosen[0]], axis=1)
        while len(chosen) < n_th:
            nxt = int(np.argmax(d))
            chosen.append(nxt)
            d = np.minimum(d, np.linalg.norm(emb - emb[nxt], axis=1))
        want = max(1, n_th // 4)
        extra = [i for i in np.argsort(scores) if corrupt[i] and i not in chosen]
        chosen += extra[:max(0, want - sum(corrupt[i] for i in chosen))]

        fig, ax = plt.subplots(figsize=(13, 11))
        ax.scatter(emb[clean, 0], emb[clean, 1], s=14, c='#9ab5e0', alpha=0.6,
                   linewidths=0, label=f'clean ({clean.sum()})')
        ax.scatter(emb[corrupt, 0], emb[corrupt, 1], s=26, c='limegreen',
                   alpha=0.9, linewidths=0, label=f'corrupted ({corrupt.sum()})')
        for i in chosen:
            edge = ('red' if flags[i] else
                    'green' if corrupt[i] else '#666666')
            ab = AnnotationBbox(
                OffsetImage(thumb(i), cmap='gray', zoom=1.0),
                (emb[i, 0], emb[i, 1]), frameon=True, pad=0.15,
                bboxprops=dict(edgecolor=edge,
                               linewidth=2.0 if corrupt[i] or flags[i] else 0.8))
            ax.add_artist(ab)
        ax.set_title(f't-SNE [{feat_label}] — {corrupt_label} corruption, Z-MIP '
                     'thumbnails (green border = corrupted, red = flagged)')
        ax.legend(loc='lower right')
        ax.set_xticks([]), ax.set_yticks([])
        m = 0.06 * (emb.max(0) - emb.min(0))
        ax.set_xlim(emb[:, 0].min() - m[0], emb[:, 0].max() + m[0])
        ax.set_ylim(emb[:, 1].min() - m[1], emb[:, 1].max() + m[1])
        fig.tight_layout()
        thumbs_png = os.path.join(args.out_dir, f'tsne_corrupt_thumbs{suffix}.png')
        fig.savefig(thumbs_png, dpi=200)
        plt.close(fig)
        print(f'-> {thumbs_png}')

    with open(os.path.join(args.out_dir, f'corrupt_report{suffix}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'corrupted', 'flagged', 'anomaly_score', 'tsne_x', 'tsne_y'])
        for i in np.argsort(scores):
            w.writerow([stems[i], int(corrupt[i]), int(flags[i]),
                        f'{scores[i]:.4f}', f'{emb[i, 0]:.2f}', f'{emb[i, 1]:.2f}'])
    print(f"-> {os.path.join(args.out_dir, f'corrupt_report{suffix}.csv')}")


if __name__ == '__main__':
    main()
