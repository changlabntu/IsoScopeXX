"""t-SNE map of stored codecs + anomaly flagging.

    python inference/latent_tsne.py --codec /home/cheese/workspace/Output/skipU300/codec

Gathers every {stem}.npz written by inference/inference_latent.py in --codec,
turns each patch's codec into a bag-of-codes feature (per-scale histogram of
codebook usage, L1-normalized, concatenated over scales), then:
  - PCA -> t-SNE to 2D, scatter plot saved as {out_dir}/tsne.png
  - IsolationForest on the PCA features flags anomalous patches (fraction
    set by --contamination), drawn in red and labeled on the plot
  - {out_dir}/anomalies.csv lists the flagged patch names with their anomaly
    score (most anomalous first) and t-SNE coordinates
  - when all stems are AAABBBCCC patch indices (AAA=Y, BBB=X, CCC=Z), a
    second figure {out_dir}/tsne_grid.png colors the same embedding by grid
    position: one RGB=(Y,X,Z) panel and one viridis panel per axis
  - with --thumbs <raw data dir>, a paper-style figure {out_dir}/tsne_thumbs.png
    annotates ~--n_thumbs dots (spread by farthest-point sampling, anomalies
    included) with downsampled Z-MIP thumbnails of the corresponding volumes

out_dir defaults to the codec folder's parent (the experiment dir). Pure
CPU/sklearn — no model, no GPU.
"""

import argparse
import csv
import os
import re
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_features(codec_dir):
    """(stems, features): per-patch concatenated per-scale code histograms."""
    files = sorted(glob(os.path.join(codec_dir, '*.npz')))
    if not files:
        raise FileNotFoundError(f'No .npz codecs in {codec_dir}')
    stems, per_scale = [], None
    for f in files:
        z = np.load(f)
        scales = sorted(k for k in z.files if k.startswith('scale_'))
        if per_scale is None:
            per_scale = [[] for _ in scales]
        stems.append(os.path.splitext(os.path.basename(f))[0])
        for k, key in enumerate(scales):
            per_scale[k].append(z[key].ravel())
    feats = []
    for idx_lists in per_scale:
        n_codes = int(max(a.max() for a in idx_lists)) + 1
        hist = np.stack([np.bincount(a, minlength=n_codes) for a in idx_lists]).astype(np.float64)
        feats.append(hist / hist.sum(axis=1, keepdims=True))  # L1 per scale
    return stems, np.concatenate(feats, axis=1)


def run(codec, out_dir=None, contamination=0.05, perplexity=30.0, pca=50,
        seed=0, thumbs=None, n_thumbs=24, thumb_px=56):
    """The whole pipeline as a callable (used by encode_stack.py --tsne)."""
    args = argparse.Namespace(codec=codec, out_dir=out_dir, contamination=contamination,
                              perplexity=perplexity, pca=pca, seed=seed, thumbs=thumbs,
                              n_thumbs=n_thumbs, thumb_px=thumb_px)
    _run(args)


def main():
    parser = argparse.ArgumentParser(description='t-SNE + anomaly map of stored codecs')
    parser.add_argument('--codec', required=True, help='Folder of {stem}.npz codec files')
    parser.add_argument('--out_dir', default=None,
                        help="Output dir for tsne.png / anomalies.csv (default: --codec's parent)")
    parser.add_argument('--contamination', type=float, default=0.05,
                        help='Expected anomaly fraction for IsolationForest (default 0.05)')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--pca', type=int, default=50,
                        help='PCA dims before t-SNE / IsolationForest (default 50)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--thumbs', default=None, metavar='DATA_DIR',
                        help='Directory of the raw {stem}.tif volumes; enables the '
                             'thumbnail-annotated figure tsne_thumbs.png')
    parser.add_argument('--n_thumbs', type=int, default=24,
                        help='How many dots get a thumbnail (default 24)')
    parser.add_argument('--thumb_px', type=int, default=56,
                        help='Thumbnail size in pixels after downsampling the MIP (default 56)')
    _run(parser.parse_args())


def _run(args):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.manifold import TSNE

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.codec).rstrip('/'))
    os.makedirs(out_dir, exist_ok=True)

    stems, feats = load_features(args.codec)
    n = len(stems)
    print(f'{n} codecs from {args.codec}, feature dim {feats.shape[1]}')

    n_pca = min(args.pca, n - 1, feats.shape[1])
    X = PCA(n_components=n_pca, random_state=args.seed).fit_transform(feats)

    iso = IsolationForest(contamination=args.contamination, random_state=args.seed)
    flags = iso.fit_predict(X) == -1                 # True = anomaly
    scores = iso.score_samples(X)                    # lower = more anomalous

    emb = TSNE(n_components=2, perplexity=min(args.perplexity, (n - 1) / 3),
               init='pca', learning_rate='auto',
               random_state=args.seed).fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(emb[~flags, 0], emb[~flags, 1], s=18, c='#4878cf', alpha=0.7,
               linewidths=0, label=f'normal ({(~flags).sum()})')
    ax.scatter(emb[flags, 0], emb[flags, 1], s=42, c='red', marker='o',
               edgecolors='darkred', linewidths=0.5, label=f'anomaly ({flags.sum()})')
    for i in np.where(flags)[0]:
        ax.annotate(stems[i], (emb[i, 0], emb[i, 1]), fontsize=6, color='darkred',
                    xytext=(3, 3), textcoords='offset points')
    ax.set_title(f't-SNE of codec code-usage histograms — {os.path.basename(out_dir)} '
                 f'(n={n}, contamination={args.contamination})')
    ax.legend(loc='best')
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    png = os.path.join(out_dir, 'tsne.png')
    fig.savefig(png, dpi=200)
    plt.close(fig)

    # grid-coordinate coloring: 9-digit AAABBBCCC stems (AAA=Y, BBB=X, CCC=Z,
    # the roiAdsp4 3D patch grid) or 6-digit rrrccc stems (row/col slice grid
    # from inference/encode_stack.py)
    axis_names = None
    if all(re.fullmatch(r'\d{9}', s) for s in stems):
        axis_names = ['Y (AAA)', 'X (BBB)', 'Z (CCC)']
    elif all(re.fullmatch(r'\d{6}', s) for s in stems):
        axis_names = ['row (rrr)', 'col (ccc)']
    if axis_names:
        nd = len(axis_names)
        grid = np.array([[int(s[3 * k:3 * k + 3]) for k in range(nd)] for s in stems], float)
        frac = grid / np.maximum(grid.max(axis=0), 1)          # per-axis 0..1
        rgb = frac if nd == 3 else np.column_stack([frac, np.full(len(stems), 0.5)])
        fig, axes = plt.subplots(2, 2, figsize=(15, 13))
        hi = grid.max(axis=0).astype(int)
        panels = [(f'RGB = ({", ".join(a.split()[0] for a in axis_names)})', rgb, None, None)]
        panels += [(f'{axis_names[k]}, 0-{hi[k]}', frac[:, k], 'viridis', hi[k])
                   for k in range(nd)]
        for ax in axes.ravel()[len(panels):]:
            ax.axis('off')
        for ax, (title, c, cmap, top) in zip(axes.ravel(), panels):
            sc = ax.scatter(emb[:, 0], emb[:, 1], s=22, c=c, cmap=cmap,
                            alpha=0.85, linewidths=0)
            ax.scatter(emb[flags, 0], emb[flags, 1], s=70, facecolors='none',
                       edgecolors='red', linewidths=1.2,
                       label=f'anomaly ({flags.sum()})')
            if cmap is not None:
                cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
                cb.set_ticks([0, 1])
                cb.set_ticklabels(['0', str(top)])
            ax.set_title(title)
            ax.set_xticks([]), ax.set_yticks([])
        axes[0, 0].legend(loc='best')
        fig.suptitle(f't-SNE colored by patch grid position — {os.path.basename(out_dir)}',
                     y=0.995)
        fig.tight_layout()
        grid_png = os.path.join(out_dir, 'tsne_grid.png')
        fig.savefig(grid_png, dpi=200)
        plt.close(fig)
        print(f'grid-colored plot -> {grid_png}')

    if args.thumbs:
        import tifffile as tiff

        def mip_thumb(stem):
            """Z-MIP of the raw (Z, Y, X) volume, downsampled to ~thumb_px."""
            v = tiff.imread(os.path.join(args.thumbs, stem + '.tif'))
            m = v.max(axis=0).astype(np.float32)                  # (Y, X) MIP
            step = max(1, m.shape[0] // args.thumb_px)
            m = m[::step, ::step]
            lo, hi = np.percentile(m, 1), np.percentile(m, 99.5)
            return np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)

        # farthest-point sampling spreads the thumbnails over the map; seed it
        # with the most anomalous point so the outlier corner is represented
        n_th = min(args.n_thumbs, n)
        chosen = [int(np.argmin(scores))]
        d = np.linalg.norm(emb - emb[chosen[0]], axis=1)
        while len(chosen) < n_th:
            nxt = int(np.argmax(d))
            chosen.append(nxt)
            d = np.minimum(d, np.linalg.norm(emb - emb[nxt], axis=1))

        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        fig, ax = plt.subplots(figsize=(13, 11))
        ax.scatter(emb[~flags, 0], emb[~flags, 1], s=14, c='#9ab5e0', alpha=0.6,
                   linewidths=0, label=f'normal ({(~flags).sum()})')
        ax.scatter(emb[flags, 0], emb[flags, 1], s=26, c='red', alpha=0.9,
                   linewidths=0, label=f'anomaly ({flags.sum()})')
        for i in chosen:
            ab = AnnotationBbox(
                OffsetImage(mip_thumb(stems[i]), cmap='gray', zoom=1.0),
                (emb[i, 0], emb[i, 1]), frameon=True, pad=0.15,
                bboxprops=dict(edgecolor='red' if flags[i] else '#666666',
                               linewidth=1.6 if flags[i] else 0.8))
            ax.add_artist(ab)
        ax.set_title(f't-SNE of codec code-usage histograms — {os.path.basename(out_dir)} '
                     f'(Z-MIP thumbnails on {n_th} farthest-point dots)')
        ax.legend(loc='lower right')
        ax.set_xticks([]), ax.set_yticks([])
        m = 0.06 * (emb.max(0) - emb.min(0))          # margin so border thumbs stay inside
        ax.set_xlim(emb[:, 0].min() - m[0], emb[:, 0].max() + m[0])
        ax.set_ylim(emb[:, 1].min() - m[1], emb[:, 1].max() + m[1])
        fig.tight_layout()
        thumbs_png = os.path.join(out_dir, 'tsne_thumbs.png')
        fig.savefig(thumbs_png, dpi=200)
        plt.close(fig)
        print(f'thumbnail plot -> {thumbs_png}')

    order = np.argsort(scores)                       # most anomalous first
    csv_path = os.path.join(out_dir, 'anomalies.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'anomaly_score', 'tsne_x', 'tsne_y'])
        for i in order:
            if flags[i]:
                w.writerow([stems[i], f'{scores[i]:.4f}', f'{emb[i, 0]:.2f}', f'{emb[i, 1]:.2f}'])

    print(f'{flags.sum()} anomalies -> {csv_path}')
    print(f'plot -> {png}')
    for i in order[:min(10, flags.sum())]:
        print(f'  {stems[i]}  score {scores[i]:.4f}')


if __name__ == '__main__':
    main()
