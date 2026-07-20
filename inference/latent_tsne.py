"""t-SNE map of stored codecs + anomaly flagging for materials-science 3D imaging.

    python inference/latent_tsne.py --codec /home/cheese/workspace/Output/skipU300/codec

CONTEXT: this is a materials-science 3D volume-imaging problem. The inputs are
anisotropic 3D scans of material samples (e.g. THX10 tomography), which the
IsoScope model encodes patch-by-patch into VQ codebook indices. Each {stem}.npz
codec here is one 3D sub-volume (a 256x256xZ patch) of a much larger material
sample. This script does an *unsupervised material-microstructure survey*: it
embeds every patch by its latent content so patches of similar microstructure
land together, and flags the unusual ones — the goal is to spot rare or defect
regions (voids, inclusions, cracks, dense/boundary structure) across a large
sample without any labels.

Gathers every {stem}.npz written by inference/inference_latent.py (or
encode_stack.py) in --codec, turns each material patch's codec into a feature
(see --feat below), then:
  - PCA -> t-SNE to 2D, scatter plot saved as {out_dir}/tsne{_feat}.png
    (nearby dots = similar microstructure)
  - IsolationForest on the PCA features flags anomalous patches (fraction
    set by --contamination) — candidate rare/defect microstructure regions,
    drawn in red and labeled on the plot
  - {out_dir}/anomalies{_feat}.csv lists the flagged patch names with their
    anomaly score (most anomalous first) and t-SNE coordinates
  - when all stems are AAABBBCCC patch indices (AAA=Y, BBB=X, CCC=Z spatial
    position of the patch inside the sample volume), a second figure
    {out_dir}/tsne_grid{_feat}.png colors the same embedding by that position:
    one RGB=(Y,X,Z) panel and one viridis panel per axis (well-mixed colors
    mean patches cluster by microstructure, not by where they sit in the sample)
  - with --thumbs <raw data dir>, a paper-style figure
    {out_dir}/tsne_thumbs{_feat}.png annotates ~--n_thumbs dots (spread by
    farthest-point sampling, anomalies included) with downsampled Z-MIP
    thumbnails of the corresponding material sub-volumes

Features (--feat): 'hist' (default) is the bag-of-codes histogram (per-scale
codebook usage, L1-normalized, concatenated); 'hist_std'/'tfidf'/'hist_bal'
reweight it (see transform_hist); 'latent' rebuilds the continuous quantized
latent from the stored indices and pools it (see load_latent_features). The
hist* features are pure CPU/sklearn; 'latent' loads the model (GPU).

WHAT WORKS: the '--feat latent' embedding (e.g. --zpool mean) is the one that
maps to real material microstructure — overlaying its IsolationForest anomalies
back onto a raw slice (inference/latent_tsne overlay, e.g.
anomalies_overlay_*.png) lands the red patches on genuine material features
(grain edges / voids / inclusions), whereas the bag-of-codes 'hist' embedding is
near-1D (PC1 ~0.93) and background-code dominated (empty/matrix regions of the
sample), so it separates only gross outliers. Prefer --feat latent for
microstructure-aware maps; the hist variants stay for cheap, model-free triage.

out_dir defaults to the codec folder's parent (the experiment dir).
"""

import argparse
import csv
import json
import os
import re
import sys
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_norm(chunk_dir):
    p = os.path.join(chunk_dir, 'norm_params.json')
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}


def gather_codecs(codec):
    """Resolve --codec to a list of (stem, npz_path, chunk_norm_params).

    A single chunk dir (npz files directly inside) -> 6-digit rrrccc stems, as
    before. A codec/ TREE (subdirs, each with npz — the z0000-0048 .. layout
    encode_stack writes) -> ONE combined set with 9-digit rrrcccZZZ stems, ZZZ
    the z-chunk ordinal (chunk dirs sorted = z order). The 9-digit form makes
    the 3D (Y=rrr, X=ccc, Z=ZZZ) grid coloring and per-chunk thumbnails work,
    so a whole-sample survey is one call on the codec/ root."""
    codec = os.path.abspath(codec.rstrip('/'))
    direct = sorted(glob(os.path.join(codec, '*.npz')))
    if direct:                                           # single chunk dir
        np0 = _load_norm(codec)
        return [(os.path.splitext(os.path.basename(f))[0], f, np0) for f in direct]
    chunk_dirs = sorted(d for d in glob(os.path.join(codec, '*'))
                        if os.path.isdir(d) and glob(os.path.join(d, '*.npz')))
    if not chunk_dirs:
        raise FileNotFoundError(f'No .npz codecs in {codec} or its subdirs')
    recs = []
    for zc, cd in enumerate(chunk_dirs):
        np0 = _load_norm(cd)
        for f in sorted(glob(os.path.join(cd, '*.npz'))):
            rc = os.path.splitext(os.path.basename(f))[0]        # rrrccc
            recs.append((f'{rc}{zc:03d}', f, np0))               # rrrcccZZZ
    return recs


def load_features(codec_dir):
    """(stems, features, blocks): per-patch concatenated per-scale code
    histograms. blocks = the (start, end) column range of each scale in the
    concatenated feature (so a caller can reweight scales independently).
    codec_dir may be a single chunk dir or a codec/ tree (see gather_codecs)."""
    recs = gather_codecs(codec_dir)
    stems, per_scale = [], None
    for stem, f, _ in recs:
        z = np.load(f)
        scales = sorted((k for k in z.files if k.startswith('scale_')),
                        key=lambda k: int(k.split('_')[1]))
        if per_scale is None:
            per_scale = [[] for _ in scales]
        stems.append(stem)
        for k, key in enumerate(scales):
            per_scale[k].append(z[key].ravel())
    feats, blocks, start = [], [], 0
    for idx_lists in per_scale:
        n_codes = int(max(a.max() for a in idx_lists)) + 1
        hist = np.stack([np.bincount(a, minlength=n_codes) for a in idx_lists]).astype(np.float64)
        feats.append(hist / hist.sum(axis=1, keepdims=True))  # L1 per scale
        blocks.append((start, start + n_codes))
        start += n_codes
    return stems, np.concatenate(feats, axis=1), blocks


def balance_scales(F, blocks):
    """Rescale each scale block to equal total variance, so all scales weigh
    equally in the (Euclidean/PCA) embedding — unlike L1-per-scale (coarse,
    spiky scales dominate the magnitude) or per-column z-score (the scale with
    the most codes dominates)."""
    F = F.copy()
    for s, e in blocks:
        v = F[:, s:e].var(0).sum()
        if v > 0:
            F[:, s:e] /= np.sqrt(v)
    return F


def transform_hist(F, feat, blocks=None):
    """Reweight the bag-of-codes histogram to counter its ~1D, background-code
    dominance (raw PC1 ~0.93 on sample0): 'hist_std' z-scores each code column,
    'tfidf' down-weights codes common across patches, 'hist_bal' rescales each
    scale block to equal variance (all 4 scales equally important).
    'hist' = untouched."""
    if feat == 'hist':
        return F
    if feat == 'hist_std':
        return (F - F.mean(0)) / (F.std(0) + 1e-9)
    if feat == 'tfidf':
        idf = np.log(len(F) / ((F > 0).sum(0) + 1))
        return F * idf
    if feat == 'hist_bal':
        return balance_scales(F, blocks)
    raise ValueError(f"unknown hist transform '{feat}'")


def load_latent_features(codec_dir, zpool='max', spatial=8, per_scale=False,
                         model=None, epoch=None, device=None, half=False):
    """Continuous latent features: rebuild each patch's quantized latent from
    the stored indices (Engine.latents_from_indices, no re-encode), pool over Z
    (max mirrors the MIP thumbnail; mean is smoother), then adaptive-pool the
    H x W plane to a spatial x spatial grid and flatten. Unlike the bag-of-codes
    histogram this lives in a metric space (similar textures -> near vectors)
    and keeps coarse spatial arrangement.

    per_scale=False sums the scales into one latent (one feature block);
    per_scale=True pools each scale separately and concatenates them, returning
    per-scale blocks so the caller can weigh scales equally. Returns
    (stems, features, blocks)."""
    import torch
    import torch.nn.functional as Fnn
    from inference import Engine
    recs = gather_codecs(codec_dir)
    if model is None:
        model = recs[0][2].get('model')
        if model is None:
            raise ValueError(f'no norm_params.json model in {codec_dir}; pass model=')
    eng = Engine.from_registry(model, epoch=epoch, device=device)
    if half:
        eng.gan.half()
    zp = (lambda t: t.amax(-1)) if zpool == 'max' else (lambda t: t.mean(-1))

    def pool(vol):                                        # (1, C, H, W, Z) -> flat (C*s*s,)
        p = Fnn.adaptive_avg_pool2d(zp(vol.float())[0], (spatial, spatial))
        return p.flatten().cpu().numpy()

    stems, feats, n_scales = [], [], None
    for stem, f, _ in recs:
        z = np.load(f)
        scales = sorted((k for k in z.files if k.startswith('scale_')),
                        key=lambda k: int(k.split('_')[1]))
        idx = [torch.from_numpy(z[k].astype(np.int64)).to(eng.device) for k in scales]
        lat = eng.latents_from_indices(idx)              # list of (1, C, H, W, Z)
        n_scales = len(lat)
        feats.append(np.concatenate([pool(L) for L in lat]) if per_scale
                     else pool(sum(lat)))
        stems.append(stem)
    feats = np.stack(feats)
    if per_scale:
        d = feats.shape[1] // n_scales                   # each scale block same width
        blocks = [(k * d, (k + 1) * d) for k in range(n_scales)]
    else:
        blocks = [(0, feats.shape[1])]
    return stems, feats, blocks


def build_features(args):
    """(stems, feature_matrix, label) for the requested --feat."""
    if args.feat == 'latent':
        stems, feats, blocks = load_latent_features(
            args.codec, zpool=args.zpool, spatial=args.spatial,
            per_scale=args.balance_scales, model=args.model, epoch=args.epoch,
            device=args.device, half=args.half)
        if args.balance_scales:
            return stems, balance_scales(feats, blocks), f'latent_{args.zpool}_bal'
        return stems, feats, f'latent_{args.zpool}'
    stems, hist, blocks = load_features(args.codec)
    return stems, transform_hist(hist, args.feat, blocks), args.feat


def build_mip_thumb(args, stem_np0):
    """A mip_thumb(stem) -> 2D [0,1] MIP thumbnail, or None if none can be
    sourced. Priority: an explicit --thumbs dir of raw {stem}.tif volumes;
    otherwise each patch's OWN recorded zarr source (source/zarr_level/z_range/
    patch in its chunk's norm_params.json — stem_np0 maps stem -> that dict), so
    a multi-chunk survey pulls each thumbnail from the right z-slab. Returns None
    (with a note) when neither is available (a TIFF-sourced codec, non-rrrccc[ZZZ]
    stems, or tensorstore missing) so the run still produces the other figures."""
    def finish(m):
        step = max(1, m.shape[0] // args.thumb_px)
        m = m[::step, ::step].astype(np.float32)
        lo, hi = np.percentile(m, 1), np.percentile(m, 99.5)
        return np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)

    if args.thumbs:
        import tifffile as tiff

        def mip_thumb(stem):
            v = tiff.imread(os.path.join(args.thumbs, stem + '.tif'))
            return finish(v.max(axis=0))                          # (Z,Y,X) -> (Y,X)
        return mip_thumb

    sample = next(iter(stem_np0.values()), {})
    if 'zarr_level' not in sample or not all(re.fullmatch(r'\d{6}|\d{9}', s) for s in stem_np0):
        print(f"thumbnails: need a zarr-sourced codec with rrrccc[ZZZ] stems "
              f"(source={sample.get('source')}); pass --thumbs a raw-tif dir. Skipping.")
        return None
    try:
        from inference.zarr_io import open_zarr
    except Exception as e:                                         # tensorstore/env
        print(f'thumbnails: cannot import zarr reader ({str(e).splitlines()[0]}); skipping')
        return None
    cache = {}

    def arr_for(np0):                              # one open store per (source, level)
        k = (np0['source'], np0['zarr_level'])
        if k not in cache:
            cache[k] = open_zarr(*k)
        return cache[k]

    def mip_thumb(stem):                                          # zarr axes (z, x, y)
        np0 = stem_np0[stem]
        patch, (za, zb) = np0['patch'], np0['z_range']
        r, c = int(stem[:3]), int(stem[3:6])                     # rrrccc or rrrcccZZZ
        v = arr_for(np0)[za:zb, c * patch:(c + 1) * patch,
                         r * patch:(r + 1) * patch].read().result()
        return finish(v.max(axis=0).T)                           # (Zc,x,y) -> (x,y) MIP -> (Y,X)
    return mip_thumb


def run(codec, out_dir=None, contamination=0.05, reducer='tsne', perplexity=30.0,
        n_neighbors=15, min_dist=0.1, umap_trim=0.0, pca=50, seed=0, thumbs=None,
        no_thumbs=False, n_thumbs=48, thumb_px=56, feat='hist', zpool='max', spatial=8,
        model=None, epoch=None, device=None, half=False, balance_scales=False):
    """The whole pipeline as a callable (used by encode_stack.py --tsne)."""
    args = argparse.Namespace(codec=codec, out_dir=out_dir, contamination=contamination,
                              reducer=reducer, perplexity=perplexity, n_neighbors=n_neighbors,
                              min_dist=min_dist, umap_trim=umap_trim, pca=pca, seed=seed,
                              thumbs=thumbs, no_thumbs=no_thumbs, n_thumbs=n_thumbs,
                              thumb_px=thumb_px, feat=feat, zpool=zpool, spatial=spatial,
                              model=model, epoch=epoch, device=device, half=half,
                              balance_scales=balance_scales)
    _run(args)


def main():
    parser = argparse.ArgumentParser(description='t-SNE + anomaly map of stored codecs')
    parser.add_argument('--codec', required=True,
                        help='A single chunk dir of {rrrccc}.npz, OR a codec/ tree of '
                             'z*/ chunk dirs -> one combined whole-sample map (3D grid '
                             'coloring by Y/X/Z-chunk)')
    parser.add_argument('--out_dir', default=None,
                        help="Output dir for tsne.png / anomalies.csv (default: --codec's parent)")
    parser.add_argument('--contamination', type=float, default=0.05,
                        help='Expected anomaly fraction for IsolationForest (default 0.05)')
    parser.add_argument('--reducer', choices=('tsne', 'umap'), default='tsne',
                        help='2D embedding method (default tsne). Outputs are prefixed by '
                             'the method (tsne_*.png / umap_*.png); anomaly flags are '
                             'method-independent (IsolationForest on the PCA features).')
    parser.add_argument('--perplexity', type=float, default=30.0, help='t-SNE perplexity (default 30)')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors (default 15; higher = more global structure)')
    parser.add_argument('--min_dist', type=float, default=0.1,
                        help='UMAP min_dist (default 0.1; lower = tighter clusters)')
    parser.add_argument('--umap_trim', type=float, default=0.0,
                        help='UMAP only: drop this fraction of points farthest from the '
                             '(median) centroid of the 2D embedding before plotting, so a '
                             'sparse noise halo stops dominating the view (default 0 = off; '
                             'e.g. 0.03 drops the farthest 3%%). Trims all UMAP figures + CSV.')
    parser.add_argument('--pca', type=int, default=50,
                        help='PCA dims before the reducer / IsolationForest (default 50)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--thumbs', default=None, metavar='DATA_DIR',
                        help='Override the thumbnail source with a directory of raw '
                             '{stem}.tif volumes. Default (no flag): thumbnails are drawn '
                             "from the codec's own recorded zarr source (norm_params.json).")
    parser.add_argument('--no_thumbs', action='store_true',
                        help='Skip the thumbnail-annotated figure tsne_thumbs.png '
                             '(on by default when a zarr-sourced codec is available)')
    parser.add_argument('--n_thumbs', type=int, default=48,
                        help='How many dots get a thumbnail (default 48)')
    parser.add_argument('--thumb_px', type=int, default=56,
                        help='Thumbnail size in pixels after downsampling the MIP (default 56)')
    parser.add_argument('--feat', choices=('hist', 'hist_std', 'tfidf', 'hist_bal', 'latent'),
                        default='hist',
                        help='Feature: hist = bag-of-codes histogram (default); hist_std = '
                             'column-standardized; tfidf = down-weight common codes; hist_bal = '
                             'equal variance per scale (all 4 scales weigh equally); latent = '
                             'continuous Z-pooled latent (needs the model). Non-hist writes '
                             'suffixed outputs (tsne_<feat>.png).')
    parser.add_argument('--balance_scales', action='store_true',
                        help='latent feat: pool each scale separately and rescale to equal '
                             'variance (all scales weigh equally) instead of summing them')
    parser.add_argument('--zpool', choices=('mean', 'max'), default='max',
                        help='latent feat: pool the latent over Z (default max, mirrors the MIP thumbnail)')
    parser.add_argument('--spatial', type=int, default=8,
                        help='latent feat: adaptive-pool the H x W plane to this grid (default 8)')
    parser.add_argument('--model', default=None,
                        help="latent feat: model name (default: codec's norm_params.json)")
    parser.add_argument('--epoch', type=int, default=None, help='latent feat: model epoch override')
    parser.add_argument('--device', default=None, help='latent feat: cuda / cpu')
    parser.add_argument('--half', action='store_true', help='latent feat: run the model in fp16')
    _run(parser.parse_args())


def _run(args):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest

    method = args.reducer                            # 'tsne' or 'umap'
    mlabel = {'tsne': 't-SNE', 'umap': 'UMAP'}[method]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.codec).rstrip('/'))
    os.makedirs(out_dir, exist_ok=True)

    stems, feats, label = build_features(args)
    suffix = '' if label == 'hist' else '_' + label      # keep legacy {method}.png for hist
    n = len(stems)
    print(f'{n} codecs from {args.codec}, feat={label}, dim {feats.shape[1]}, reducer={method}')

    n_pca = min(args.pca, n - 1, feats.shape[1])
    X = PCA(n_components=n_pca, random_state=args.seed).fit_transform(feats)

    # anomaly flagging is on the PCA features, independent of the 2D reducer
    iso = IsolationForest(contamination=args.contamination, random_state=args.seed)
    flags = iso.fit_predict(X) == -1                 # True = anomaly
    scores = iso.score_samples(X)                    # lower = more anomalous

    if method == 'umap':
        import umap
        emb = umap.UMAP(n_components=2, n_neighbors=min(args.n_neighbors, n - 1),
                        min_dist=args.min_dist, init='pca' if n_pca >= 2 else 'random',
                        random_state=args.seed).fit_transform(X)
    else:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, perplexity=min(args.perplexity, (n - 1) / 3),
                   init='pca', learning_rate='auto',
                   random_state=args.seed).fit_transform(X)

    # UMAP lays a dense material core inside a sparse noise halo; the halo
    # dominates the plot extent (and the farthest-point thumbnail sampler).
    # Drop the farthest --umap_trim fraction by distance to the (robust median)
    # centroid so the core fills the view. All downstream figures + CSV follow.
    if method == 'umap' and args.umap_trim > 0:
        center = np.median(emb, axis=0)
        dist = np.linalg.norm(emb - center, axis=1)
        cutoff = np.quantile(dist, 1 - args.umap_trim)
        keep = dist <= cutoff
        print(f'umap_trim: dropping {int((~keep).sum())} of {len(keep)} points beyond '
              f'distance {cutoff:.2f} to centroid (farthest {args.umap_trim * 100:.0f}%)')
        emb, stems = emb[keep], [s for s, k in zip(stems, keep) if k]
        flags, scores, n = flags[keep], scores[keep], int(keep.sum())

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(emb[~flags, 0], emb[~flags, 1], s=18, c='#4878cf', alpha=0.7,
               linewidths=0, label=f'normal ({(~flags).sum()})')
    ax.scatter(emb[flags, 0], emb[flags, 1], s=42, c='red', marker='o',
               edgecolors='darkred', linewidths=0.5, label=f'anomaly ({flags.sum()})')
    for i in np.where(flags)[0]:
        ax.annotate(stems[i], (emb[i, 0], emb[i, 1]), fontsize=6, color='darkred',
                    xytext=(3, 3), textcoords='offset points')
    ax.set_title(f'{mlabel} [{label}] — {os.path.basename(out_dir)} '
                 f'(n={n}, contamination={args.contamination})')
    ax.legend(loc='best')
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    png = os.path.join(out_dir, f'{method}{suffix}.png')
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
        fig.suptitle(f'{mlabel} [{label}] colored by patch grid position — '
                     f'{os.path.basename(out_dir)}', y=0.995)
        fig.tight_layout()
        grid_png = os.path.join(out_dir, f'{method}_grid{suffix}.png')
        fig.savefig(grid_png, dpi=200)
        plt.close(fig)
        print(f'grid-colored plot -> {grid_png}')

    stem_np0 = {stem: np0 for stem, _, np0 in gather_codecs(args.codec)}
    thumb_fn = None if args.no_thumbs else build_mip_thumb(args, stem_np0)
    if thumb_fn is not None:
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
                OffsetImage(thumb_fn(stems[i]), cmap='gray', zoom=1.0),
                (emb[i, 0], emb[i, 1]), frameon=True, pad=0.15,
                bboxprops=dict(edgecolor='red' if flags[i] else '#666666',
                               linewidth=1.6 if flags[i] else 0.8))
            ax.add_artist(ab)
        ax.set_title(f'{mlabel} [{label}] — {os.path.basename(out_dir)} '
                     f'(Z-MIP thumbnails on {n_th} farthest-point dots)')
        ax.legend(loc='lower right')
        ax.set_xticks([]), ax.set_yticks([])
        m = 0.06 * (emb.max(0) - emb.min(0))          # margin so border thumbs stay inside
        ax.set_xlim(emb[:, 0].min() - m[0], emb[:, 0].max() + m[0])
        ax.set_ylim(emb[:, 1].min() - m[1], emb[:, 1].max() + m[1])
        fig.tight_layout()
        thumbs_png = os.path.join(out_dir, f'{method}_thumbs{suffix}.png')
        fig.savefig(thumbs_png, dpi=200)
        plt.close(fig)
        print(f'thumbnail plot -> {thumbs_png}')

    order = np.argsort(scores)                       # most anomalous first
    csv_path = os.path.join(out_dir, f'anomalies_{method}{suffix}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'anomaly_score', 'emb_x', 'emb_y'])
        for i in order:
            if flags[i]:
                w.writerow([stems[i], f'{scores[i]:.4f}', f'{emb[i, 0]:.2f}', f'{emb[i, 1]:.2f}'])

    print(f'{flags.sum()} anomalies -> {csv_path}')
    print(f'plot -> {png}')
    for i in order[:min(10, flags.sum())]:
        print(f'  {stems[i]}  score {scores[i]:.4f}')


if __name__ == '__main__':
    main()
