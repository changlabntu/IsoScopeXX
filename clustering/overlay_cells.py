"""M4b — cell-level overlays for the harness winner (PLAN §6). Run only after
harness.py has picked a winning feature.

Two colorings of the same 48-px cell mosaic (8x finer than the old patch
boxes), both gated by raw cell contrast (cell-level version of the
latent_overlay gate):

  --mode umap     UMAP -> 3D, standardized, mapped through LAB
                  (L = 57.5 + 22.5*clip(e1/2), a/b = 40*clip(e2,e3 / 2), so
                  L stays in [35, 80]) -> RGB: similar material textures get
                  similar colors, distinct microstructures distinct hues.
  --mode anchors  linear probe (fit on all anchor cells) -> per-cell class
                  probabilities -> barycentric mix of fixed class hues; the
                  LOCO confusion matrix is printed for honesty.
  --mode clusters discrete k-means classes from a harness.py `cluster` run
                  (--clusters clusters_{...}.npz): one flat hue per named
                  cluster — the most readable "texture type" map.

--smooth N (umap mode) averages each cell's embedding over its (2N+1)^2 cell
neighbourhood within the chunk before coloring — kills the per-cell histogram
confetti (48-px histograms carry only ~250 tokens) while keeping regions.

Outputs (Output/clustering/): {name}.npy float32
(z, x, y, cell_row, cell_col, emb1, emb2, emb3[, class, p_class]) + one PNG
per chunk under {name}/. In clusters mode emb1 = cluster id, emb2 = distance
to centroid, class/p_class = named-class id / 1.

  python clustering/overlay_cells.py --feat Output/clustering/<winner>.npz \
      --mode umap --smooth 1 [--name ...] [--gate_k 2.0] [--chunks a,b]
  python clustering/overlay_cells.py --mode clusters \
      --clusters Output/clustering/clusters_<winner>_k8.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from clustering import common, render
from clustering.harness import (ANCHORS_DEFAULT, CLUSTER_HUES as
                                CLUSTER_HUES_FULL, probe_loco, read_anchors)

CLASS_HUES = np.array([[0.90, 0.10, 0.10], [0.10, 0.45, 0.95],
                       [0.10, 0.75, 0.25], [0.95, 0.75, 0.05],
                       [0.75, 0.15, 0.85], [0.05, 0.80, 0.80],
                       [0.95, 0.45, 0.05]])


def embed_umap(F, sample, seed=0):
    """(N, 3) UMAP embedding: fit on a subsample, transform the rest."""
    import umap
    rng = np.random.RandomState(seed)
    fit_idx = (rng.choice(len(F), min(sample, len(F)), replace=False)
               if len(F) > sample else np.arange(len(F)))
    um = umap.UMAP(n_components=3, random_state=seed).fit(F[fit_idx])
    if len(fit_idx) == len(F):
        return um.embedding_
    out = np.empty((len(F), 3), np.float32)
    out[fit_idx] = um.embedding_
    rest = np.setdiff1d(np.arange(len(F)), fit_idx)
    for a in range(0, len(rest), 50000):
        out[rest[a:a + 50000]] = um.transform(F[rest[a:a + 50000]])
        print(f'umap transform {min(a + 50000, len(rest))}/{len(rest)}',
              flush=True)
    return out


def lab_rgb(emb):
    """Standardized 3D embedding -> RGB via LAB (fixed spec from the plan)."""
    from skimage.color import lab2rgb
    e = (emb - emb.mean(0)) / (emb.std(0) + 1e-09)
    e = np.clip(e / 2.0, -1, 1)
    lab = np.stack([57.5 + 22.5 * e[:, 0], 40 * e[:, 1], 40 * e[:, 2]], 1)
    return np.clip(lab2rgb(lab[None])[0], 0, 1)


def smooth_embedding(emb, chunk, gy, gx, rad=1):
    """Mean of each cell's embedding over its (2*rad+1)^2 neighbourhood on the
    chunk-wide cell grid (missing/gated-out neighbours simply don't count)."""
    out = np.empty_like(emb)
    offs = [(dy, dx) for dy in range(-rad, rad + 1)
            for dx in range(-rad, rad + 1)]
    for c in np.unique(chunk):
        m = chunk == c
        y, x = gy[m], gx[m]
        H, W = y.max() + 1 + rad, x.max() + 1 + rad
        acc = np.zeros((H + rad, W + rad, emb.shape[1]))
        cnt = np.zeros((H + rad, W + rad))
        acc[y + rad, x + rad] = emb[m]
        cnt[y + rad, x + rad] = 1
        s = np.zeros((m.sum(), emb.shape[1]))
        d = np.zeros(m.sum())
        for dy, dx in offs:
            s += acc[y + rad + dy, x + rad + dx]
            d += cnt[y + rad + dy, x + rad + dx]
        out[m] = s / d[:, None]
    return out


def load_cluster_rows(path):
    """clusters_{...}.npz (harness.py cluster) -> (labels, names, FeatureSet-
    like row keys). Cluster names come from the sibling *_names.csv when it
    has been filled in; unnamed clusters keep c{i}."""
    import json as _json
    z = np.load(path, allow_pickle=False)
    meta = _json.loads(str(z['meta']))
    k = int(meta['k'])
    names = {i: f'c{i}' for i in range(k)}
    names_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f'{meta["name"]}_names.csv')
    if os.path.exists(names_csv):
        with open(names_csv) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith('#') or ln.startswith('cluster,'):
                    continue
                parts = [p.strip() for p in ln.split(',')]
                if len(parts) >= 3 and parts[2]:
                    names[int(parts[0])] = parts[2]
    return z, meta, names


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--feat', default=None,
                    help='feature npz (required for umap/anchors modes)')
    ap.add_argument('--mode', choices=('umap', 'anchors', 'clusters'),
                    required=True)
    ap.add_argument('--clusters', default=None,
                    help='clusters_{...}.npz (required for clusters mode)')
    ap.add_argument('--smooth', type=int, default=0,
                    help='umap mode: average embeddings over the (2N+1)^2 '
                         'cell neighbourhood before coloring')
    ap.add_argument('--name', default=None)
    ap.add_argument('--codec', default=common.CODEC)
    ap.add_argument('--anchors', default=ANCHORS_DEFAULT)
    ap.add_argument('--gate_k', type=float, default=2.0)
    ap.add_argument('--sample', type=int, default=100000,
                    help='UMAP fit subsample (transform covers the rest)')
    ap.add_argument('--alpha', type=float, default=0.55)
    ap.add_argument('--chunks', default=None,
                    help='render only these chunks (npy still covers all)')
    args = ap.parse_args()
    cls_idx = p_cls = None
    if args.mode == 'clusters':
        if not args.clusters:
            sys.exit('--mode clusters needs --clusters clusters_{...}.npz')
        z, meta, names = load_cluster_rows(args.clusters)
        cell_px = int(meta.get('cell_px', common.CELL))
        name = args.name or f'{meta["name"]}_map'
        lab_arr, dist = z['label'].astype(int), z['dist']
        ch, st = z['chunk'].astype(str), z['stem'].astype(str)
        cr, cc = z['cell_row'], z['cell_col']
        rgbs = CLUSTER_HUES_FULL[lab_arr % len(CLUSTER_HUES_FULL)]
        emb = np.stack([lab_arr.astype(np.float32), dist,
                        np.zeros(len(lab_arr), np.float32)], 1)
        cls_idx = lab_arr.astype(np.float32)
        p_cls = np.ones(len(lab_arr), np.float32)
        classes = [names[i] for i in sorted(names)]
        fs_meta_name = meta['feat_name']
        for i in sorted(names):
            print(f'  cluster {i} = {names[i]}  '
                  f'rgb {CLUSTER_HUES_FULL[i % len(CLUSTER_HUES_FULL)]}')
    elif not args.feat:
        sys.exit(f'--mode {args.mode} needs --feat')
    else:
        fs = common.load_feature_set(args.feat)
        cell_px = int(fs.meta.get('cell_px', common.CELL))
        name = args.name or (
            f'{os.path.splitext(os.path.basename(args.feat))[0]}_{args.mode}'
            + (f'_sm{args.smooth}' if args.smooth and args.mode == 'umap'
               else ''))
        fs_meta_name = fs.meta.get('name')
        keep = common.row_gate(fs, args.codec, cell_px, args.gate_k)
        print(f'gate kept {keep.sum()}/{len(keep)} cells (k={args.gate_k})',
              flush=True)
        F = fs.feat[keep]
        ch, st = fs.chunk[keep], fs.stem[keep]
        cr, cc = fs.cell_row[keep], fs.cell_col[keep]
    if args.mode == 'umap':
        emb = embed_umap(F, args.sample)
        if args.smooth > 0:
            n = common.ncell(cell_px)
            rr = np.array([common.stem_rc(s)[0] for s in st])
            cn = np.array([common.stem_rc(s)[1] for s in st])
            gy = rr * n + np.maximum(cr, 0)
            gx = cn * n + np.maximum(cc, 0)
            emb = smooth_embedding(emb, ch, gy, gx, args.smooth)
            print(f'smoothed embeddings over {2*args.smooth+1}x'
                  f'{2*args.smooth+1} cells', flush=True)
        rgbs = lab_rgb(emb)
        classes = None
    elif args.mode == 'anchors':
        anchor_rows = read_anchors(args.anchors)
        lab = {(r['chunk'], r['stem']): r['class'] for r in anchor_rows}
        ya = np.array([lab.get((c, s), '') for c, s in zip(fs.chunk, fs.stem)])
        tr = ya != ''
        classes = sorted(set(ya[tr]))
        if len(classes) > len(CLASS_HUES):
            sys.exit(f'{len(classes)} classes > {len(CLASS_HUES)} hues')
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(fs.feat[tr])
        lin = LogisticRegression(max_iter=1000, multi_class='multinomial',
                                 C=1.0).fit(sc.transform(fs.feat[tr]), ya[tr])
        pid = np.array([f'{c}/{s}' for c, s in zip(fs.chunk[tr], fs.stem[tr])])
        pr = probe_loco(fs.feat[tr], ya[tr], fs.chunk[tr], pid)
        print(f'LOCO probe: lin patch {pr["lin_patch"]:.3f} / cell '
              f'{pr["lin_cell"]:.3f}; classes {pr["classes"]}')
        for row in pr['confusion']:
            print('   ', row)
        proba = lin.predict_proba(sc.transform(F))
        rgbs = np.clip(proba @ CLASS_HUES[:len(classes)], 0, 1)
        cls_idx = proba.argmax(1).astype(np.float32)
        p_cls = proba.max(1).astype(np.float32)
        emb = np.zeros((len(F), 3), np.float32)
        emb[:, :min(3, proba.shape[1])] = proba[:, :3]
        for i, c in enumerate(classes):
            print(f'  class {i} = {c}  hue {CLASS_HUES[i]}')
    xyz = np.array([common.cell_xyz(c, s, r, q, cell_px)
                    for c, s, r, q in zip(ch, st, cr, cc)], np.float32)
    cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2],
            cr.astype(np.float32), cc.astype(np.float32),
            emb[:, 0], emb[:, 1], emb[:, 2]]
    if cls_idx is not None:
        cols += [cls_idx, p_cls]
    mat = np.stack(cols, 1).astype(np.float32)
    os.makedirs(common.OUT, exist_ok=True)
    np.save(os.path.join(common.OUT, f'{name}.npy'), mat)
    print(f'wrote {os.path.join(common.OUT, name)}.npy {mat.shape} '
          '(z, x, y, cell_row, cell_col, emb1..3'
          + (', class, p_class)' if cls_idx is not None else ')'), flush=True)
    fig_dir = os.path.join(common.OUT, name)
    os.makedirs(fig_dir, exist_ok=True)
    rows = np.array([common.stem_rc(s)[0] for s in st])
    colns = np.array([common.stem_rc(s)[1] for s in st])
    size = np.where(cr < 0, common.PATCH, cell_px)
    x_px = colns * common.PATCH + np.maximum(cc, 0) * cell_px
    y_px = rows * common.PATCH + np.maximum(cr, 0) * cell_px
    chunks = args.chunks.split(',') if args.chunks else sorted(set(ch))
    for c in chunks:
        sl, zmid = common.read_mid_slice(args.codec, c)
        fig, ax, ds = render.chunk_figure(sl)
        m = ch == c
        render.draw_boxes(ax, x_px[m], y_px[m],
                          int(size[m][0]) if m.any() else cell_px,
                          rgbs[m], ds=ds, alpha=args.alpha)
        if args.mode == 'umap':
            mode_lbl = ('UMAP3→LAB' + (f' smoothed {2*args.smooth+1}x'
                                       f'{2*args.smooth+1}'
                                       if args.smooth else ''))
        else:
            mode_lbl = (f'{"anchor" if args.mode == "anchors" else "cluster"} '
                        f'classes {classes}')
        ax.set_title(f'{c} z={zmid} — {fs_meta_name} | {mode_lbl} | '
                     f'cell {cell_px}px, gate k={args.gate_k}')
        fig.tight_layout()
        png = os.path.join(fig_dir, f'{c}.png')
        fig.savefig(png, dpi=130)
        render.close(fig)
        print(f'wrote {png}', flush=True)


if __name__ == '__main__':
    main()
