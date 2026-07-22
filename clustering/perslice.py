"""Per-slice class inference (1-1 slice heatmaps) from a registered version.

The 7-slice-window features remain the ANCHOR regime: a version's k-means
labels (e.g. v0b fiber classes, plus its parent's junk classes) define what
the classes are. This script re-anchors those classes in SINGLE-SLICE feature
space — per-slice F1 histograms + F2 co-occurrence per 48-px cell, weighted
by the stored idf vectors — by averaging the single-slice features of the
already-labeled cells at the window centre (local z 24). Every cell of every
slice is then assigned to its nearest class reference: pure inference, class
meanings unchanged, color matches the content of that slice.

Caveats: a single slice carries ~7x fewer tokens per cell than the window
(36 s3 + 9 s2), so per-slice maps are noisier than the window maps; and
cells are forced to the nearest known class — texture unlike anything the
mid-windows sampled is not flagged as novel.

ON-DEMAND ONLY (2026-07-22): whole-volume per-slice computation is retired
and fully removed — slices are computed only as requested via --z, against
per-version cached class references (perslice_{version}_refs.npz; delete it
after changing labels or names). Outputs land in
Output/clustering/perslice_{version}/.

  python clustering/perslice.py --version v1a1 --z 115          # ~seconds
  python clustering/perslice.py --version v1a1 --z 100-130
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from clustering import common
from clustering.feat_f1 import ZLOCAL_MID
from clustering.feat_f2 import anisotropy, tri_lookup
from clustering.harness import CLUSTER_HUES, load_versions
from clustering.metacodes import build_metacodes


def slice_f1_counts(grids, assign, scales, K, n):
    """[(Z,hk,wk) per scale] -> (Z, n*n, len(scales)*K) uint16 counts."""
    Z = grids[scales[0]].shape[0]
    out = np.zeros((Z, n * n, len(scales) * K), np.uint16)
    for b, s in enumerate(scales):
        g = grids[s]
        m = assign[s][g]
        hk = g.shape[1]
        ti = np.arange(hk) // (hk // n)
        cell = ti[:, None] * n + ti[None, :]
        zidx = np.arange(Z)[:, None, None]
        flat = (zidx * (n * n * K) + cell[None] * K + m).ravel()
        cnt = np.bincount(flat, minlength=Z * n * n * K).reshape(Z, n * n, K)
        out[:, :, b * K:(b + 1) * K] = cnt
    return out


def slice_f2_counts(grid, assign, T, ntri, K, n):
    """(Z,hk,wk) s3 codes -> (Z, n*n, 3, ntri) pair counts; the z direction
    pairs slice z with z+1 (last slice falls back to z-1) so every slice has
    a defined z block."""
    Z, hk = grid.shape[0], grid.shape[1]
    m = assign[grid]
    ti = np.arange(hk) // (hk // n)
    cell = ti[:, None] * n + ti[None, :]
    out = np.zeros((Z, n * n, 3, ntri), np.int64)
    zidx = np.arange(Z)[:, None, None]

    def acc(d, a, b, cid):
        flat = (np.broadcast_to(zidx, a.shape).ravel() * (n * n * ntri)
                + np.broadcast_to(cid[None], a.shape).ravel() * ntri
                + T[a.ravel(), b.ravel()])
        out[:, :, d, :] += np.bincount(
            flat, minlength=Z * n * n * ntri).reshape(Z, n * n, ntri)

    acc(0, m[:, :, :-1], m[:, :, 1:], cell[:, :-1])
    acc(1, m[:, :-1, :], m[:, 1:, :], cell[:-1, :])
    za = np.concatenate([m[:-1], m[-2:-1]], 0)   # partner slice per z
    zb = np.concatenate([m[1:], m[-1:]], 0)
    flat = (np.broadcast_to(zidx, za.shape).ravel() * (n * n * ntri)
            + np.broadcast_to(cell[None], za.shape).ravel() * ntri
            + T[za.ravel(), zb.ravel()])
    out[:, :, 2, :] += np.bincount(
        flat, minlength=Z * n * n * ntri).reshape(Z, n * n, ntri)
    return out


def tfidf_sqrt(counts, idf):
    tf = counts / (counts.sum(-1, keepdims=True) + 1e-12)
    F = tf * idf
    F /= F.sum(-1, keepdims=True) + 1e-12
    return np.sqrt(F)


def batch_f1_counts(g_list, assign, scales, K, n):
    """[(B, Z, hk, wk) codes per scale] -> (B, Z, n*n, len(scales)*K)
    float64 counts, vectorized across the whole patch batch."""
    B, Z = g_list[0].shape[:2]
    out = np.zeros((B, Z, n * n, len(scales) * K), np.float64)
    for bi, (g, s) in enumerate(zip(g_list, scales)):
        m = assign[s][g]
        hk = g.shape[2]
        ti = np.arange(hk) // (hk // n)
        cell = ti[:, None] * n + ti[None, :]
        base = ((np.arange(B)[:, None, None, None] * Z
                 + np.arange(Z)[None, :, None, None]) * (n * n)
                + cell[None, None]) * K
        cnt = np.bincount((base + m).ravel(), minlength=B * Z * n * n * K)
        out[..., bi * K:(bi + 1) * K] = cnt.reshape(B, Z, n * n, K)
    return out


def batch_f2_counts(g, assign, T, ntri, n):
    """(B, Z, hk, wk) s3 codes -> (B*Z*n*n, 3, ntri) int32 pair counts,
    same direction/edge conventions as slice_f2_counts."""
    m = assign[g]
    B, Z, hk = m.shape[:3]
    ti = np.arange(hk) // (hk // n)
    cell = ti[:, None] * n + ti[None, :]
    pz = (np.arange(B)[:, None, None, None] * Z
          + np.arange(Z)[None, :, None, None])
    out = np.zeros((B * Z * n * n, 3, ntri), np.int32)

    def acc(d, a, b, cid):
        flat = ((pz * (n * n) + cid) * ntri + T[a, b]).ravel()
        out[:, d, :] += np.bincount(
            flat, minlength=B * Z * n * n * ntri).reshape(-1, ntri).astype(np.int32)

    acc(0, m[:, :, :, :-1], m[:, :, :, 1:], cell[None, None, :, :-1])
    acc(1, m[:, :, :-1, :], m[:, :, 1:, :], cell[None, None, :-1, :])
    za = np.concatenate([m[:, :-1], m[:, -2:-1]], 1)
    zb = np.concatenate([m[:, 1:], m[:, -1:]], 1)
    acc(2, za, zb, cell[None, None])
    return out


class SliceFeaturizer:
    """Single-slice f1+f2 features matching the window feature's columns."""

    def __init__(self, feat_params):
        p = feat_params
        self.K, self.n = p['K'], common.ncell(p['cell_px'])
        self.scales = (2, 3)
        self.s3 = p['scale']
        assign = build_metacodes(p['codec'], self.scales, self.K,
                                 p['context'], model=p['model'],
                                 epoch=p['epoch'])
        self.assign = {s: assign[s] for s in self.scales}
        f1p = {'family': 'f1', 'codec': p['codec'], 'model': p['model'],
               'epoch': p['epoch'], 'scales': list(self.scales), 'K': self.K,
               'context': p['context'], 'zwin': p['zwin'],
               'cell_px': p['cell_px'], 'version': 1}
        f2p = {k: p[k] for k in ('family', 'codec', 'model', 'epoch', 'scale',
                                 'K', 'context', 'zwin', 'cell_px', 'version')}
        self.idf1 = np.load(os.path.join(common.cache_dir('f1', f1p), 'idf.npy'))
        d2 = common.cache_dir('f2', f2p)
        self.idf2 = np.load(os.path.join(d2, 'idf.npy'))
        self.top = np.load(os.path.join(d2, 'top_pairs.npy'))[:, :p['top_m']]
        self.T, self.ntri = tri_lookup(self.K)

    def patch(self, codec, chunk, stem):
        """(Z, n*n, D) float32 single-slice features for one patch."""
        grids = common.load_codes(codec, chunk, stem)
        c1 = slice_f1_counts(grids, self.assign, self.scales, self.K, self.n)
        F1 = tfidf_sqrt(c1.astype(np.float64), self.idf1)
        c2 = slice_f2_counts(grids[self.s3], self.assign[self.s3],
                             self.T, self.ntri, self.K, self.n)
        sel = np.concatenate([c2[:, :, d, self.top[d]] for d in range(3)], -1)
        F2 = tfidf_sqrt(sel.astype(np.float64), self.idf2)
        Z = c2.shape[0]
        an = anisotropy(c2.reshape(-1, 3, self.ntri), self.T,
                        self.K).reshape(Z, -1, 3 * self.K + 3)
        return np.concatenate([F1, F2, an], -1).astype(np.float32)

    def batch(self, g_by_scale, g_s3):
        """([(B, Z, hk, wk) codes for self.scales], (B, Z, 48, 48) s3 codes)
        -> (B, Z, n*n, D) float32; vectorized equivalent of patch()."""
        B, Z = g_s3.shape[:2]
        c1 = batch_f1_counts(g_by_scale, self.assign, self.scales, self.K,
                             self.n).reshape(B * Z * self.n * self.n, -1)
        F1 = tfidf_sqrt(c1, self.idf1)
        c2 = batch_f2_counts(g_s3, self.assign[self.s3], self.T, self.ntri,
                             self.n)
        sel = np.concatenate([c2[:, d, self.top[d]] for d in range(3)], -1)
        F2 = tfidf_sqrt(sel.astype(np.float64), self.idf2)
        an = anisotropy(c2, self.T, self.K)
        D = F1.shape[1] + F2.shape[1] + an.shape[1]
        return np.concatenate([F1, F2, an], -1).astype(np.float32).reshape(
            B, Z, self.n * self.n, D)

    def chunk_features(self, codec, chunk, stems, z0=None, z1=None,
                       batch=16):
        """(S, Z', n*n, D) for a whole chunk via the codec_fast store
        (falls back to per-patch npz reads). z0:z1 selects a z range."""
        fast = common._fast_store(codec, chunk)
        outs = []
        if fast is not None:
            idx, arrs = fast
            order = np.array([idx[s] for s in stems])
            for a in range(0, len(order), batch):
                sel = order[a:a + batch]
                gs = [np.asarray(arrs[s][sel, z0:z1], np.int64)
                      for s in self.scales]
                g3 = np.asarray(arrs[self.s3][sel, z0:z1], np.int64)
                outs.append(self.batch(gs, g3))
        else:
            for st in stems:
                f = self.patch(codec, chunk, st)[z0:z1]
                outs.append(f[None])
        return np.concatenate(outs, 0)


def load_labels(reg, version):
    """{(chunk, stem, cr, cc): class_id} + id->name map. Fiber classes come
    from `version`; background classes from the FULL ancestor chain — each
    ancestor's non-keep cells (walking to the root, so a grandchild version
    still gets the root's true background/junk references)."""
    v = reg[version]
    names, labels = {}, {}
    z = np.load(v['clusters'], allow_pickle=False)
    for c, s, r, q, l in zip(z['chunk'].astype(str), z['stem'].astype(str),
                             z['cell_row'], z['cell_col'], z['label']):
        labels[(c, s, int(r), int(q))] = int(l)
    for i, nm in v['classes'].items():
        names[int(i)] = nm
    n_own = v['k']
    offset = n_own
    cur = v.get('parent')
    while cur:
        pv = reg[cur]
        pkeep = set(pv.get('keep') or [])
        z = np.load(pv['clusters'], allow_pickle=False)
        for c, s, r, q, l in zip(z['chunk'].astype(str), z['stem'].astype(str),
                                 z['cell_row'], z['cell_col'], z['label']):
            key = (c, s, int(r), int(q))
            if int(l) in pkeep or key in labels:
                continue                   # kept cells owned by descendants
            labels[key] = offset + int(l)
        for i, nm in pv['classes'].items():
            if int(i) not in pkeep:
                names[offset + int(i)] = f'bg:{nm}'
        offset += pv['k']
        cur = pv.get('parent')
    return labels, names, n_own


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--version', required=True)
    ap.add_argument('--codec', default=common.CODEC)
    ap.add_argument('--chunks', default=None,
                    help='subset for smoke tests (default: all)')
    ap.add_argument('--z', default=None,
                    help='on-demand overlays for these z slices only (same '
                         'syntax as --render): computes features just for '
                         'them, using cached refs — seconds per slice, no '
                         'volume write')
    ap.add_argument('--alpha', type=float, default=0.55)
    args = ap.parse_args()
    reg = load_versions()
    if args.version not in reg:
        sys.exit(f'unknown version {args.version}')
    if not args.z:
        sys.exit('per-slice is on-demand only: pass --z <slices|a-b>, e.g. '
                 '--z 115 or --z 100-130')
    v = reg[args.version]
    feat_meta = common.load_feature_set(v['feat']).meta
    fz = SliceFeaturizer(feat_meta['params'])
    labels, names, n_fiber = load_labels(reg, args.version)
    print(f'{args.version}: {len(labels)} labeled cells, classes {names}',
          flush=True)

    chunks = args.chunks.split(',') if args.chunks else common.list_chunks(args.codec)
    n = fz.n
    n_cls = max(names) + 1

    # pass A: single-slice features at the window centre -> class references
    # (cached per version; delete the npz after changing labels/names/keeps)
    refs_npz = os.path.join(common.OUT, f'perslice_{args.version}_refs.npz')
    if os.path.exists(refs_npz) and args.chunks is None:
        z = np.load(refs_npz)
        refs, ref_ids = z['refs'], z['ref_ids']
        print(f'refs cache hit {os.path.basename(refs_npz)}', flush=True)
    else:
        acc = np.zeros((n_cls, 0), np.float64)
        cnt = np.zeros(n_cls, np.int64)
        for ch in chunks:
            stems = common.list_stems(args.codec, ch)
            zc = ZLOCAL_MID                               # 48-slice chunks
            F = fz.chunk_features(args.codec, ch, stems, z0=zc, z1=zc + 2)[:, 0]
            if acc.shape[1] == 0:
                acc = np.zeros((n_cls, F.shape[-1]), np.float64)
            for si, st in enumerate(stems):
                for ci in range(n):
                    for cj in range(n):
                        l = labels.get((ch, st, ci, cj))
                        if l is not None:
                            acc[l] += F[si, ci * n + cj]
                            cnt[l] += 1
            print(f'refs {ch}: {int(cnt.sum())} labeled cells accumulated',
                  flush=True)
        ok = cnt > 0
        refs = acc[ok] / cnt[ok, None]
        ref_ids = np.where(ok)[0]
        print('reference cells per class: '
              + ' '.join(f'{names[i]}={cnt[i]}' for i in ref_ids), flush=True)
        if args.chunks is None:
            np.savez(refs_npz, refs=refs, ref_ids=ref_ids)
            print(f'cached refs -> {refs_npz}', flush=True)

    if args.z:
        from clustering import render
        from inference.zarr_io import open_zarr
        out = os.path.join(common.OUT, f'perslice_{args.version}')
        os.makedirs(out, exist_ok=True)
        np0 = common.load_norm(args.codec, chunks[0])
        arr = open_zarr(np0['source'], np0['zarr_level'])
        r2 = (refs ** 2).sum(1)
        zs = []
        for part in args.z.split(','):
            if '-' in part:
                a, b = (int(x) for x in part.split('-'))
                zs += list(range(a, b + 1))
            else:
                zs.append(int(part))
        for zabs in sorted(set(zs)):
            ch = common.list_chunks(args.codec)[zabs // 48]
            zl = zabs % 48
            z0 = zl if zl < 47 else 46
            stems = common.list_stems(args.codec, ch)
            F = fz.chunk_features(args.codec, ch, stems, z0=z0, z1=z0 + 2)
            F = F[:, zl - z0]                              # (S, n*n, D)
            X = F.reshape(-1, F.shape[-1])
            lab = ref_ids[(r2[None] - 2 * (X @ refs.T)).argmin(1)]
            lab = lab.reshape(len(stems), n, n)
            sl = np.asarray(arr[zabs].read().result()).T.astype(np.float32)
            fig, ax, ds = render.chunk_figure(sl)
            keep_m, xs, ys, rgbs = [], [], [], []
            for si, st in enumerate(stems):
                r, c = common.stem_rc(st)
                for ci in range(n):
                    for cj in range(n):
                        l = lab[si, ci, cj]
                        if l < n_fiber:
                            xs.append(c * common.PATCH + cj * int(feat_meta['cell_px']))
                            ys.append(r * common.PATCH + ci * int(feat_meta['cell_px']))
                            rgbs.append(CLUSTER_HUES[l % len(CLUSTER_HUES)])
            render.draw_boxes(ax, np.array(xs), np.array(ys), int(feat_meta['cell_px']),
                              np.array(rgbs), ds=ds, alpha=args.alpha)
            ax.set_title(f'z={zabs} — perslice {args.version} (--z on-demand) '
                         f'| fiber classes {[names[i] for i in range(n_fiber)]}')
            fig.tight_layout()
            png = os.path.join(out, f'z{zabs:04d}.png')
            fig.savefig(png, dpi=130)
            render.close(fig)
            print(f'wrote {png}', flush=True)
        return

    # (whole-volume classification retired 2026-07-22 — the --z path above is
    # the only computation mode; --render serves pre-retirement volumes)


if __name__ == '__main__':
    main()
