"""M3 / F2 — directional metacode co-occurrence (PLAN §5, second order).

Per 48-px cell, metacode PAIR counts for three neighbour directions on the s3
token grid — right (+x), down (+y), next-z — counted over the 7-slice z-window
and symmetrized within each direction (K x K upper triangle). Captures how the
material is arranged (layered vs speckled vs fibrous) and its anisotropy —
exactly what mean-pooling destroyed.

Feature = [top-M pairs per direction, tf-idf + sqrt like F1]  ++  an explicit
anisotropy block: per-metacode same-code pair fractions for H/V/Z (3K dims)
plus 3 scalars (H-V asymmetry, inplane-vs-Z asymmetry, Z-persistence).

  python clustering/feat_f2.py --K 24 --zwin 3 [--top_m 64] [--concat-f1 F1.npz]

Per-chunk full-triangle counts cached under cache/f2-{key}/ (so --top_m can
change without recounting); assembled npz -> Output/clustering/.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from clustering import common
from clustering.feat_f1 import assemble, zwindow
from clustering.metacodes import build_metacodes


def tri_lookup(K):
    """(K, K) -> upper-triangle pair index; T[a, b] == T[b, a]."""
    T = np.zeros((K, K), np.int32)
    t = 0
    for a in range(K):
        for b in range(a, K):
            T[a, b] = T[b, a] = t
            t += 1
    return T, t


def cooc_counts(m, T, ntri, n):
    """(Zw, hk, wk) metacodes -> (n, n, 3, ntri) uint16 pair counts.
    Directions: 0 = right(+x), 1 = down(+y), 2 = next-z; each pair is binned
    to the cell of its first (left/top/earlier) token."""
    hk = m.shape[1]
    tpc = hk // n
    ti = np.arange(hk) // tpc
    cell_id = ti[:, None] * n + ti[None, :]
    out = np.zeros((n * n, 3, ntri), np.int64)
    pairs = ((0, m[:, :, :-1], m[:, :, 1:], cell_id[:, :-1]),
             (1, m[:, :-1, :], m[:, 1:, :], cell_id[:-1, :]),
             (2, m[:-1], m[1:], cell_id))
    for d, a, b, cid in pairs:
        flat = (np.broadcast_to(cid, a.shape).ravel() * ntri
                + T[a.ravel(), b.ravel()])
        out[:, d, :] += np.bincount(flat, minlength=n * n * ntri).reshape(n * n, ntri)
    return out.reshape(n, n, 3, ntri).astype(np.uint16)


def chunk_cooc(codec, chunk, assign, scale, K, zwin, cell_px, cache_d):
    T, ntri = tri_lookup(K)
    cache = os.path.join(cache_d, f'{chunk}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return list(z['stems']), z['counts']
    stems = common.list_stems(codec, chunk)
    n = common.ncell(cell_px)
    counts = np.zeros((len(stems), n, n, 3, ntri), np.uint16)
    for i, st in enumerate(stems):
        g = common.load_codes(codec, chunk, st)[scale]
        m = assign[g[zwindow(g.shape[0], zwin)]]
        counts[i] = cooc_counts(m, T, ntri, n)
    np.savez_compressed(cache, stems=np.array(stems), counts=counts)
    print(f'f2 {chunk}: counted {len(stems)} patches', flush=True)
    return stems, counts


def anisotropy(counts, T, K):
    """(N, 3, ntri) -> (N, 3K + 3) float32: per-metacode same-code fractions
    per direction, then H-V asymmetry, inplane-vs-Z asymmetry, Z-persistence."""
    diag = np.array([T[k, k] for k in range(K)])
    tot = counts.sum(-1).astype(np.float64) + 1e-09
    same_k = counts[:, :, diag].astype(np.float64)
    frac_k = same_k / tot[:, :, None]
    p = same_k.sum(-1) / tot
    scal = np.stack([p[:, 0] - p[:, 1],
                     0.5 * (p[:, 0] + p[:, 1]) - p[:, 2],
                     p[:, 2]], 1)
    return np.concatenate([frac_k.reshape(len(counts), -1), scal],
                          1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--codec', default=common.CODEC)
    ap.add_argument('--K', type=int, default=24)
    ap.add_argument('--scale', type=int, default=3,
                    help='token grid for pairs (default s3 = 8 px/token)')
    ap.add_argument('--zwin', type=int, default=3)
    ap.add_argument('--cell_px', type=int, default=common.CELL)
    ap.add_argument('--top_m', type=int, default=64,
                    help='pairs kept per direction (by document frequency)')
    ap.add_argument('--no-context', action='store_true')
    ap.add_argument('--chunks', default=None)
    ap.add_argument('--model', default='skipU')
    ap.add_argument('--epoch', type=int, default=300)
    ap.add_argument('--concat-f1', default=None,
                    help='path to an F1 npz to concatenate (extra harness row)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    assign = build_metacodes(args.codec, (args.scale,), args.K,
                             not args.no_context, model=args.model,
                             epoch=args.epoch)[args.scale]
    params = {'family': 'f2', 'codec': args.codec, 'model': args.model,
              'epoch': args.epoch, 'scale': args.scale, 'K': args.K,
              'context': not args.no_context, 'zwin': args.zwin,
              'cell_px': args.cell_px, 'version': 1}
    cache_d = common.cache_dir('f2', params)
    chunks = args.chunks.split(',') if args.chunks else common.list_chunks(args.codec)
    T, ntri = tri_lookup(args.K)
    n = common.ncell(args.cell_px)
    counts, chs, sts, crs, ccs = [], [], [], [], []
    for ch in chunks:
        stems, cnt = chunk_cooc(args.codec, ch, assign, args.scale, args.K,
                                args.zwin, args.cell_px, cache_d)
        counts.append(cnt.reshape(-1, 3, ntri))
        for st in stems:
            chs += [ch] * (n * n)
            sts += [st] * (n * n)
        rr, cc2 = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        crs.append(np.tile(rr.ravel(), len(stems)))
        ccs.append(np.tile(cc2.ravel(), len(stems)))
    C = np.concatenate(counts)
    df = (C > 0).sum(0)
    top = np.argsort(-df, 1)[:, :args.top_m]
    np.save(os.path.join(cache_d, 'top_pairs.npy'), top)
    sel = np.concatenate([C[:, d, top[d]] for d in range(3)], 1)
    F, idf = assemble(sel)
    np.save(os.path.join(cache_d, 'idf.npy'), idf)
    F = np.concatenate([F, anisotropy(C, T, args.K)], 1)
    key = common.params_key(params)
    tag = '' if args.chunks is None else '_sub'
    out = args.out or os.path.join(common.OUT,
                                   f'f2_K{args.K}zw{args.zwin}_{key}{tag}.npz')
    meta = {'name': os.path.splitext(os.path.basename(out))[0], 'family': 'f2',
            'cell_px': args.cell_px, 'zwin': args.zwin,
            'params': dict(params, top_m=args.top_m)}
    chs, sts = np.array(chs), np.array(sts)
    crs, ccs = np.concatenate(crs), np.concatenate(ccs)
    common.save_features(out, F, chs, sts, crs, ccs, meta)
    if args.concat_f1:
        f1 = common.load_feature_set(args.concat_f1)
        key_self = {(c, s, r, q): i
                    for i, (c, s, r, q) in enumerate(zip(chs, sts, crs, ccs))}
        order = np.array([key_self[(c, s, r, q)] for c, s, r, q in
                          zip(f1.chunk, f1.stem, f1.cell_row, f1.cell_col)])
        Fc = np.concatenate([f1.feat, F[order]], 1)
        outc = out.replace('f2_', 'f1+f2_')
        metac = dict(meta, name=os.path.splitext(os.path.basename(outc))[0],
                     family='f1+f2')
        common.save_features(outc, Fc, f1.chunk, f1.stem, f1.cell_row,
                             f1.cell_col, metac)


if __name__ == '__main__':
    main()
