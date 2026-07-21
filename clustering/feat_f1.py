"""M2 / F1 — per-cell metacode histograms (PLAN §5, first priority).

Per 48-px cell and z-window slice, the codec provides 6x6 s3 tokens (8 px each)
plus 3x3 s2 tokens; over the 7-slice window (zwin=3 around the chunk mid
slice, local z 24) that is ~252 + ~63 tokens. Codes -> metacodes
(metacodes.build_metacodes), per-cell K-bin counts per scale, then globally:
tf-idf -> L1 -> sqrt (Hellinger), so Euclidean/cosine downstream behave.
Brightness enters only through which codes fire.

  python clustering/feat_f1.py --K 24 --zwin 3 [--cell_px 48] [--chunks a,b]

Per-chunk raw counts cached under cache/f1-{key}/ (key covers model, epoch,
scales, K, context, zwin, cell_px — the PLAN §8 fix); the assembled feature
npz lands in Output/clustering/ and is what harness.py eval consumes.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from clustering import common
from clustering.metacodes import build_metacodes

ZLOCAL_MID = 24


def zwindow(nz, zwin):
    zc = min(ZLOCAL_MID, nz // 2)
    return slice(max(0, zc - zwin), min(nz, zc + zwin + 1))


def cell_counts(grid, assign, K, n):
    """(Z, hk, wk) codes -> (n, n, K) metacode counts, tokens binned to the
    n x n cell grid (one vectorized bincount)."""
    m = assign[grid]
    hk = grid.shape[1]
    tpc = hk // n
    ti = np.arange(hk) // tpc
    cell_id = ti[:, None] * n + ti[None, :]
    flat = (cell_id[None] * K + m).ravel()
    return np.bincount(flat, minlength=n * n * K).reshape(n, n, K).astype(np.uint16)


def chunk_counts(codec, chunk, assign, scales, K, zwin, cell_px, cache_d):
    """(stems, counts (S, n, n, len(scales)*K) uint16), per-chunk cached."""
    cache = os.path.join(cache_d, f'{chunk}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return list(z['stems']), z['counts']
    stems = common.list_stems(codec, chunk)
    n = common.ncell(cell_px)
    counts = np.zeros((len(stems), n, n, len(scales) * K), np.uint16)
    for i, st in enumerate(stems):
        grids = common.load_codes(codec, chunk, st)
        for b, s in enumerate(scales):
            g = grids[s]
            counts[i, :, :, b * K:(b + 1) * K] = cell_counts(
                g[zwindow(g.shape[0], zwin)], assign[s], K, n)
    np.savez_compressed(cache, stems=np.array(stems), counts=counts)
    print(f'f1 {chunk}: counted {len(stems)} patches', flush=True)
    return stems, counts


def assemble(all_counts, sqrt=True):
    """Stacked raw counts (N, D) -> tf-idf -> L1 -> sqrt; returns (F, idf)."""
    Craw = all_counts.astype(np.float64)
    df = (Craw > 0).sum(0)
    idf = np.log(len(Craw) / (1.0 + df))
    tf = Craw / (Craw.sum(1, keepdims=True) + 1e-12)
    F = tf * idf
    F /= F.sum(1, keepdims=True) + 1e-12
    if sqrt:
        F = np.sqrt(F)
    return F.astype(np.float32), idf.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--codec', default=common.CODEC)
    ap.add_argument('--K', type=int, default=24)
    ap.add_argument('--scales', default='2,3')
    ap.add_argument('--zwin', type=int, default=3)
    ap.add_argument('--cell_px', type=int, default=common.CELL)
    ap.add_argument('--no-context', action='store_true')
    ap.add_argument('--chunks', default=None,
                    help='subset for smoke tests (default: all)')
    ap.add_argument('--model', default='skipU')
    ap.add_argument('--epoch', type=int, default=300)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    scales = tuple(int(s) for s in args.scales.split(','))
    assign = build_metacodes(args.codec, scales, args.K, not args.no_context,
                             model=args.model, epoch=args.epoch)
    params = {'family': 'f1', 'codec': args.codec, 'model': args.model,
              'epoch': args.epoch, 'scales': list(scales), 'K': args.K,
              'context': not args.no_context, 'zwin': args.zwin,
              'cell_px': args.cell_px, 'version': 1}
    cache_d = common.cache_dir('f1', params)
    chunks = args.chunks.split(',') if args.chunks else common.list_chunks(args.codec)
    n = common.ncell(args.cell_px)
    counts, chs, sts, crs, ccs = [], [], [], [], []
    for ch in chunks:
        stems, cnt = chunk_counts(args.codec, ch, assign, scales, args.K,
                                  args.zwin, args.cell_px, cache_d)
        counts.append(cnt.reshape(-1, cnt.shape[-1]))
        for st in stems:
            chs += [ch] * (n * n)
            sts += [st] * (n * n)
        rr, cc2 = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        crs.append(np.tile(rr.ravel(), len(stems)))
        ccs.append(np.tile(cc2.ravel(), len(stems)))
    F, idf = assemble(np.concatenate(counts))
    np.save(os.path.join(cache_d, 'idf.npy'), idf)
    key = common.params_key(params)
    tag = '' if args.chunks is None else '_sub'
    out = args.out or os.path.join(common.OUT,
                                   f'f1_K{args.K}zw{args.zwin}_{key}{tag}.npz')
    meta = {'name': os.path.splitext(os.path.basename(out))[0], 'family': 'f1',
            'cell_px': args.cell_px, 'zwin': args.zwin, 'params': params}
    common.save_features(out, F, chs, sts, np.concatenate(crs),
                         np.concatenate(ccs), meta)


if __name__ == '__main__':
    main()
