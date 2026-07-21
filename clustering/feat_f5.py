"""M5 / F5 — raw-image control (PLAN §5): per-cell uniform LBP histograms
from the raw mid slices, no codec involved. Decides the interpretation of the
code features: if F5 separates the anchor classes and the code features
don't, the codec entangles material texture; if neither does, the material is
a microstructural continuum at this scale (a legitimate terminal answer).

Two LBP configs per cell — (P=8, R=1) 10 bins + (P=16, R=2) 18 bins — each
L1-normalized then sqrt (Hellinger), concatenated to 28 dims. Mid slice only
(cheap by design; the z-window is a latent-feature concern).

  python clustering/feat_f5.py [--cell_px 48] [--chunks a,b]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from clustering import common

CONFIGS = ((8, 1), (16, 2))


def chunk_lbp(codec, chunk, cell_px, cache_d):
    cache = os.path.join(cache_d, f'{chunk}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return list(z['stems']), z['hist']
    from skimage.feature import local_binary_pattern
    sl, _ = common.read_mid_slice(codec, chunk)
    stems = common.list_stems(codec, chunk)
    n = common.ncell(cell_px)
    maps = [local_binary_pattern(sl, P, R, 'uniform').astype(np.int32)
            for P, R in CONFIGS]
    nb = [P + 2 for P, _ in CONFIGS]
    cell_id = np.arange(common.PATCH) // cell_px
    cid = cell_id[:, None] * n + cell_id[None, :]
    hist = np.zeros((len(stems), n, n, sum(nb)), np.float32)
    for i, st in enumerate(stems):
        r, c = common.stem_rc(st)
        off = 0
        for mp, b in zip(maps, nb):
            reg = mp[r * common.PATCH:(r + 1) * common.PATCH,
                     c * common.PATCH:(c + 1) * common.PATCH]
            h = np.bincount((cid * b + reg).ravel(),
                            minlength=n * n * b).reshape(n, n, b)
            hist[i, :, :, off:off + b] = h
            off += b
    np.savez_compressed(cache, stems=np.array(stems), hist=hist)
    print(f'f5 {chunk}: {len(stems)} patches', flush=True)
    return stems, hist


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--codec', default=common.CODEC)
    ap.add_argument('--cell_px', type=int, default=common.CELL)
    ap.add_argument('--chunks', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    params = {'family': 'f5', 'codec': args.codec, 'cell_px': args.cell_px,
              'configs': [list(c) for c in CONFIGS], 'version': 1}
    cache_d = common.cache_dir('f5', params)
    chunks = args.chunks.split(',') if args.chunks else common.list_chunks(args.codec)
    n = common.ncell(args.cell_px)
    nb = [P + 2 for P, _ in CONFIGS]
    feats, chs, sts, crs, ccs = [], [], [], [], []
    for ch in chunks:
        stems, hist = chunk_lbp(args.codec, ch, args.cell_px, cache_d)
        h = hist.reshape(-1, sum(nb))
        off, parts = 0, []
        for b in nb:
            blk = h[:, off:off + b]
            parts.append(np.sqrt(blk / (blk.sum(1, keepdims=True) + 1e-09)))
            off += b
        feats.append(np.concatenate(parts, 1).astype(np.float32))
        for st in stems:
            chs += [ch] * (n * n)
            sts += [st] * (n * n)
        rr, cc2 = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        crs.append(np.tile(rr.ravel(), len(stems)))
        ccs.append(np.tile(cc2.ravel(), len(stems)))
    tag = '' if args.chunks is None else '_sub'
    out = args.out or os.path.join(common.OUT, f'f5_lbp{tag}.npz')
    meta = {'name': os.path.splitext(os.path.basename(out))[0], 'family': 'f5',
            'cell_px': args.cell_px, 'params': params}
    common.save_features(out, np.concatenate(feats), chs, sts,
                         np.concatenate(crs), np.concatenate(ccs), meta)


if __name__ == '__main__':
    main()
