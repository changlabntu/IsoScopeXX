"""M2a — metacodes (PLAN §5 F1 stage 1): cluster the 256 codebook entries into
K ~ 24-32 metacodes per scale, so per-cell histograms are dense enough to
compare.

Each codebook vector (R^4, quantizers[k].embedding.weight) is optionally
augmented with its mean in-plane 3x3 usage context (the mean embedding of the
8 neighbours of every occurrence, sampled over --n_patches mid slices) — usage
context is what makes two interchangeable codes similar even when their raw
embeddings differ. Columns standardized, KMeans(seed 0).

Cached under Output/clustering/cache/metacodes-{key}/ (assign_s{k}.npy int8,
centers_s{k}.npy, params.json). Import build_metacodes() from feat_f1/f2, or
run standalone:

  python clustering/metacodes.py --K 24 --scales 2,3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from clustering import common

QUANT_PTH = ('/home/cheese/workspace/logs/skipU/roiD192gfC/max5skip4/'
             'checkpoints/20260716_055927/quantizers_model_epoch_300.pth')
N_EMBED = 256


def load_codebooks(model='skipU', epoch=300):
    """[(256, 4) float32 per scale] — direct CPU torch.load of the pickled
    quantizers ModuleList (falls back to Engine.from_registry)."""
    import torch
    try:
        ml = torch.load(QUANT_PTH, map_location='cpu', weights_only=False)
    except Exception:
        from inference import Engine
        ml = Engine.from_registry(model, epoch=epoch, device='cpu').gan.quantizers
    return [q.embedding.weight.detach().cpu().numpy().astype(np.float32)
            for q in ml]


def context_embedding(codec, scale, embed, n_patches=2000, seed=0):
    """(256, 4) mean embedding of each code's 8 in-plane neighbours, sampled
    over the mid slice of n_patches random patches. Codes never seen keep
    their own embedding (context == self)."""
    rng = np.random.RandomState(seed)
    recs = [(ch, st) for ch in common.list_chunks(codec)
            for st in common.list_stems(codec, ch)]
    sel = [recs[i] for i in rng.choice(len(recs), min(n_patches, len(recs)),
                                       replace=False)]
    acc = np.zeros((N_EMBED, embed.shape[1]), np.float64)
    cnt = np.zeros(N_EMBED, np.int64)
    offs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dy, dx) != (0, 0)]
    for ch, st in sel:
        g = common.load_codes(codec, ch, st)[scale]
        g = g[g.shape[0] // 2]
        for dy, dx in offs:
            ctr = g[max(0, -dy):g.shape[0] - max(0, dy),
                    max(0, -dx):g.shape[1] - max(0, dx)].ravel()
            nb = g[max(0, dy):g.shape[0] + min(0, dy),
                   max(0, dx):g.shape[1] + min(0, dx)].ravel()
            np.add.at(acc, ctr, embed[nb])
            np.add.at(cnt, ctr, 1)
    ctx = embed.copy().astype(np.float64)
    seen = cnt > 0
    ctx[seen] = acc[seen] / cnt[seen, None]
    return ctx.astype(np.float32)


def build_metacodes(codec=common.CODEC, scales=(2, 3), K=24, context=True,
                    n_patches=2000, model='skipU', epoch=300):
    """dict scale -> (256,) int8 metacode assignment (cached)."""
    params = {'family': 'metacodes', 'codec': codec, 'model': model,
              'epoch': epoch, 'scales': list(scales), 'K': K,
              'context': context, 'n_patches': n_patches, 'version': 1}
    d = common.cache_dir('metacodes', params)
    paths = {s: os.path.join(d, f'assign_s{s}.npy') for s in scales}
    if all(os.path.exists(p) for p in paths.values()):
        print(f'metacodes cache hit {os.path.basename(d)}', flush=True)
        return {s: np.load(p) for s, p in paths.items()}
    from sklearn.cluster import KMeans
    books = load_codebooks(model, epoch)
    out = {}
    for s in scales:
        X = books[s]
        if context:
            print(f's{s}: sampling 3x3 context over {n_patches} patches...',
                  flush=True)
            X = np.concatenate([X, context_embedding(codec, s, books[s],
                                                     n_patches)], 1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-09)
        km = KMeans(K, n_init=10, random_state=0).fit(X)
        out[s] = km.labels_.astype(np.int8)
        np.save(paths[s], out[s])
        np.save(os.path.join(d, f'centers_s{s}.npy'),
                km.cluster_centers_.astype(np.float32))
        sizes = np.bincount(km.labels_, minlength=K)
        print(f's{s}: K={K} metacodes, cluster sizes '
              f'{sizes.min()}..{sizes.max()}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--codec', default=common.CODEC)
    ap.add_argument('--K', type=int, default=24)
    ap.add_argument('--scales', default='2,3')
    ap.add_argument('--no-context', action='store_true')
    ap.add_argument('--n_patches', type=int, default=2000)
    ap.add_argument('--model', default='skipU')
    ap.add_argument('--epoch', type=int, default=300)
    args = ap.parse_args()
    build_metacodes(args.codec, tuple(int(s) for s in args.scales.split(',')),
                    args.K, not args.no_context, args.n_patches,
                    args.model, args.epoch)


if __name__ == '__main__':
    main()
