"""M1 / F4 — cells-from-cache baseline (PLAN §5): the free feature every code
statistic must beat.

Reshapes the existing latent_overlay caches
(Output/regis2/cluster_fable/feat_zw3_z*.npz: fps (814, 1024) = 4 scales x
C4 x 8 x 8 per patch) into one 16-dim vector per 48-px cell (the 8x8 pooling
grid maps to 48-px cells exactly): feat[cell i,j] = concat over scales of
fps_scale[:, i, j]. Still mean-pooled continuous latent — the honest baseline.

  python clustering/feat_f4.py [--scales all|fine] [--out ...]

No GPU, no new caches; consumes the old-style caches as-is (they predate the
params-hash scheme; built with skipU/300, spatial=8, zwin=3 — recorded in
meta rather than re-verified).
"""
import argparse
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from clustering import common

FPS_DIR = '/home/cheese/workspace/Output/regis2/cluster_fable'
C, S = 4, 8


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fps_dir', default=FPS_DIR)
    ap.add_argument('--zwin', type=int, default=3)
    ap.add_argument('--scales', choices=('all', 'fine'), default='all',
                    help='fine = s2+s3 blocks only (8-dim cells)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(common.OUT, f'f4_cells_{args.scales}.npz')
    caches = sorted(glob(os.path.join(args.fps_dir,
                                      f'feat_zw{args.zwin}_z*.npz')))
    if not caches:
        sys.exit(f'no feat_zw{args.zwin}_*.npz in {args.fps_dir} — run '
                 'clustering/latent_overlay.py once to build them')
    feats, chs, sts, crs, ccs = [], [], [], [], []
    for path in caches:
        chunk = os.path.basename(path)[len(f'feat_zw{args.zwin}_'):-4]
        z = np.load(path, allow_pickle=True)
        stems, fps = list(z['stems']), z['fps']
        blocks = [tuple(b) for b in z['blocks']]
        keep = blocks if args.scales == 'all' else blocks[2:]
        g = np.stack([fps[:, a:b].reshape(-1, C, S, S) for a, b in keep], 1)
        for i in range(S):
            for j in range(S):
                feats.append(g[:, :, :, i, j].reshape(len(stems), -1))
                chs += [chunk] * len(stems)
                sts += list(stems)
                crs += [i] * len(stems)
                ccs += [j] * len(stems)
        print(f'{chunk}: {len(stems)} patches -> {len(stems) * S * S} cells',
              flush=True)
    meta = {'name': os.path.splitext(os.path.basename(out))[0],
            'family': 'f4', 'cell_px': common.PATCH // S,
            'params': {'source': 'feat_zw%d caches (skipU/300, spatial=8, '
                                 'pre-params-hash)' % args.zwin,
                       'scales': args.scales, 'zwin': args.zwin}}
    common.save_features(out, np.concatenate(feats), chs, sts, crs, ccs, meta)


if __name__ == '__main__':
    main()
