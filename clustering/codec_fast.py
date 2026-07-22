"""One-time codec consolidation: compressed per-patch npz -> per-chunk
memory-mappable uint8 .npy ("fast store"), so repeated analytics stop paying
zlib decompression + 13,838 file opens on every pass.

Layout (derived cache — the original codec/ stays the source of truth):
  {codec_parent}/fast/params.json           source path, dtype, model/epoch
  {codec_parent}/fast/{chunk}/stems.npy     (S,) patch stems, list_stems order
  {codec_parent}/fast/{chunk}/scale_{k}.npy (S, Z, hk, wk) uint8

common.load_codes() automatically prefers the fast store when it exists and
its params.json source matches, so every existing script speeds up unchanged.
Each converted chunk is verified against N random original npz before the
next chunk starts.

  python clustering/codec_fast.py [--codec ...] [--chunks a,b] [--verify 8]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from clustering import common


def convert_chunk(codec, chunk, out_root, verify=8, rng=None):
    out_dir = os.path.join(out_root, chunk)
    stems = common.list_stems(codec, chunk)
    stems_f = os.path.join(out_dir, 'stems.npy')
    if os.path.exists(stems_f) and len(np.load(stems_f)) == len(stems):
        print(f'{chunk}: fast store exists, skip', flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    first = common.load_codes(codec, chunk, stems[0])
    arrs = [np.empty((len(stems),) + g.shape, np.uint8) for g in first]
    for i, st in enumerate(stems):
        for a, g in zip(arrs, common.load_codes(codec, chunk, st)):
            assert g.min() >= 0 and g.max() < 256, f'{chunk}/{st} not uint8'
            a[i] = g
    for k, a in enumerate(arrs):
        np.save(os.path.join(out_dir, f'scale_{k}.npy'), a)
    np.save(stems_f, np.array(stems))
    rng = rng or np.random.RandomState(0)
    mm = [np.load(os.path.join(out_dir, f'scale_{k}.npy'), mmap_mode='r')
          for k in range(len(arrs))]
    for i in rng.choice(len(stems), min(verify, len(stems)), replace=False):
        orig = common.load_codes(codec, chunk, stems[i])
        for a, g in zip(mm, orig):
            assert np.array_equal(np.asarray(a[i], np.int32), g), \
                f'verify FAILED {chunk}/{stems[i]}'
    print(f'{chunk}: {len(stems)} patches -> fast store (verified '
          f'{min(verify, len(stems))})', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--codec', default=common.CODEC)
    ap.add_argument('--chunks', default=None)
    ap.add_argument('--verify', type=int, default=8)
    args = ap.parse_args()
    codec = os.path.abspath(args.codec)
    out_root = os.path.join(os.path.dirname(codec), 'fast')
    os.makedirs(out_root, exist_ok=True)
    pj = os.path.join(out_root, 'params.json')
    if not os.path.exists(pj):
        norm = common.load_norm(codec, common.list_chunks(codec)[0])
        with open(pj, 'w') as f:
            json.dump({'source': codec, 'dtype': 'uint8', 'version': 1,
                       'model': norm.get('model'), 'epoch': norm.get('epoch')},
                      f, indent=2)
    chunks = args.chunks.split(',') if args.chunks else common.list_chunks(codec)
    for ch in chunks:
        convert_chunk(codec, ch, out_root, args.verify)
    print(f'fast store at {out_root}', flush=True)


if __name__ == '__main__':
    main()
