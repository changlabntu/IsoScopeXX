"""Shared plumbing for the clustering/ material-texture pipeline (PLAN.md).

Three things live here and nowhere else:

1. Codec geometry — chunk/stem/cell coordinate conventions matching the
   thx10codec layout (`z{a:04d}-{b:04d}/{rrr}{ccc}.npz`, 384-px patches) and
   the regis2 (z, x, y) frame (`x = (col+0.5)*patch`, same as gaps.csv).
2. The params-keyed cache scheme (fixes PLAN §8): every cache family is a dir
   `Output/clustering/cache/{family}-{md5(params)[:10]}/` holding a
   human-readable params.json plus per-chunk npz files. A loader that finds a
   dir whose params.json disagrees with the requested params raises instead of
   silently reusing it.
3. The standard feature-matrix interface consumed by harness.py: one npz with
   feat (N, D) float32 plus row keys chunk/stem/cell_row/cell_col and a meta
   JSON string. Patch-level features use cell_row = cell_col = -1.
"""
import hashlib
import json
import os
import sys
from collections import namedtuple
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

CODEC = '/media/cheese/Ghc_data3/THX10/thx10codec/codec'
OUT = '/home/cheese/workspace/Output/clustering'
CACHE = os.path.join(OUT, 'cache')
PATCH = 384
CELL = 48


def ncell(cell_px=CELL):
    """Cells per patch edge (8 at 48 px, 4 at 96 px)."""
    assert PATCH % cell_px == 0, f'cell_px {cell_px} must divide {PATCH}'
    return PATCH // cell_px


def list_chunks(codec=CODEC):
    return sorted(d for d in os.listdir(codec)
                  if os.path.isdir(os.path.join(codec, d)))


def chunk_z(chunk):
    """'z0240-0288' -> (za, zb, zmid)."""
    za, zb = (int(p) for p in chunk[1:].split('-'))
    return za, zb, (za + zb) // 2


def list_stems(codec, chunk):
    return sorted(os.path.splitext(os.path.basename(f))[0]
                  for f in glob(os.path.join(codec, chunk, '*.npz')))


_FAST_CACHE = {}


def _fast_store(codec, chunk):
    """Memory-mapped per-chunk uint8 arrays written by codec_fast.py, or None.
    Ignored unless {codec_parent}/fast/params.json points at this codec."""
    key = (codec, chunk)
    if key in _FAST_CACHE:
        return _FAST_CACHE[key]
    res = None
    root = os.path.join(os.path.dirname(os.path.abspath(codec)), 'fast')
    d = os.path.join(root, chunk)
    pj = os.path.join(root, 'params.json')
    if os.path.exists(os.path.join(d, 'stems.npy')) and os.path.exists(pj):
        with open(pj) as f:
            src_ok = json.load(f).get('source') == os.path.abspath(codec)
        if src_ok:
            stems = np.load(os.path.join(d, 'stems.npy')).astype(str)
            idx = {s: i for i, s in enumerate(stems)}
            arrs = []
            for k in range(16):
                p = os.path.join(d, f'scale_{k}.npy')
                if not os.path.exists(p):
                    break
                arrs.append(np.load(p, mmap_mode='r'))
            if arrs:
                res = (idx, arrs)
    _FAST_CACHE[key] = res
    return res


def load_codes(codec, chunk, stem):
    """[(Z, hk, wk) int32 per scale, coarse->fine] for one patch. Reads the
    codec_fast.py store when present (no decompression), else the npz."""
    fast = _fast_store(codec, chunk)
    if fast is not None:
        idx, arrs = fast
        i = idx.get(stem)
        if i is not None:
            return [np.asarray(a[i], dtype=np.int32) for a in arrs]
    z = np.load(os.path.join(codec, chunk, stem + '.npz'))
    keys = sorted((k for k in z.files if k.startswith('scale_')),
                  key=lambda k: int(k.split('_')[1]))
    return [z[k][0] for k in keys]


def load_norm(codec, chunk):
    with open(os.path.join(codec, chunk, 'norm_params.json')) as f:
        return json.load(f)


def stem_rc(stem):
    return int(stem[:3]), int(stem[3:6])


def patch_xyz(chunk, stem, patch=PATCH):
    """Patch-centre (z, x, y) in the codec zarr frame (gaps.csv convention)."""
    r, c = stem_rc(stem)
    return chunk_z(chunk)[2], (c + 0.5) * patch, (r + 0.5) * patch


def cell_xyz(chunk, stem, cr, cc, cell_px=CELL, patch=PATCH):
    """Cell-centre (z, x, y); cr/cc = -1 falls back to the patch centre."""
    if cr < 0:
        return patch_xyz(chunk, stem, patch)
    r, c = stem_rc(stem)
    return (chunk_z(chunk)[2],
            c * patch + (cc + 0.5) * cell_px,
            r * patch + (cr + 0.5) * cell_px)


def params_key(params):
    blob = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(blob.encode()).hexdigest()[:10]


def cache_dir(family, params):
    """The cache dir for (family, params); created with params.json on first
    use, verified against params.json on every later use."""
    d = os.path.join(CACHE, f'{family}-{params_key(params)}')
    pj = os.path.join(d, 'params.json')
    if os.path.isdir(d):
        with open(pj) as f:
            stored = json.load(f)
        if stored != params:
            raise RuntimeError(f'cache key collision in {d}:\n'
                               f'stored {stored}\nwanted {params}')
    else:
        os.makedirs(d)
        with open(pj, 'w') as f:
            json.dump(params, f, indent=2, sort_keys=True)
    return d


FeatureSet = namedtuple('FeatureSet', 'feat chunk stem cell_row cell_col meta')


def save_features(path, feat, chunk, stem, cell_row, cell_col, meta):
    """Write a harness-consumable feature npz. meta must be JSON-serializable
    and include at least name/family/cell_px/params."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path,
                        feat=np.asarray(feat, np.float32),
                        chunk=np.asarray(chunk),
                        stem=np.asarray(stem),
                        cell_row=np.asarray(cell_row, np.int16),
                        cell_col=np.asarray(cell_col, np.int16),
                        meta=json.dumps(meta))
    print(f'wrote {path}  feat {np.asarray(feat).shape} '
          f'({meta.get("family", "?")})', flush=True)


def load_feature_set(path):
    z = np.load(path, allow_pickle=False)
    return FeatureSet(z['feat'], z['chunk'].astype(str), z['stem'].astype(str),
                      z['cell_row'], z['cell_col'], json.loads(str(z['meta'])))


def read_mid_slice(codec, chunk):
    """(slice_yx float32, zmid) via the chunk's recorded zarr source."""
    from inference.zarr_io import open_zarr
    np0 = load_norm(codec, chunk)
    za, zb = np0['z_range']
    arr = open_zarr(np0['source'], np0['zarr_level'])
    zmid = (za + zb) // 2
    return np.asarray(arr[zmid].read().result()).T.astype(np.float32), zmid


def raw_cell_stats(codec, chunk, cell_px=CELL):
    """Cached per-cell std/mean of the chunk's raw mid slice.

    Returns dict(stems (S,) <U6, cell_std (S, n, n), cell_mean (S, n, n),
    patch_std (S,), patch_mean (S,)) with n = cells per patch edge. Shared by
    the harness amp-leak metric, the overlay content gate, and F5.
    """
    params = {'family': 'rawstats', 'codec': codec, 'cell_px': cell_px,
              'version': 1}
    d = cache_dir('rawstats', params)
    cache = os.path.join(d, f'{chunk}.npz')
    if os.path.exists(cache):
        return dict(np.load(cache))
    sl, _ = read_mid_slice(codec, chunk)
    stems = list_stems(codec, chunk)
    n = ncell(cell_px)
    cs = np.empty((len(stems), n, n), np.float32)
    cm = np.empty_like(cs)
    ps = np.empty(len(stems), np.float32)
    pm = np.empty_like(ps)
    for i, st in enumerate(stems):
        r, c = stem_rc(st)
        reg = sl[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
        cells = reg.reshape(n, cell_px, n, cell_px)
        cs[i] = cells.std(axis=(1, 3))
        cm[i] = cells.mean(axis=(1, 3))
        ps[i], pm[i] = reg.std(), reg.mean()
    out = dict(stems=np.array(stems), cell_std=cs, cell_mean=cm,
               patch_std=ps, patch_mean=pm)
    np.savez_compressed(cache, **out)
    print(f'rawstats {chunk}: cached {len(stems)} patches', flush=True)
    return out


def cell_gate(stats, gate_k=2.0):
    """(S, n, n) bool — cell-level raw-contrast gate, per-chunk noise floor =
    median cell std of the darkest-decile cells (latent_overlay's patch gate,
    at cell granularity)."""
    cm, cs = stats['cell_mean'], stats['cell_std']
    dark = cm <= np.percentile(cm, 10)
    floor = np.median(cs[dark])
    return cs > gate_k * floor


def row_gate(fs, codec, cell_px, gate_k):
    """Per-row content gate for a FeatureSet (patch-level rows gate on the
    whole-patch std against the same cell-derived floor)."""
    keep = np.zeros(len(fs.feat), bool)
    for ch in sorted(set(fs.chunk)):
        stats = raw_cell_stats(codec, ch, cell_px)
        gate = cell_gate(stats, gate_k)
        stem_i = {s: i for i, s in enumerate(stats['stems'])}
        m = fs.chunk == ch
        si = np.array([stem_i[s] for s in fs.stem[m]])
        cr = np.maximum(fs.cell_row[m], 0)
        cc = np.maximum(fs.cell_col[m], 0)
        keep[m] = gate[si, cr, cc]
        patch_rows = fs.cell_row[m] < 0
        if patch_rows.any():
            cm, cs = stats['cell_mean'], stats['cell_std']
            dark = cm <= np.percentile(cm, 10)
            floor = np.median(cs[dark])
            keep_m = keep[m]
            keep_m[patch_rows] = stats['patch_std'][si[patch_rows]] > gate_k * floor
            keep[m] = keep_m
    return keep
