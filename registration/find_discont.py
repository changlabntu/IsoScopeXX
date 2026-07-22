"""Mine REAL patches for genuine Z-discontinuities: latent screen -> pixel verify.

    CUDA_VISIBLE_DEVICES=0 python registration/find_discont.py --control

Screens all stored codecs (no re-encode: Engine.latents_from_indices) for
sparse slice-to-slice "jump" spikes, ranks patches by a MAD-self-normalized
spike score, then numerically verifies the top-N candidates on the RAW tifs:

  GEOMETRIC  a true discontinuity — large, offset-consistent phase-corr shift
             whose alignment recovers most of the pair-NCC gap to the patch's
             own baseline
  CONTENT    NCC drop with no coherent recoverable shift (structure change)
  NOISE      neither (below thresholds)

Scoring: spike_t = (max(t1) - median(t1)) / (MAD(t1) + eps) over the 31
adjacent-latent-plane phase-corr magnitudes — a chunk-type event is one hot
pair among quiet ones, so max-median concentrates it, and the per-patch MAD
normalization collapses the weak-texture false-positive mode (erratic t on
near-black patches -> high MAD -> low score). Gates: t_max below --t_min_px
(image px) or latent texture below the population p5 (--tex_min overrides).

Pitfalls: discontinuities exactly BETWEEN 32-slice CCC chunks are invisible
within patches; a real acquisition-chunk boundary shows up as the same z* in
many patches sharing CCC (the report prints that consensus histogram).
Convention everywhere: phase_corr_shift(fixed=later, moving=earlier) — d
moves the earlier slice's content forward, and feeds params_to_mat directly
as the forward warp of the earlier slice.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402

from registration import affine, enhance, features  # noqa: E402
from registration.evaluate import ncc  # noqa: E402
from registration.register import phase_corr_shift  # noqa: E402
from registration.perturb import apply  # noqa: E402

DEFAULT_CODEC = '/home/cheese/workspace/Output/skipU300/codec'
DEFAULT_RAW = '/home/cheese/workspace/Data/thx10/roiAdsp4'
DEFAULT_OUT = '/home/cheese/workspace/Output/regis2/find_discont'
STRIDE = 8
EPS_MAD = 0.05     # latent px floor inside the spike normalization

FIELDS = ['stem', 'ccc', 'score', 'spike_t', 'spike_ncc', 'coloc', 'z_star',
          'dx_px', 'dy_px', 't_max_px', 't_med_px', 't_mad_px', 'ncc_min',
          'ncc_med', 'coh2_px', 'tex', 'gated', 'edge']


def discover(codec_dir, limit=None, chunks=None):
    """npz list + stems. Supports both flat layouts ({stem}.npz) and
    encode_stack's chunked layout (z{a}-{b}/{rrr}{ccc}.npz); `chunks`
    (comma-separated dir names) restricts to those z-chunks."""
    subs = sorted(d for d in glob(os.path.join(codec_dir, 'z*-*'))
                  if os.path.isdir(d))
    if chunks:
        want = chunks.split(',')
        missing = [c for c in want if not any(os.path.basename(d) == c for d in subs)]
        if missing:
            raise FileNotFoundError(f'chunk(s) {missing} not under {codec_dir}; '
                                    f'have {[os.path.basename(d) for d in subs]}')
        subs = [d for d in subs if os.path.basename(d) in want]
    if subs:
        files = [f for d in subs for f in sorted(glob(os.path.join(d, '*.npz')))]
    else:
        files = sorted(glob(os.path.join(codec_dir, '*.npz')))
    files = files[:limit]
    stems = [os.path.splitext(os.path.relpath(f, codec_dir))[0] for f in files]
    return files, stems


def group_label(stem):
    """Consensus grouping key: the z-chunk dir (chunked layout) or CCC
    (9-digit roiAdsp4 stems) — patches sharing it share absolute Z."""
    d, base = os.path.split(stem)
    if d:
        return d
    return base[6:9] if len(base) == 9 and base.isdigit() else '?'


def load_raw(stem, args):
    """Raw (Z, Y, X) [-1, 1] patch for verification: tif (flat layout) or a
    window-normalized block from the source OME-Zarr (chunked layout; patch
    size and z-range come from norm_params.json / the chunk dir name)."""
    if args.zarr:
        from inference.zarr_io import open_zarr
        d, base = os.path.split(stem)
        za, zb = (int(v) for v in d[1:].split('-'))
        row, col = int(base[:3]), int(base[3:6])
        p = args.patch
        arr = open_zarr(args.zarr, args.zarr_level)
        blk = arr[za:zb, col * p:(col + 1) * p,
                  row * p:(row + 1) * p].read().result()
        v = blk.astype(np.float32).transpose(0, 2, 1)      # (z,x,y) -> (Z,Y,X)
        lo, hi = args.window
        return np.clip((v - lo) / (hi - lo), 0, 1) * 2 - 1
    return tifffile.imread(os.path.join(args.raw, stem + '.tif')).astype(np.float32)


def load_planes(eng, npz_path):
    """Codec npz -> summed latent planes (Z, C, h, w) float32 on device."""
    z = np.load(npz_path)
    keys = sorted((k for k in z.files if k.startswith('scale_')),
                  key=lambda k: int(k.split('_')[1]))
    idx = [torch.from_numpy(z[k].astype(np.int64)).to(eng.device) for k in keys]
    return sum(eng.latents_from_indices(idx))[0].permute(3, 0, 1, 2).float()


def pair_signals(planes, offsets=(1, 2)):
    """Per-pair phase-corr shifts/magnitudes (latent px), adjacent NCC, and
    the patch texture statistic (median per-plane latent std)."""
    Z = planes.shape[0]
    sig = {}
    for o in offsets:
        d = np.array([phase_corr_shift(planes[z + o], planes[z])
                      for z in range(Z - o)], dtype=np.float64)
        sig[f'd{o}'], sig[f't{o}'] = d, np.hypot(d[:, 0], d[:, 1])
    flat = planes.cpu().numpy().reshape(Z, -1)
    sig['ncc1'] = np.array([ncc(flat[z], flat[z + 1]) for z in range(Z - 1)])
    sig['tex'] = float(np.median(planes.std(dim=(1, 2, 3)).cpu().numpy()))
    return sig


def screen_patch(stem, sig, t_min_px):
    """Spike score + evidence columns for one patch (tex gate applied later)."""
    t1, d1, d2, ncc1 = sig['t1'], sig['d1'], sig['d2'], sig['ncc1']
    med = float(np.median(t1))
    mad = float(np.median(np.abs(t1 - med)))
    z_star = int(np.argmax(t1))
    spike_t = float((t1[z_star] - med) / (mad + EPS_MAD))
    mad_ncc = float(np.median(np.abs(ncc1 - np.median(ncc1))))
    spike_ncc = float((np.median(ncc1) - ncc1.min()) / (mad_ncc + 1e-3))
    coloc = int(abs(int(np.argmin(ncc1)) - z_star) <= 1)
    edge = int(z_star == 0 or z_star == len(t1) - 1)
    if 1 <= z_star and z_star - 1 < len(d2):
        r = d2[z_star - 1] - (d1[z_star - 1] + d1[z_star])
    elif z_star + 1 < len(d1):                       # z_star == 0
        r = d2[z_star] - (d1[z_star] + d1[z_star + 1])
    else:
        r = np.array([np.nan, np.nan])
    coh2_px = float(STRIDE * np.hypot(r[0], r[1]))
    return dict(stem=stem, ccc=group_label(stem),
                score=spike_t, spike_t=spike_t, spike_ncc=spike_ncc,
                coloc=coloc, z_star=z_star,
                dx_px=float(STRIDE * d1[z_star, 0]),
                dy_px=float(STRIDE * d1[z_star, 1]),
                t_max_px=float(STRIDE * t1[z_star]),
                t_med_px=float(STRIDE * med), t_mad_px=float(STRIDE * mad),
                ncc_min=float(ncc1.min()), ncc_med=float(np.median(ncc1)),
                coh2_px=coh2_px, tex=sig['tex'],
                gated=int(STRIDE * t1[z_star] < t_min_px), edge=edge)


def verify_candidate(vol, rec, args, device):
    """Raw-pixel verification at the suspect pair (see module docstring)."""
    Z = vol.shape[0]
    a = rec['z_star']
    b = a + 1
    t = torch.from_numpy(vol)[:, None].to(device)    # (Z, 1, 256, 256)
    with torch.inference_mode():
        d1 = np.array(phase_corr_shift(t[b], t[a]))
        d1_mag = float(np.hypot(*d1))
        d_agree = float(np.hypot(d1[0] - rec['dx_px'], d1[1] - rec['dy_px']))
        resids = []
        if a - 1 >= 0:
            d01 = np.array(phase_corr_shift(t[a], t[a - 1]))
            d02 = np.array(phase_corr_shift(t[b], t[a - 1]))
            resids.append(float(np.hypot(*(d02 - (d01 + d1)))))
        if b + 1 < Z:
            d0a = np.array(phase_corr_shift(t[b + 1], t[a]))
            d0b = np.array(phase_corr_shift(t[b + 1], t[b]))
            resids.append(float(np.hypot(*(d0a - (d1 + d0b)))))
        resid = max(resids) if resids else float('nan')

        m = max(4, int(np.ceil(d1_mag)) + 2)
        crop = np.s_[m:-m, m:-m]
        ncc_raw = ncc(vol[a][crop], vol[b][crop])
        mat = affine.params_to_mat(0.0, d1[0], d1[1], 1.0)
        aw = affine.warp_slices(t[a][None], mat[None], mode='bicubic',
                                fill=-1.0)[0, 0].cpu().numpy()
        ncc_aligned = ncc(aw[crop], vol[b][crop])
    base = np.array([ncc(vol[z][crop], vol[z + 1][crop])
                     for z in range(Z - 1) if z != a])
    ncc_base = float(np.median(base))
    mad_base = float(np.median(np.abs(base - ncc_base)))
    recovery = float(np.clip((ncc_aligned - ncc_raw)
                             / max(ncc_base - ncc_raw, 1e-6), -1.0, 2.0))
    if (d1_mag >= args.d_min_px and recovery >= args.rec_min
            and (np.isnan(resid) or resid <= args.cons_max_px)):
        verdict = 'GEOMETRIC'
    elif ncc_raw <= ncc_base - max(args.gap_min, 3 * mad_base):
        verdict = 'CONTENT'
    else:
        verdict = 'NOISE'
    return dict(rec, d1x_px=float(d1[0]), d1y_px=float(d1[1]),
                d1_mag_px=d1_mag, d_agree_px=d_agree, resid_px=resid,
                ncc_raw=float(ncc_raw), ncc_aligned=float(ncc_aligned),
                ncc_base=ncc_base, recovery=recovery, verdict=verdict)


def gap_xyz(stem, z_star, patch):
    """Verified gap -> level-0 (x, y, z) in the source zarr's own axis frame
    (x = dim1 / ccc-indexed, y = dim2 / rrr-indexed): patch-center x/y, z = the
    earlier slice of the suspect pair (the gap sits between z and z+1).
    Chunked stems (z{a}-{b}/{rrr}{ccc}) only."""
    d, base = os.path.split(stem)
    za = int(d[1:].split('-')[0])
    row, col = int(base[:3]), int(base[3:6])
    return (int((col + 0.5) * patch), int((row + 0.5) * patch), za + z_star)


def merge_gaps(path, xyz):
    """Merge gap coordinates into the running x,y,z csv (dedup, z-sorted) so
    successive chunk scans keep appending to one whole-stack list."""
    have = set()
    if os.path.exists(path):
        with open(path) as f:
            have = {tuple(int(float(v)) for v in ln.split(','))
                    for ln in f.read().splitlines()[1:] if ln.strip()}
    n0 = len(have)
    have |= set(xyz)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'z'])
        for p in sorted(have, key=lambda t: (t[2], t[1], t[0])):
            w.writerow(p)
    print(f'-> {path} (+{len(have) - n0} new, {len(have)} gaps total)')


def montage(results, vols, out_png):
    """Horizontal layout: candidates as columns, rows = XZ | YZ | info; a red
    arrow at the left edge marks the suspect pair (no line across the data)."""
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(2.1 * n, 7.5), squeeze=False,
                             gridspec_kw=dict(height_ratios=[3, 3, 2]))
    colors = dict(GEOMETRIC='green', CONTENT='darkorange', NOISE='gray')
    for c, res in enumerate(results):
        vol = vols[res['stem']]
        lo, hi = np.percentile(vol, [1, 99.5])
        yb = (res['z_star'] + 1) * STRIDE
        for r, plane in enumerate((vol[:, vol.shape[1] // 2, :],
                                   vol[:, :, vol.shape[2] // 2])):
            ax = axes[r][c]
            ax.imshow(np.repeat(plane, STRIDE, axis=0), cmap='gray',
                      vmin=lo, vmax=hi)
            ax.annotate('', xy=(12, yb), xytext=(-30, yb),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.3),
                        annotation_clip=False)
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(('XZ (Y mid)', 'YZ (X mid)')[r], fontsize=9)
        ax = axes[2][c]
        ax.axis('off')
        ax.text(0.05, 0.95,
                f"{res['stem']}\nscore {res['score']:.1f}  z*={res['z_star']}\n"
                f"d1 ({res['d1x_px']:+.2f}, {res['d1y_px']:+.2f}) px\n"
                f"resid {res['resid_px']:.2f}\n"
                f"ncc {res['ncc_raw']:.3f}->{res['ncc_aligned']:.3f}\n"
                f"(base {res['ncc_base']:.3f})\n"
                f"recovery {res['recovery']:.2f}\n{res['verdict']}",
                fontsize=7, va='top', color=colors[res['verdict']])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'-> {out_png}')


def main():
    parser = argparse.ArgumentParser(
        description='Find real Z-discontinuities: latent screen -> pixel verify')
    parser.add_argument('--codec', default=DEFAULT_CODEC)
    parser.add_argument('--raw', default=DEFAULT_RAW,
                        help='Raw tif dir (flat layout)')
    parser.add_argument('--zarr', default=None,
                        help='Source OME-Zarr for chunked codecs (verification '
                             'loads patches from here instead of --raw; needs '
                             "py38zarr; default: the codec's norm_params.json "
                             "'source')")
    parser.add_argument('--zarr_level', default=None)
    parser.add_argument('--chunks', default=None,
                        help='Comma-separated z-chunk dirs to screen '
                             '(e.g. z0288-0336; default: all)')
    parser.add_argument('--patch', type=int, default=None,
                        help='Zarr patch size in px (default: norm_params.json)')
    parser.add_argument('--window', type=float, nargs=2, default=None,
                        metavar=('LO', 'HI'),
                        help='Intensity window for --zarr patches (default: '
                             "the codec's norm_params.json)")
    parser.add_argument('--out_dir', default=DEFAULT_OUT)
    parser.add_argument('--topn', type=int, default=50)
    parser.add_argument('--model', default='skipU')
    parser.add_argument('--t_min_px', type=float, default=2.0,
                        help='Screen gate: min spike magnitude (image px)')
    parser.add_argument('--tex_min', type=float, default=None,
                        help='Weak-texture gate (default: population p5)')
    parser.add_argument('--d_min_px', type=float, default=2.0)
    parser.add_argument('--rec_min', type=float, default=0.5)
    parser.add_argument('--cons_max_px', type=float, default=1.5)
    parser.add_argument('--gap_min', type=float, default=0.05)
    parser.add_argument('--gaps_file', default=None,
                        help='Running x,y,z csv of verified (non-NOISE) gaps, '
                             'merged across chunk scans (default for chunked '
                             "codecs: <codec>/../gaps.csv; 'off' disables)")
    parser.add_argument('--limit', type=int, default=None, help='Debug subset')
    parser.add_argument('--control', action='store_true',
                        help='Inject a known 5 px chunk step into the quietest '
                             'clean patch as a positive control')
    parser.add_argument('--control_stem', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    eng = features.get_engine(args.model)
    for nm in eng.gan.netg_names:
        getattr(eng.gan, nm).eval()
    device = eng.device

    files, stems = discover(args.codec, args.limit, args.chunks)
    np_json = sorted({os.path.join(os.path.dirname(f), 'norm_params.json')
                      for f in files})
    np_json = [p for p in np_json if os.path.dirname(p) != args.codec
               and os.path.exists(p)]
    if np_json:                       # chunked layout: verify from source zarr
        meta = json.load(open(np_json[0]))
        if args.zarr is None:
            args.zarr = meta.get('source')
        if args.zarr_level is None:
            args.zarr_level = meta.get('zarr_level', '0')
        if args.window is None:
            args.window = (meta['window_lo'], meta['window_hi'])
        if args.patch is None:
            args.patch = meta['patch']
        print(f"norm_params: zarr {args.zarr} level {args.zarr_level}, "
              f'patch {args.patch}, window {args.window}')
    if args.zarr_level is None:
        args.zarr_level = '0'
    print(f'screening {len(files)} codecs from {args.codec}')
    records = []
    with torch.inference_mode():
        for i, (f, stem) in enumerate(zip(files, stems)):
            records.append(screen_patch(stem, pair_signals(load_planes(eng, f)),
                                        args.t_min_px))
            if i % 256 == 0:
                print(f'  {i}/{len(files)}', flush=True)

    tex_min = (args.tex_min if args.tex_min is not None
               else float(np.percentile([r['tex'] for r in records], 5)))
    for r in records:
        if r['tex'] < tex_min:
            r['gated'] = 1
    print(f'tex gate: {tex_min:.4f} ({sum(r["tex"] < tex_min for r in records)} '
          'patches gated)')

    ctrl_vol = None
    if args.control:
        # quietest texture-valid patch (the t gate excludes everyone on clean
        # data — that must not break control selection)
        clean = [r for r in records if r['tex'] >= tex_min]
        stem0 = args.control_stem or min(clean, key=lambda r: r['score'])['stem']
        vol0 = tifffile.imread(os.path.join(args.raw, stem0 + '.tif')).astype(np.float32)
        Z = vol0.shape[0]
        zb, (tx, ty) = Z // 2, (4.0, -3.0)
        mats = np.stack([affine.identity()] * zb
                        + [affine.params_to_mat(0.0, tx, ty)] * (Z - zb))
        ctrl_vol = apply(vol0, mats, device)
        print(f'control: {stem0}+ctrl  truth: break z={zb - 1}->{zb}, '
              f'step ({tx}, {ty}) px')
        sl = enhance.encode_full(eng, ctrl_vol, zbatch=8)
        planes = sum(sl)[0].permute(3, 0, 1, 2).float().to(device)
        with torch.inference_mode():
            records.append(screen_patch(stem0 + '+ctrl', pair_signals(planes),
                                        args.t_min_px))

    records.sort(key=lambda r: r['score'], reverse=True)
    csv_path = os.path.join(args.out_dir, 'candidates.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v)
                        for k, v in r.items()})
    scores = np.array([r['score'] for r in records if r['tex'] >= tex_min])
    print(f'-> {csv_path}')
    print(f'score quantiles (tex-valid): p50 {np.percentile(scores, 50):.1f}  '
          f'p90 {np.percentile(scores, 90):.1f}  '
          f'p99 {np.percentile(scores, 99):.1f}  max {scores.max():.1f}')

    # top-N by score among texture-valid patches; the t gate stays as an info
    # flag (on clean data everything may be sub-threshold — verifying the top
    # scorers anyway IS the numerical investigation)
    top = [r for r in records if r['tex'] >= tex_min][:args.topn]
    print(f'\nverifying top {len(top)} on raw tifs ...')
    results, vols = [], {}
    for r in top:
        if r['stem'].endswith('+ctrl'):
            vol = ctrl_vol
        else:
            vol = load_raw(r['stem'], args)
        vols[r['stem']] = vol
        results.append(verify_candidate(vol, r, args, device))

    vcsv = os.path.join(args.out_dir, 'verified.csv')
    vfields = FIELDS + ['d1x_px', 'd1y_px', 'd1_mag_px', 'd_agree_px',
                        'resid_px', 'ncc_raw', 'ncc_aligned', 'ncc_base',
                        'recovery', 'verdict']
    with open(vcsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=vfields)
        w.writeheader()
        for r in results:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f'-> {vcsv}\n')
    print(f"{'stem':>15s} {'score':>6s} {'z*':>3s} {'|d1|px':>7s} {'resid':>6s} "
          f"{'ncc raw->aln (base)':>22s} {'rec':>5s}  verdict")
    for r in results:
        print(f"{r['stem']:>15s} {r['score']:6.1f} {r['z_star']:3d} "
              f"{r['d1_mag_px']:7.2f} {r['resid_px']:6.2f} "
              f"{r['ncc_raw']:8.3f}->{r['ncc_aligned']:.3f} ({r['ncc_base']:.3f})"
              f" {r['recovery']:5.2f}  {r['verdict']}")
    hist = Counter((r['ccc'], r['z_star']) for r in results)
    consensus = {k: v for k, v in hist.items() if v > 1}
    print(f'\n(ccc, z*) consensus in top-{len(top)}: '
          f'{consensus if consensus else "none"}')
    print(json.dumps(Counter(r['verdict'] for r in results), indent=None))

    montage(results, vols, os.path.join(args.out_dir, 'verify_montage.png'))

    if args.gaps_file is None and np_json:
        args.gaps_file = os.path.join(
            os.path.dirname(os.path.normpath(args.codec)), 'gaps.csv')
    gaps = [r for r in results if r['verdict'] != 'NOISE'
            and os.path.dirname(r['stem'])]
    if args.gaps_file and args.gaps_file != 'off' and gaps:
        merge_gaps(args.gaps_file,
                   [gap_xyz(r['stem'], r['z_star'], args.patch) for r in gaps])


if __name__ == '__main__':
    main()
