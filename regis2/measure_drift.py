"""Measure global per-slice drift across a whole-volume chunked codec.

    CUDA_VISIBLE_DEVICES=0 python regis2/measure_drift.py \
        --codec /media/cheese/Ghc_data3/THX10/thx10codec/codec

Complement to find_discont.py: that screen looks for sparse per-patch SPIKES
in the adjacent-latent-plane phase-corr series; this script integrates the
cross-patch MEDIAN of the same measurement into a whole-stack trajectory —
smooth stage drift that the spike score deliberately ignores.

Per chunk, a spatial subgrid of patches (--grid_step) is loaded from the
stored codec indices (no re-encode) and every adjacent pair's translation is
measured (register.phase_corr_shift, latent px * 8 -> image px). Aggregation
per absolute z:

  step        cross-patch MEDIAN (dx, dy) — robust to the regional gap
              events (they touch <5% of patches)
  coherence   R = |mean v| / mean |v| over contributing patches (1 = every
              patch sees the same vector = rigid shift; ~0 = incoherent =
              content noise). Real drift needs BOTH |step| above floor AND
              high R.
  selection   background patches are POISON here, and not because they are
              noisy: their latent planes carry the encoder's stationary
              fixed-pattern structure, so phase-corr locks to a confident
              ~0 shift — an artifactual "no motion" vote that pins the
              median wherever background is the majority. Patches are kept
              only if their find_discont tex (joined from --tex_csv_base's
              per-chunk candidates.csv) clears --tex_min, default 1.21 = the
              antimode of the strongly bimodal stack-wide tex distribution
              (background spike 1.188-1.200, textured tail above). Two
              filters tried first and rejected: a per-chunk tex PERCENTILE
              (cut falls below background level in background-heavy chunks
              -> chunk-wise 0.000 flatline) and exact-zero exclusion (there
              are no exact zeros — the lock is near-zero, not zero).

Caveats: translation only (no rotation); the 16 between-chunk pairs
(z = 47/48, ...) are unmeasurable within the codec and enter the cumulative
sum as 0-steps (marked); and a pairwise measurement cannot distinguish true
stage drift from genuinely slanted microstructure (the unobservable gauge) —
coherence separates rigid from content-driven, not stage from sample.

Outputs in --out_dir: drift_per_z.csv, drift.png, drift_raw.npz.
"""

import argparse
import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from registration import features  # noqa: E402
from registration.register import phase_corr_shift  # noqa: E402
from regis2.find_discont import discover, load_planes, STRIDE  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Global drift over a chunked codec')
    parser.add_argument('--codec', required=True)
    parser.add_argument('--out_dir',
                        default='/home/cheese/workspace/Output/regis2/find_discont_thx10')
    parser.add_argument('--model', default='skipU')
    parser.add_argument('--grid_step', type=int, default=2,
                        help='Take every Nth grid row/col per chunk (default 2)')
    parser.add_argument('--chunks', default=None)
    parser.add_argument('--raw_npz', default=None,
                        help='Re-aggregate from a saved drift_raw.npz instead '
                             'of re-measuring')
    parser.add_argument('--tex_csv_base', default=None,
                        help='Dir of per-chunk find_discont outputs whose '
                             'candidates.csv supply per-patch tex '
                             '(default: --out_dir)')
    parser.add_argument('--tex_min', type=float, default=1.21,
                        help='Absolute tex floor (stack-wide antimode; '
                             'see docstring)')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.raw_npz:
        raw = np.load(args.raw_npz, allow_pickle=True)
        all_za, all_d1 = raw['all_za'], raw['all_d1']
        all_stems = list(raw['all_stems'])
    else:
        eng = features.get_engine(args.model)
        for nm in eng.gan.netg_names:
            getattr(eng.gan, nm).eval()
        files, stems = discover(args.codec, chunks=args.chunks)
        picked = [(f, s) for f, s in zip(files, stems)
                  if int(os.path.basename(s)[:3]) % args.grid_step == 0
                  and int(os.path.basename(s)[3:6]) % args.grid_step == 0]
        print(f'{len(picked)}/{len(files)} patches (grid step {args.grid_step})')
        all_stems, all_za, all_d1 = [], [], []
        with torch.inference_mode():
            for i, (f, stem) in enumerate(picked):
                za = int(os.path.basename(os.path.dirname(stem))[1:].split('-')[0])
                planes = load_planes(eng, f)         # (Z, C, h, w)
                d1 = np.array([phase_corr_shift(planes[z + 1], planes[z])
                               for z in range(planes.shape[0] - 1)]) * STRIDE
                all_stems.append(stem), all_za.append(za), all_d1.append(d1)
                if i % 200 == 0:
                    print(f'  {i}/{len(picked)}', flush=True)
        all_za, all_d1 = np.array(all_za), np.array(all_d1)

    # selection: drop background patches (fixed-pattern zero-lock, see
    # docstring) via the find_discont tex column
    from glob import glob as _glob
    tex = {}
    for c in _glob(os.path.join(args.tex_csv_base or args.out_dir,
                                'z*-*', 'candidates.csv')):
        for r in csv.DictReader(open(c)):
            tex[r['stem']] = float(r['tex'])
    if not tex:
        raise FileNotFoundError('no candidates.csv under --tex_csv_base — '
                                'run find_discont first (tex source)')
    pt = np.array([tex.get(s, np.nan) for s in all_stems])
    live = pt >= args.tex_min
    print(f'{live.sum()}/{len(live)} textured patches '
          f'(tex >= {args.tex_min}; {np.isnan(pt).sum()} without tex)')
    per_z_vecs = {}                                  # abs z -> list of (dx, dy) image px
    for za, d1 in zip(all_za[live], all_d1[live]):
        for z in range(d1.shape[0]):
            per_z_vecs.setdefault(za + z, []).append(d1[z])

    zs = np.array(sorted(per_z_vecs))
    med = np.array([np.median(per_z_vecs[z], axis=0) for z in zs])
    mad = np.array([np.median(np.abs(np.asarray(per_z_vecs[z]) - m), axis=0)
                    for z, m in zip(zs, med)])
    R = np.array([float(np.linalg.norm(np.mean(v, axis=0))
                        / (np.mean(np.linalg.norm(v, axis=1)) + 1e-9))
                  for v in (np.asarray(per_z_vecs[z]) for z in zs)])
    n = np.array([len(per_z_vecs[z]) for z in zs])

    # cumulative trajectory over the full z range; unmeasured between-chunk
    # steps contribute 0 and are marked
    z_all = np.arange(zs.min(), zs.max() + 1)
    step_all = np.zeros((len(z_all), 2))
    measured = np.isin(z_all, zs)
    step_all[measured] = med
    cum = np.cumsum(step_all, axis=0)

    csv_path = os.path.join(args.out_dir, 'drift_per_z.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['z', 'dx_px', 'dy_px', 'mad_x', 'mad_y', 'coherence', 'n'])
        for k, z in enumerate(zs):
            w.writerow([z, f'{med[k, 0]:.4f}', f'{med[k, 1]:.4f}',
                        f'{mad[k, 0]:.4f}', f'{mad[k, 1]:.4f}',
                        f'{R[k]:.3f}', n[k]])
    np.savez(os.path.join(args.out_dir, 'drift_raw.npz'),
             zs=zs, med=med, mad=mad, R=R, n=n, cum=cum, z_all=z_all,
             all_stems=np.array(all_stems), all_za=all_za, all_d1=all_d1)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             facecolor='#fcfcfb')
    for ax in axes:
        ax.set_facecolor('#fcfcfb')
        for b in range(48, int(z_all.max()), 48):
            ax.axvline(b, color='#e2e1dc', lw=0.7, zorder=0)
    axes[0].plot(zs, med[:, 0], color='#2a78d6', lw=1.2, label='dx')
    axes[0].plot(zs, med[:, 1], color='#eb6834', lw=1.2, label='dy')
    axes[0].set_ylabel('median step (px/slice)')
    axes[0].legend(frameon=False)
    axes[1].plot(z_all, cum[:, 0], color='#2a78d6', lw=1.4, label='x')
    axes[1].plot(z_all, cum[:, 1], color='#eb6834', lw=1.4, label='y')
    axes[1].set_ylabel('cumulative shift (px)')
    axes[1].legend(frameon=False)
    axes[2].plot(zs, R, color='#52514e', lw=1.0)
    axes[2].set_ylabel('coherence R')
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel('z (slice; gray lines = unmeasured chunk boundaries)')
    axes[0].set_title('THX10 global drift — cross-patch median of adjacent-slice '
                      'phase-corr shifts', color='#0b0b0b')
    fig.tight_layout()
    png = os.path.join(args.out_dir, 'drift.png')
    fig.savefig(png, dpi=160, facecolor='#fcfcfb')

    print(f'-> {csv_path}\n-> {png}')
    print(f'median |step| {np.median(np.hypot(*med.T)):.3f} px/slice, '
          f'median coherence {np.median(R):.2f}')
    print(f'cumulative drift: x {cum[-1, 0]:+.1f} px, y {cum[-1, 1]:+.1f} px, '
          f'max |cum| {np.hypot(*cum.T).max():.1f} px over {len(z_all)} slices')


if __name__ == '__main__':
    main()
