"""Step 3 — score recovered registrations against ground truth + figures.

    python registration/evaluate.py --dir /home/cheese/workspace/Output/registration/reg1024z120s0

For every recovered_{label}.json in --dir (plus the unregistered 'corrupted'
row as the do-nothing baseline), reports:

- transform recovery: per-slice absolute rotation/translation/scale error vs
  gt_transforms.json, both raw (the M_0 = I anchoring makes chains directly
  comparable) and after fitting the single global gauge (affine.fit_gauge);
  plus neighbor-pair (offset-1) relative errors, which are gauge-free.
- volume restoration: NCC + SSIM of registered_{label}.npy vs original.npy on
  the central 3/4 crop (border fill excluded).
- stack coherence: mean adjacent-slice NCC and Z-gradient energy (original and
  corrupted included for reference).

Writes metrics.csv and two figures:
  errors.png    per-slice translation / rotation error curves per method
                (gray = corrupted, i.e. the GT drift itself)
  reslice.png   XZ + YZ reslices: original | corrupted | each registered
                (the qualitative jagged -> smooth check)
"""

import argparse
import csv
import json
import os
import sys
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from registration import affine  # noqa: E402


def ncc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def transform_errors(pred_mats, gt_mats):
    """Per-slice absolute errors + offset-1 relative errors between two
    (Z, 2, 3) chains. Returns dict of per-slice arrays."""
    rot_e, trans_e, scale_e = [], [], []
    for p, g in zip(pred_mats, gt_mats):
        d = affine.compose(p, affine.invert(g))       # residual transform
        r, tx, ty, s = affine.mat_to_params(d)
        rot_e.append(abs(r))
        trans_e.append(float(np.hypot(tx, ty)))
        scale_e.append(abs(s - 1))
    rel_rot, rel_trans = [], []
    for z in range(len(pred_mats) - 1):
        dp = affine.compose(pred_mats[z + 1], affine.invert(pred_mats[z]))
        dg = affine.compose(gt_mats[z + 1], affine.invert(gt_mats[z]))
        r, tx, ty, _ = affine.mat_to_params(affine.compose(dp, affine.invert(dg)))
        rel_rot.append(abs(r))
        rel_trans.append(float(np.hypot(tx, ty)))
    return dict(rot=np.array(rot_e), trans=np.array(trans_e),
                scale=np.array(scale_e),
                rel_rot=np.array(rel_rot), rel_trans=np.array(rel_trans))


def volume_metrics(vol, original, crop):
    y0, y1, x0, x1 = crop
    a, b = vol[:, y0:y1, x0:x1], original[:, y0:y1, x0:x1]
    try:
        from skimage.metrics import structural_similarity
        ssim = float(np.mean([structural_similarity(x, y, data_range=2.0)
                              for x, y in zip(a, b)]))
    except ImportError:
        ssim = float('nan')
    adj = float(np.mean([ncc(a[z], a[z + 1]) for z in range(len(a) - 1)]))
    zgrad = float(np.mean((a[1:] - a[:-1]) ** 2))
    return dict(ncc=ncc(a, b), ssim=ssim, adj_ncc=adj, zgrad=zgrad)


def main():
    parser = argparse.ArgumentParser(description='Evaluate recovered registrations')
    parser.add_argument('--dir', required=True, help="perturb.py's experiment dir")
    args = parser.parse_args()

    with open(os.path.join(args.dir, 'gt_transforms.json')) as f:
        gt = json.load(f)
    gt_mats = np.array(gt['abs_mats'])
    original = np.load(os.path.join(args.dir, 'original.npy'))
    corrupted = np.load(os.path.join(args.dir, 'corrupted.npy'))
    Z, Y, X = original.shape
    crop = (Y // 8, Y - Y // 8, X // 8, X - X // 8)      # central 3/4

    ident = np.stack([affine.identity()] * Z)
    methods = [('corrupted', ident, corrupted)]
    for path in sorted(glob(os.path.join(args.dir, 'recovered_*.json'))):
        with open(path) as f:
            rec = json.load(f)
        vol = np.load(os.path.join(args.dir, f"registered_{rec['label']}.npy"))
        methods.append((rec['label'], np.array(rec['abs_mats']), vol))

    rows, curves = [], {}
    ref_coherence = volume_metrics(original, original, crop)
    print(f"original: adj_ncc {ref_coherence['adj_ncc']:.4f} "
          f"zgrad {ref_coherence['zgrad']:.5f}")
    for label, mats, vol in methods:
        # recovered chain M_z maps original -> corrupted; a perfect run has
        # mats == gt_mats. 'corrupted' uses identity = the do-nothing baseline.
        err = transform_errors(mats, gt_mats)
        _, gauged = affine.fit_gauge(mats, gt_mats)
        gerr = transform_errors(gauged, gt_mats)
        vm = volume_metrics(vol, original, crop)
        rows.append(dict(method=label,
                         trans_err_mean=err['trans'].mean(),
                         trans_err_med=np.median(err['trans']),
                         trans_err_max=err['trans'].max(),
                         rot_err_mean=err['rot'].mean(), rot_err_max=err['rot'].max(),
                         scale_err_mean=err['scale'].mean(),
                         gauged_trans_err_mean=gerr['trans'].mean(),
                         rel_trans_err_mean=err['rel_trans'].mean(),
                         rel_rot_err_mean=err['rel_rot'].mean(), **vm))
        curves[label] = err
        print(f"{label:14s} trans err mean {err['trans'].mean():7.2f} px  "
              f"rot err mean {err['rot'].mean():6.3f} deg  ncc {vm['ncc']:.4f}  "
              f"adj_ncc {vm['adj_ncc']:.4f}")

    csv_path = os.path.join(args.dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f'metrics -> {csv_path}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label, err in curves.items():
        kw = dict(color='0.6', lw=2) if label == 'corrupted' else {}
        axes[0].plot(err['trans'], label=label, **kw)
        axes[1].plot(err['rot'], label=label, **kw)
    axes[0].set_title('|translation error| per slice (px)')
    axes[1].set_title('|rotation error| per slice (deg)')
    for ax in axes:
        ax.set_xlabel('slice z'), ax.legend(), ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, 'errors.png'), dpi=150)
    plt.close(fig)

    cols = [('original', original)] + [(la, v) for la, _, v in methods]
    w0 = max(0, X // 2 - 256)
    fig, axes = plt.subplots(2, len(cols), figsize=(4 * len(cols), 7), squeeze=False)
    for c, (label, vol) in enumerate(cols):
        axes[0][c].imshow(vol[:, Y // 2, w0:w0 + 512], cmap='gray', aspect='auto',
                          vmin=-1, vmax=1)
        axes[1][c].imshow(vol[:, w0:w0 + 512, X // 2], cmap='gray', aspect='auto',
                          vmin=-1, vmax=1)
        axes[0][c].set_title(label)
        for r in (0, 1):
            axes[r][c].set_xticks([]), axes[r][c].set_yticks([])
    axes[0][0].set_ylabel('XZ'), axes[1][0].set_ylabel('YZ')
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, 'reslice.png'), dpi=150)
    plt.close(fig)
    print(f"figures -> {os.path.join(args.dir, 'errors.png')}, "
          f"{os.path.join(args.dir, 'reslice.png')}")


if __name__ == '__main__':
    main()
