"""Score + visualize the enhanced-volume comparison (see enhance.py).

    python registration/evaluate_enhanced.py --dir $OUT/registration/full1024

Six methods, fixed order:
  original_tri            original.npy, trilinear Z x8 (display/deviation ref)
  original_enh            enhanced/original_enh.npy  — the REFERENCE volume
  corrupted_tri           corrupted.npy, trilinear Z x8
  corrupted_enh           do-nothing baseline (corrupt then enhance)
  registered_latent_enh   OLD way: pixel-space registration, then enhance
  latentreg_enh           NEW way: latent-space registration, then decode

Per method: global NCC + strided per-slice SSIM vs original_enh on the
central 3/4 Y/X crop (evaluate.py's convention), mean adjacent-slice NCC and
Z-gradient energy (coherence). Metrics stream over Z slabs from mmapped
volumes. Writes enhanced/metrics_enhanced.csv and two figures:
  reslice_enhanced.png       XZ (Y=mid) + YZ (X=mid), all six columns
  reslice_enhanced_zoom.png  --zoom window, Z range straddling z-chunk seams
Also prints NCC(original_enh vs the pre-existing enhanced_original.npy) as a
cross-check when that file exists (old ad-hoc run; informational only).
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
import torch.nn.functional as F  # noqa: E402

from registration.evaluate import ncc  # noqa: E402

UP = 8
SLAB = 64


def tri_z(vol):
    """(Z, Y, X) -> (Z*UP, Y, X) float32, trilinear along Z only."""
    t = torch.from_numpy(np.ascontiguousarray(vol))[None, None].float()
    out = F.interpolate(t, size=(vol.shape[0] * UP, vol.shape[1], vol.shape[2]),
                        mode='trilinear', align_corners=False)
    return out[0, 0].numpy()


def stream_metrics(vol, ref, crop, ssim_stride):
    """Metrics vs ref on the central crop, streamed over Z slabs.
    vol/ref: (Z', Y, X) arrays (mmap or ndarray), same shape."""
    y0, y1, x0, x1 = crop
    Zp = vol.shape[0]
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        structural_similarity = None

    sa = sb = saa = sbb = sab = 0.0          # global NCC accumulators
    n = 0
    ssims, adj, zgrad = [], [], []
    prev = None
    for z in range(0, Zp, SLAB):
        a = np.asarray(vol[z:z + SLAB, y0:y1, x0:x1], dtype=np.float32)
        b = np.asarray(ref[z:z + SLAB, y0:y1, x0:x1], dtype=np.float32)
        sa += a.sum(); sb += b.sum()
        saa += (a * a).sum(); sbb += (b * b).sum(); sab += (a * b).sum()
        n += a.size
        if structural_similarity is not None:
            for k in range((-z) % ssim_stride, a.shape[0], ssim_stride):
                ssims.append(structural_similarity(a[k], b[k], data_range=2.0))
        block = a if prev is None else np.concatenate([prev, a])
        adj.extend(ncc(block[k], block[k + 1]) for k in range(len(block) - 1))
        zgrad.append(((block[1:] - block[:-1]) ** 2).mean())
        prev = a[-1:]
    cov = sab - sa * sb / n
    var = np.sqrt((saa - sa ** 2 / n) * (sbb - sb ** 2 / n)) + 1e-12
    return dict(ncc=float(cov / var),
                ssim=float(np.mean(ssims)) if ssims else float('nan'),
                adj_ncc=float(np.mean(adj)), zgrad=float(np.mean(zgrad)))


def main():
    parser = argparse.ArgumentParser(description='Evaluate enhanced-volume comparison')
    parser.add_argument('--dir', required=True, help="perturb.py's experiment dir")
    parser.add_argument('--ssim_stride', type=int, default=8,
                        help='SSIM on every Nth slice (default 8)')
    parser.add_argument('--zoom', type=int, default=512, help='Zoom window (px)')
    args = parser.parse_args()
    enh_dir = os.path.join(args.dir, 'enhanced')

    original = np.load(os.path.join(args.dir, 'original.npy'))
    corrupted = np.load(os.path.join(args.dir, 'corrupted.npy'))
    Z, Y, X = original.shape
    crop = (Y // 8, Y - Y // 8, X // 8, X - X // 8)      # central 3/4

    methods = [('original_tri', tri_z(original)),
               ('original_enh', np.load(os.path.join(enh_dir, 'original_enh.npy'),
                                        mmap_mode='r')),
               ('corrupted_tri', tri_z(corrupted)),
               ('corrupted_enh', np.load(os.path.join(enh_dir, 'corrupted_enh.npy'),
                                         mmap_mode='r')),
               ('registered_latent_enh',
                np.load(os.path.join(enh_dir, 'registered_latent_enh.npy'),
                        mmap_mode='r')),
               ('latentreg_enh', np.load(os.path.join(enh_dir, 'latentreg_enh.npy'),
                                         mmap_mode='r'))]
    ref = methods[1][1]

    rows, reslices = [], {}
    for label, vol in methods:
        assert vol.shape == (Z * UP, Y, X), f'{label}: {vol.shape}'
        m = stream_metrics(vol, ref, crop, args.ssim_stride)
        rows.append(dict(method=label, ncc_vs_origenh=m['ncc'],
                         ssim_vs_origenh=m['ssim'], adj_ncc=m['adj_ncc'],
                         zgrad=m['zgrad']))
        reslices[label] = (np.asarray(vol[:, Y // 2, :], dtype=np.float32),
                          np.asarray(vol[:, :, X // 2], dtype=np.float32))
        print(f"{label:22s} ncc {m['ncc']:.4f}  ssim {m['ssim']:.4f}  "
              f"adj_ncc {m['adj_ncc']:.4f}  zgrad {m['zgrad']:.5f}")

    by = {r['method']: r for r in rows}
    print(f"\nheadline (NCC vs original_enh): corrupted "
          f"{by['corrupted_enh']['ncc_vs_origenh']:.4f}  |  old way (pixel reg) "
          f"{by['registered_latent_enh']['ncc_vs_origenh']:.4f}  |  new way "
          f"(latent reg) {by['latentreg_enh']['ncc_vs_origenh']:.4f}")

    old = os.path.join(args.dir, 'enhanced_original.npy')
    if os.path.exists(old):
        m = stream_metrics(np.load(old, mmap_mode='r'), ref, crop, 10 ** 9)
        print(f'cross-check: pre-existing enhanced_original.npy vs original_enh '
              f'ncc {m["ncc"]:.4f} (informational)')

    csv_path = os.path.join(enh_dir, 'metrics_enhanced.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f'metrics -> {csv_path}')

    def figure(sl_z, sl_w, fname):
        fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 8),
                                 squeeze=False)
        for c, (label, _) in enumerate(methods):
            xz, yz = reslices[label]
            axes[0][c].imshow(xz[sl_z, sl_w], cmap='gray', aspect='auto',
                              vmin=-1, vmax=1)
            axes[1][c].imshow(yz[sl_z, sl_w], cmap='gray', aspect='auto',
                              vmin=-1, vmax=1)
            axes[0][c].set_title(label, fontsize=9)
            for r in (0, 1):
                axes[r][c].set_xticks([]), axes[r][c].set_yticks([])
        axes[0][0].set_ylabel(f'XZ at Y={Y // 2}')
        axes[1][0].set_ylabel(f'YZ at X={X // 2}')
        fig.tight_layout()
        fig.savefig(os.path.join(enh_dir, fname), dpi=150)
        plt.close(fig)
        print(f"figure -> {os.path.join(enh_dir, fname)}")

    figure(np.s_[:], np.s_[:], 'reslice_enhanced.png')
    # zoom straddles the z-chunk seams (multiples of ztile*UP = 192)
    w0 = max(0, X // 2 - args.zoom // 2)
    figure(np.s_[96:96 + 320], np.s_[w0:w0 + args.zoom], 'reslice_enhanced_zoom.png')


if __name__ == '__main__':
    main()
