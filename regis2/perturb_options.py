"""Preview perturbation options on a 3D TIFF patch (regis2 experiment).

    python regis2/perturb_options.py                      # defaults to 007003005.tif

Loads a (Z, Y, X) float32 [-1, 1] TIFF (default the 32x256x256 thx10 selected
patch), applies several per-slice corruption models, trilinearly upsamples
everything 8x in Z, and writes one PNG of ZX reslices side by side:

  original   no corruption
  walk       similarity random walk — registration/perturb.py's model
             (rel rot ~U(+-0.5 deg), shift ~U(+-3 px), scale ~U(1+-0.005),
             composed cumulatively; serial-section stage drift)
  jitter     INDEPENDENT per-slice similarity (same ranges as walk's relative
             step, but not accumulated — bounded, high-frequency misalignment)
  drift      smooth deterministic drift (sinusoidal translation +-10 px, one
             cycle over the stack, rotation +-1 deg) — pure low-frequency
             motion, the mode pairwise registration observes worst
  walk_big   the walk with 2x steps (rot 1 deg, shift 6 px, scale 0.01)

Transforms are applied with affine.warp_slices (bicubic, fill -1); components
imported from registration/ (nothing there is modified).
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from registration import affine  # noqa: E402
from registration.perturb import sample_transforms  # noqa: E402

DEFAULT_TIF = '/home/cheese/workspace/Data/thx10/selected/007003005.tif'
DEFAULT_OUT = '/home/cheese/workspace/Output/regis2'
UP = 8


def jitter_transforms(nz, rot, trans, scale, seed):
    """Independent (non-accumulated) per-slice similarity; slice 0 = identity."""
    rng = np.random.default_rng(seed)
    mats = [affine.identity()]
    for _ in range(1, nz):
        mats.append(affine.params_to_mat(rng.uniform(-rot, rot),
                                         rng.uniform(-trans, trans),
                                         rng.uniform(-trans, trans),
                                         1.0 + rng.uniform(-scale, scale)))
    return np.stack(mats)


def drift_transforms(nz, amp=10.0, rot=1.0, cycles=1.0):
    """Smooth deterministic drift: sinusoidal translation + rotation."""
    z = np.arange(nz) / max(nz - 1, 1)
    mats = [affine.params_to_mat(rot * np.sin(2 * np.pi * cycles * zi),
                                 amp * np.sin(2 * np.pi * cycles * zi),
                                 amp * (1 - np.cos(2 * np.pi * cycles * zi)) / 2,
                                 1.0)
            for zi in z]
    mats[0] = affine.identity()
    return np.stack(mats)


def apply(vol, mats, device):
    t = torch.from_numpy(vol)[:, None].to(device)
    with torch.no_grad():
        w = affine.warp_slices(t, mats, mode='bicubic', fill=-1.0)
    return w[:, 0].cpu().numpy()


def upz(vol):
    t = torch.from_numpy(np.ascontiguousarray(vol))[None, None].float()
    out = F.interpolate(t, size=(vol.shape[0] * UP, vol.shape[1], vol.shape[2]),
                        mode='trilinear', align_corners=False)
    return out[0, 0].numpy()


def main():
    parser = argparse.ArgumentParser(description='Preview perturbation options')
    parser.add_argument('--tif', default=DEFAULT_TIF, help='(Z, Y, X) [-1, 1] tiff')
    parser.add_argument('--out_base', default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    vol = tifffile.imread(args.tif).astype(np.float32)
    Z, Y, X = vol.shape
    print(f'{args.tif}: ({Z}, {Y}, {X})')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _, walk = sample_transforms(Z, 0.5, 3.0, 0.005, args.seed)
    _, walk_big = sample_transforms(Z, 1.0, 6.0, 0.01, args.seed + 1)
    options = [('original', None),
               ('walk', np.stack(walk)),
               ('jitter', jitter_transforms(Z, 0.5, 3.0, 0.005, args.seed)),
               ('drift', drift_transforms(Z)),
               ('walk_big', np.stack(walk_big))]

    name = os.path.splitext(os.path.basename(args.tif))[0]
    out_dir = os.path.join(args.out_base, name)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(options), figsize=(4 * len(options), 4.6))
    for ax, (label, mats) in zip(axes, options):
        v = vol if mats is None else apply(vol, mats, device)
        ax.imshow(upz(v)[:, Y // 2, :], cmap='gray', vmin=-1, vmax=1)
        ax.set_title(label)
        ax.set_xticks([]), ax.set_yticks([])
    axes[0].set_ylabel(f'ZX at Y={Y // 2} (Z x{UP} trilinear)')
    fig.tight_layout()
    out_png = os.path.join(out_dir, 'perturb_options.png')
    fig.savefig(out_png, dpi=150)
    print(f'-> {out_png}')


if __name__ == '__main__':
    main()
