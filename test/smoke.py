"""End-to-end smoke test: load a real checkpoint and run one synthetic patch.

    python test/smoke.py [checkpoint_dir]

Without an argument, picks the most recently written checkpoint run under
$LOGS (cfg/env.json 'brcb') that contains .pth files.
"""

import os
import sys
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json  # noqa: E402
import torch  # noqa: E402

from test.load import load_model  # noqa: E402
from test.inference import infer_volume  # noqa: E402


def newest_checkpoint_dir():
    with open(os.path.join(REPO_ROOT, 'cfg', 'env.json')) as f:
        logs_root = json.load(f)['brcb']['LOGS']
    runs = {}
    for f in glob(os.path.join(logs_root, '**', 'checkpoints', '*', '*_model_epoch_*.pth'),
                  recursive=True):
        d = os.path.dirname(f)
        runs[d] = max(runs.get(d, 0), os.path.getmtime(f))
    if not runs:
        raise FileNotFoundError(f'No checkpoints with .pth files under {logs_root}')
    return max(runs, key=runs.get)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else newest_checkpoint_dir()
    print(f'Checkpoint: {ckpt}')

    gan, cfg = load_model(ckpt)
    device = next(gan.parameters()).device
    print(f'Model: {cfg.models} (netG {cfg.netG}) on {device}')

    yx = 64
    z_in = cfg.cropz if getattr(cfg, 'cropz', 0) > 0 else 32
    vol = torch.rand(yx, yx, z_in).numpy()  # (Y, X, Z) in [0, 1]
    if getattr(cfg, 'nm', '00') in ('11', '11p'):
        vol = vol * 2 - 1

    xupx = infer_volume(gan, vol, device)[0]

    import numpy as np
    assert np.isfinite(xupx).all(), 'XupX contains NaN/Inf'
    assert xupx.shape[:2] == (yx, yx), f'Y/X changed: {xupx.shape}'
    if hasattr(gan, 'uprate'):
        z_expected = (z_in // getattr(gan.hparams, 'dsp', 1)) * gan.uprate
        assert xupx.shape[2] == z_expected, \
            f'Z {xupx.shape[2]} != expected {z_expected} (uprate {gan.uprate})'
    print(f'PASS: in (Y,X,Z)={vol.shape} -> XupX {xupx.shape}, '
          f'range [{xupx.min():.3f}, {xupx.max():.3f}]')


if __name__ == '__main__':
    main()
