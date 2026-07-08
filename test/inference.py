"""Run isotropic super-resolution inference on 3D .tif patches, one file at a time.

Loads any experiment's checkpoint (see test/load.py) and pushes each volume
through the model's own generation() path — no stitching, each .tif is treated
as one patch. The isotropic output XupX is saved as float32 .tif in (Z, Y, X)
page order.

Usage:
    python test/inference.py \
        --checkpoint $LOGS/THX10SDM20xw/thx10/vqcleanM0aMSskipP/Scale4/band5 \
        --source /path/to/patches_dir_or_file.tif \
        --destination test/out/band5
    # --dsp 1 to feed a real anisotropic volume as-is (default: config's dsp,
    #   which mimics training by Z-subsampling an isotropic input)
"""

import argparse
import inspect
import os
import sys
from glob import glob
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json  # noqa: E402
import numpy as np  # noqa: E402
import tifffile as tiff  # noqa: E402
import torch  # noqa: E402

from test.load import load_model  # noqa: E402


def normalize(vol, nm, norm_stats=None, key=None):
    """Mirror dataloader.data_multi.PairedImageDataset._normalize_image."""
    if nm == '01':
        vol = (vol - vol.min()) / (vol.max() - vol.min())
    elif nm == '11':
        vol = (vol - vol.min()) / (vol.max() - vol.min())
        vol = vol * 2 - 1
    elif nm == '11p':
        if norm_stats is None or key not in norm_stats:
            print(f"WARNING: nm='11p' but no stats for key '{key}' — falling back to '11'")
            return normalize(vol, '11')
        lo, hi = norm_stats[key]
        vol = np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1) * 2 - 1
    return vol  # '00': untouched


def center_crop(vol, cropsize, cropz):
    """Center-crop a (Y, X, Z) volume; 0 leaves an axis untouched."""
    y, x, z = vol.shape
    if cropsize > 0:
        vol = vol[(y - cropsize) // 2:(y - cropsize) // 2 + cropsize,
                  (x - cropsize) // 2:(x - cropsize) // 2 + cropsize, :]
    if cropz > 0:
        vol = vol[:, :, (z - cropz) // 2:(z - cropz) // 2 + cropz]
    return vol


@torch.no_grad()
def infer_volume(gan, vol, device):
    """Run one (Y, X, Z) float32 volume through gan.generation().

    Returns (xupx, xup): the isotropic output and the trilinear-upsampled
    input, both as (Y, X, Z) numpy arrays.
    """
    t = torch.from_numpy(vol.astype(np.float32))[None, None].to(device)  # (1,1,Y,X,Z)
    batch = {'img': [t]}
    if 'deterministic' in inspect.signature(gan.generation).parameters:
        gan.generation(batch, deterministic=True)
    else:
        gan.generation(batch)
    xupx = gan.XupX[0, 0].float().cpu().numpy()
    xup = gan.Xup[0, 0].float().cpu().numpy()
    return xupx, xup


def main():
    parser = argparse.ArgumentParser(description='Per-file 3D patch inference')
    parser.add_argument('--checkpoint', required=True,
                        help='Timestamped checkpoint dir, its checkpoints/ parent, or the experiment root')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch to load (default: latest)')
    parser.add_argument('--source', required=True, help='A .tif file or a directory of .tif files')
    parser.add_argument('--destination', default=None,
                        help='Output dir (default: test/out/{experiment name})')
    parser.add_argument('--nm', default=None, help="Normalization override (default: config's --nm)")
    parser.add_argument('--norm_stats', default=None,
                        help="norm_stats.json path for nm='11p' (key = tif parent dir name)")
    parser.add_argument('--dsp', type=int, default=None,
                        help="Z-subsample override; 1 = feed anisotropic volume as-is (default: config's --dsp)")
    parser.add_argument('--cropsize', type=int, default=0, help='Center-crop Y/X (0 = full)')
    parser.add_argument('--cropz', type=int, default=0, help='Center-crop Z before dsp (0 = full)')
    parser.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')
    parser.add_argument('--save_input', action='store_true',
                        help='Also save the trilinear-upsampled input as {stem}_input.tif')
    parser.add_argument('--eval', dest='train_mode', action='store_false',
                        help='Use .eval() for the generator components (BatchNorm running stats, '
                             'dropout off -> deterministic). Default is .train() for MC dropout: '
                             'batch-stat BN and active dropout, stochastic per run. The GAN '
                             'module itself always stays eval so generation() still skips the '
                             'cropz training crop.')
    args = parser.parse_args()

    gan, cfg = load_model(args.checkpoint, epoch=args.epoch, device=args.device)
    device = next(gan.parameters()).device
    if args.train_mode:
        for name in gan.netg_names:
            getattr(gan, name).train()
        print(f'train mode (default, MC dropout): {", ".join(gan.netg_names)} set to .train() '
              f'(gan stays eval; pass --eval for deterministic inference)')
    # Older models (e.g. vqclean) apply the cropz training crop unconditionally in
    # generation() — not gated on self.training — which would silently truncate the
    # input to its first cropz slices. Zero it; use this script's --cropz to crop.
    gan.hparams.cropz = 0
    if args.dsp is not None:
        gan.hparams.dsp = args.dsp
    nm = args.nm if args.nm is not None else getattr(cfg, 'nm', '00')

    norm_stats = None
    if nm == '11p':
        stats_path = args.norm_stats
        if stats_path and os.path.isfile(stats_path):
            with open(stats_path) as f:
                raw = json.load(f)
            stats = raw.get('stats', raw)
            norm_stats = {k: (float(v[0]), float(v[1])) for k, v in stats.items()}
        else:
            print("WARNING: nm='11p' without --norm_stats — will fall back to '11'")

    if os.path.isdir(args.source):
        files = sorted(glob(os.path.join(args.source, '*.tif'))
                       + glob(os.path.join(args.source, '*.tiff')))
    else:
        files = [args.source]
    if not files:
        raise FileNotFoundError(f'No .tif files found at {args.source}')

    dest = args.destination or os.path.join(REPO_ROOT, 'test', 'out',
                                            os.path.basename(args.checkpoint.rstrip('/')))
    os.makedirs(dest, exist_ok=True)
    print(f'{len(files)} file(s) -> {dest}  (nm={nm}, dsp={gan.hparams.dsp})')

    for path in files:
        vol = tiff.imread(path).astype(np.float32)          # (Z, Y, X)
        vol = np.transpose(vol, (1, 2, 0))                  # -> (Y, X, Z), Z last
        vol = center_crop(vol, args.cropsize, args.cropz)
        vol = normalize(vol, nm, norm_stats, key=Path(path).parent.name)

        xupx, xup = infer_volume(gan, vol, device)

        stem = Path(path).stem
        out_path = os.path.join(dest, stem + '.tif')
        tiff.imwrite(out_path, np.transpose(xupx, (2, 0, 1)).astype(np.float32))
        if args.save_input:
            tiff.imwrite(os.path.join(dest, stem + '_input.tif'),
                         np.transpose(xup, (2, 0, 1)).astype(np.float32))
        print(f'  {stem}: in (Z,Y,X)={vol.shape[2]}x{vol.shape[0]}x{vol.shape[1]}'
              f' -> out {xupx.shape[2]}x{xupx.shape[0]}x{xupx.shape[1]}'
              f'  range [{xupx.min():.3f}, {xupx.max():.3f}]')


if __name__ == '__main__':
    main()
