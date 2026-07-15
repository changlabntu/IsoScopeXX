"""Save the 2D VQ-autoencoder reconstruction of a 3D .tif volume.

Every model in the lineage reconstructs the input slice stack through its 2D
encoder -> quantizer -> decoder head (`self.reconstructions = self.forward(
input_slice)[0]` inside generation()) before the 3D generator runs. This
script calls that head directly — no 3D net_g, no Z upsampling — so the output
has the SAME (Z, Y, X) size as the input.

Usage (see test/inference2d.sh):
    python test/inference2d.py --checkpoint <run dir> --epoch 100 \
        --source /path/to/volume.tif --out out/summary2d/MSskipE.tif \
        [--save_input out/summary2d/input.tif]
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import tifffile as tiff  # noqa: E402
import torch  # noqa: E402

from test.load import load_model  # noqa: E402
from test.inference import normalize, invert_gamma, DEFAULT_OUT  # noqa: E402


@torch.no_grad()
def recon_volume(gan, vol, device):
    """(Y, X, Z) float32 volume -> slice-wise 2D reconstruction, (Z, Y, X)."""
    t = torch.from_numpy(vol.astype(np.float32))[None, None].to(device)  # (1,1,Y,X,Z)
    input_slice = t.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]                # (Z,C,Y,X)
    dec = gan.forward(input_slice)[0]                                    # (Z,C,Y,X)
    return dec[:, 0].float().cpu().numpy()                               # (Z,Y,X)


def main():
    parser = argparse.ArgumentParser(description='2D VQ-decoder reconstruction of one volume')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--epoch', type=int, default=None, help='Epoch to load (default: latest)')
    parser.add_argument('--source', required=True, help='A single 3D .tif (page order Z,Y,X)')
    parser.add_argument('--out', default=None,
                        help=f'Output .tif path (default: {DEFAULT_OUT}/summary2d/{{experiment name}}.tif)')
    parser.add_argument('--nm', default=None, help="Normalization override (default: config's --nm)")
    parser.add_argument('--gamma', type=float, default=None,
                        help="nm='11g' gamma override (default: config's --gamma)")
    parser.add_argument('--gamma_lo', type=float, default=None,
                        help="nm='11g' noise-floor override (default: config's --gamma_lo)")
    parser.add_argument('--no_invert', action='store_true',
                        help="nm='11g': keep the reconstruction (and saved input) in gamma "
                             'space instead of inverting back to the pre-gamma [-1, 1] scale')
    parser.add_argument('--save_input', default=None,
                        help='Also save the normalized raw input volume to this .tif path')
    parser.add_argument('--device', default=None)
    parser.add_argument('--eval', dest='train_mode', action='store_false',
                        help='.eval() the generator components (same convention as inference.py; '
                             'the 2D path is nearly deterministic either way — GroupNorm, dropout 0)')
    args = parser.parse_args()

    gan, cfg = load_model(args.checkpoint, epoch=args.epoch, device=args.device)
    device = next(gan.parameters()).device
    if args.train_mode:
        for name in gan.netg_names:
            getattr(gan, name).train()

    nm = args.nm if args.nm is not None else getattr(cfg, 'nm', '00')
    gamma = args.gamma if args.gamma is not None else (getattr(cfg, 'gamma', None) or 0.25)
    gamma_lo = args.gamma_lo if args.gamma_lo is not None else (getattr(cfg, 'gamma_lo', None) or -1.0)
    invert = nm == '11g' and not args.no_invert
    if nm == '11g':
        print(f"nm='11g': gamma={gamma}, gamma_lo={gamma_lo}, "
              f"output {'inverted back to pre-gamma scale' if invert else 'kept in gamma space'}")

    vol = tiff.imread(args.source).astype(np.float32)   # (Z, Y, X)
    vol = np.transpose(vol, (1, 2, 0))                  # (Y, X, Z)
    vol = normalize(vol, nm, gamma=gamma, gamma_lo=gamma_lo)

    out = args.out or os.path.join(DEFAULT_OUT, 'summary2d',
                                   os.path.basename(args.checkpoint.rstrip('/')) + '.tif')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if args.save_input:
        os.makedirs(os.path.dirname(args.save_input), exist_ok=True)
        # When inverting, run the input through the same round trip so it is
        # directly comparable to the reconstruction (floor band flattened to gamma_lo).
        inp = invert_gamma(vol, gamma, gamma_lo) if invert else vol
        tiff.imwrite(args.save_input, np.transpose(inp, (2, 0, 1)).astype(np.float32))
        print(f'input -> {args.save_input}')

    rec = recon_volume(gan, vol, device)
    if invert:
        rec = invert_gamma(rec, gamma, gamma_lo)
    tiff.imwrite(out, rec.astype(np.float32))
    print(f'{os.path.basename(out)}: in (Z,Y,X)={vol.shape[2]}x{vol.shape[0]}x{vol.shape[1]}'
          f' -> recon {rec.shape[0]}x{rec.shape[1]}x{rec.shape[2]}'
          f'  range [{rec.min():.3f}, {rec.max():.3f}]')


if __name__ == '__main__':
    main()
