"""Inference a row of adjacent patches and combine the outputs into one strip.

Runs a model over every thAAABBBCCC.tif patch in --source (a directory like
Data/THX10SDM20xw/aRow/, cut by tif_to_patches: page order (Z, Y, X), global
[-1, 1] intensity map), saves each isotropic output into {source}/{tag}/, then
concatenates the outputs along the axis the patch indices vary on — AAA = Y
(rows), BBB = X (cols), CCC = Z — into {source}/{tag}.tif. Patch names must
form a contiguous run on exactly one index (a row); anything else aborts.

Normalization follows test/inference.py: the run's --nm (nm='11g' applies the
config's --gamma/--gamma_lo) and outputs are inverted back to the pre-gamma
[-1, 1] scale by default.

View orientation: outputs are written in XY page order (page z=k, rows Y,
cols X) — the same orientation as the source patches and the concatenated
original.tif, so strips overlay directly. (The first version saved ZX pages,
the inference3d.sh convention; that view is misaligned with original.tif and
needed an ImageJ Stacks > Reslice > Top to compare — hence the switch.)

Run-time gamma adjustment (nm='11g'): encode and decode are decoupled.
  --gamma / --gamma_lo      set the ENCODE transform (what the model sees) and
                            the decode defaults. Deviating from the trained
                            values (roiD192gf: 0.5 / -0.8) is off-distribution;
                            the legitimate use is moving --gamma_lo to admit
                            (lower) or clip (higher) more of the faint band
                            before the model enhances it.
  --gamma_dec / --gamma_lo_dec  override the DECODE inversion only — a pure
                            monotonic tone remap of the output; smaller
                            --gamma_dec darkens midtones. Saved outputs can be
                            re-toned without re-inference:
                            x' = ((x - lo)/(1 - lo))**(gamma/gamma_dec) * (1 - lo) + lo

Usage:
    python test/inference_row.py \
        --checkpoint /home/gary/workspace/logs/thx10-071226/vqcleanM0aMSskipUB/roiD192gf/max5skip4 \
        --epoch 300 --source /home/gary/workspace/Data/THX10SDM20xw/aRow --tag skipUB
"""

import argparse
import os
import re
import sys
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import tifffile as tiff  # noqa: E402

from test.load import load_model  # noqa: E402
from test.inference import normalize, invert_gamma, infer_volume  # noqa: E402

# patch index position in thAAABBBCCC -> (name, volume axis in (Y, X, Z))
INDEX_AXES = {0: ('AAA/Y', 0), 1: ('BBB/X', 1), 2: ('CCC/Z', 2)}


def row_files(source):
    """Sorted patch files + the (Y, X, Z) axis their indices vary along."""
    files = sorted(f for f in glob(os.path.join(source, 'th*.tif'))
                   if re.fullmatch(r'th\d{9}\.tif', os.path.basename(f)))
    if len(files) < 2:
        raise FileNotFoundError(f'Need >= 2 thAAABBBCCC.tif patches in {source}, got {len(files)}')
    idx = np.array([[int(os.path.basename(f)[2 + 3 * k:5 + 3 * k]) for k in range(3)]
                    for f in files])
    varying = [k for k in range(3) if len(set(idx[:, k])) > 1]
    if len(varying) != 1:
        raise ValueError(f'Patches must form a row on exactly one index, '
                         f'got varying indices {[INDEX_AXES[k][0] for k in varying]}')
    k = varying[0]
    if list(idx[:, k]) != list(range(idx[0, k], idx[0, k] + len(files))):
        raise ValueError(f'{INDEX_AXES[k][0]} indices are not contiguous: {list(idx[:, k])}')
    print(f'{len(files)} patches, row along {INDEX_AXES[k][0]} '
          f'({idx[0, k]:03d}..{idx[-1, k]:03d})')
    return files, INDEX_AXES[k][1]


def main():
    parser = argparse.ArgumentParser(description='Row-of-patches inference + combined strip')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--epoch', type=int, default=None, help='Epoch to load (default: latest)')
    parser.add_argument('--source', required=True, help='Directory of thAAABBBCCC.tif patches')
    parser.add_argument('--tag', required=True,
                        help='Output name: patches -> {source}/{tag}/, strip -> {source}/{tag}.tif')
    parser.add_argument('--nm', default=None, help="Normalization override (default: config's --nm)")
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--gamma_lo', type=float, default=None)
    parser.add_argument('--gamma_dec', type=float, default=None,
                        help='Decode-only gamma for the output inversion (default: --gamma). '
                             'Smaller than --gamma darkens midtones; pure tone remap, the '
                             'model input keeps the trained transform')
    parser.add_argument('--gamma_lo_dec', type=float, default=None,
                        help='Decode-only floor for the output inversion (default: --gamma_lo)')
    parser.add_argument('--no_invert', action='store_true',
                        help="nm='11g': keep outputs in gamma space")
    parser.add_argument('--device', default=None)
    parser.add_argument('--eval', dest='train_mode', action='store_false',
                        help='Deterministic .eval() generator (default: train mode / MC dropout, '
                             'as in test/inference.py)')
    args = parser.parse_args()

    files, cat_axis = row_files(args.source)

    gan, cfg = load_model(args.checkpoint, epoch=args.epoch, device=args.device)
    device = next(gan.parameters()).device
    if args.train_mode:
        for name in gan.netg_names:
            getattr(gan, name).train()
    gan.hparams.cropz = 0

    nm = args.nm if args.nm is not None else getattr(cfg, 'nm', '00')
    gamma = args.gamma if args.gamma is not None else (getattr(cfg, 'gamma', None) or 0.25)
    gamma_lo = args.gamma_lo if args.gamma_lo is not None else (getattr(cfg, 'gamma_lo', None) or -1.0)
    gamma_dec = args.gamma_dec if args.gamma_dec is not None else gamma
    gamma_lo_dec = args.gamma_lo_dec if args.gamma_lo_dec is not None else gamma_lo
    invert = nm == '11g' and not args.no_invert
    if nm == '11g':
        print(f"nm='11g': encode gamma={gamma}, gamma_lo={gamma_lo}; "
              f"decode gamma={gamma_dec}, gamma_lo={gamma_lo_dec}"
              + ('' if invert else ' (kept in gamma space)'))

    dest = os.path.join(args.source, args.tag)
    os.makedirs(dest, exist_ok=True)

    outputs = []
    for path in files:
        vol = tiff.imread(path).astype(np.float32)      # (Z, Y, X)
        vol = np.transpose(vol, (1, 2, 0))              # (Y, X, Z)
        vol = normalize(vol, nm, gamma=gamma, gamma_lo=gamma_lo)
        xupx, _ = infer_volume(gan, vol, device)        # (Y, X, Z)
        if invert:
            xupx = invert_gamma(xupx, gamma_dec, gamma_lo_dec)
        outputs.append(xupx.astype(np.float32))
        name = os.path.basename(path)
        tiff.imwrite(os.path.join(dest, name), np.transpose(xupx, (2, 0, 1)).astype(np.float32))
        print(f'  {name}: {vol.shape} -> {xupx.shape}  range [{xupx.min():.3f}, {xupx.max():.3f}]')

    strip = np.concatenate(outputs, axis=cat_axis)      # (Y, X, Z) with one long axis
    strip_path = os.path.join(args.source, args.tag + '.tif')
    tiff.imwrite(strip_path, np.transpose(strip, (2, 0, 1)).astype(np.float32))  # XY pages
    print(f'{strip_path}: (Y, X, Z) = {strip.shape}, saved as XY pages '
          f'(pages Z, rows Y, cols X)')


if __name__ == '__main__':
    main()
