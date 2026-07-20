"""Codec round-trip over a folder of volumes: encode to stored indices,
decode the isotropic volume back FROM the stored indices.

    python inference/inference_latent.py --model skipU --exp myexp \
        --source /home/cheese/workspace/Data/thx10/roiAdsp4 --range 0 4

For every selected {stem}.tif (page order Z, Y, X) this writes under
{out_base}/{exp}/:
    codec/{stem}.npz    per-scale codebook indices (scale_0..scale_{K-1},
                        int32) + a meta record — the compact stored form;
                        provably sufficient: decode/ is produced from it
    decode/{stem}.tif   the decoded isotropic volume, denormalized to the
                        pre-gamma [-1, 1] scale, ZX page order (page y=k,
                        rows Z, cols X — the synthesized axis in-plane)
    input/{stem}.tif    the trilinear-interpolated input at the same size,
                        same scale and ZX order, for side-by-side inspection

The model comes from inference/registry.py (checkpoint, model_file, epoch,
normalization); --epoch/--device override the spec. Generator components run
in .train() by default (batch-stat BN + MC dropout, the test/ convention the
models were trained under) — the codec is still deterministic (the 2D VQ
stack is GroupNorm, dropout 0), but each decode is one MC draw; pass --eval
for fully deterministic running-stat inference.
"""

import argparse
import json
import os
import sys
import time
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import tifffile as tiff  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from inference import Engine, available, get as registry_get  # noqa: E402

DEFAULT_OUT_BASE = '/home/cheese/workspace/Output'


def to_zx_pages(vol_yxz):
    """(Y, X, Z) float32 array -> ZX tif pages (Y, Z, X): page y=k, rows Z, cols X."""
    return np.ascontiguousarray(np.transpose(vol_yxz, (0, 2, 1)))


def main():
    parser = argparse.ArgumentParser(
        description='Encode volumes to stored codec indices and decode them back')
    parser.add_argument('--model', required=True,
                        help=f'Registry model name ({", ".join(available())})')
    parser.add_argument('--exp', required=True,
                        help='Experiment name: outputs go to {out_base}/{exp}/')
    parser.add_argument('--source', required=True, help='Directory of .tif volumes (Z, Y, X pages)')
    parser.add_argument('--range', type=int, nargs=2, default=None, metavar=('START', 'END'),
                        help='Slice of the sorted file list, files[START:END] (default: all)')
    parser.add_argument('--epoch', type=int, default=None, help="Override the registry's epoch")
    parser.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')
    parser.add_argument('--eval', dest='train_mode', action='store_false',
                        help='Deterministic running-stat inference (default: generator '
                             'components in .train() — batch-stat BN + MC dropout)')
    parser.add_argument('--half', action='store_true',
                        help='Run the model and inputs in fp16 (halves GPU memory; codebook '
                             'argmin in fp16 can shift a few indices vs an fp32 run)')
    parser.add_argument('--out_base', default=DEFAULT_OUT_BASE,
                        help=f'Output base dir (default: {DEFAULT_OUT_BASE})')
    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.source, '*.tif'))
                   + glob(os.path.join(args.source, '*.tiff')))
    if args.range is not None:
        files = files[args.range[0]:args.range[1]]
    if not files:
        raise FileNotFoundError(f'No .tif files selected in {args.source} (range {args.range})')

    eng = Engine.from_registry(args.model, epoch=args.epoch, device=args.device,
                               train_mode=args.train_mode)
    if args.half:
        eng.gan.half()
    root = os.path.join(args.out_base, args.exp)
    dirs = {d: os.path.join(root, d) for d in ('codec', 'decode', 'input')}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    epoch = args.epoch if args.epoch is not None else registry_get(args.model).get('epoch')
    meta = dict(model=args.model, epoch=epoch if epoch is not None else 'latest', **eng.spec)
    print(f'{len(files)} file(s) -> {root}  (model {args.model}, spec {eng.spec}, '
          f'device {eng.device})')

    times, ratios = [], []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        t0 = time.perf_counter()

        vol = tiff.imread(path).astype(np.float32)      # (Z, Y, X)
        vol = np.transpose(vol, (1, 2, 0))              # (Y, X, Z)
        x = torch.from_numpy(eng.normalize(vol))[None, None].to(eng.device)
        if args.half:
            x = x.half()

        _, indices = eng.encode(x)
        codec = {f'scale_{k}': idx.cpu().numpy().astype(np.int32)
                 for k, idx in enumerate(indices)}
        np.savez_compressed(os.path.join(dirs['codec'], stem + '.npz'),
                            meta=json.dumps({**meta, 'input_shape_zyx':
                                             [vol.shape[2], vol.shape[0], vol.shape[1]]}),
                            **codec)

        # decode from the stored form, so codec/ alone reproduces decode/
        out = eng.decode(eng.latents_from_indices(indices))         # (1, 1, Y, X, Z')
        dec = eng.denormalize(out[0, 0].float().cpu().numpy())
        tiff.imwrite(os.path.join(dirs['decode'], stem + '.tif'),
                     to_zx_pages(dec).astype(np.float32))

        xup = F.interpolate(x, size=out.shape[2:], mode='trilinear')
        inp = eng.denormalize(xup[0, 0].float().cpu().numpy())
        tiff.imwrite(os.path.join(dirs['input'], stem + '.tif'),
                     to_zx_pages(inp).astype(np.float32))

        times.append(time.perf_counter() - t0)
        raw_bytes = vol.size * 4
        codec_bytes = os.path.getsize(os.path.join(dirs['codec'], stem + '.npz'))
        ratios.append(raw_bytes / codec_bytes)
        print(f'  {stem}: (Z,Y,X) {vol.shape[2]}x{vol.shape[0]}x{vol.shape[1]}'
              f' -> {out.shape[4]}x{out.shape[2]}x{out.shape[3]}'
              f'  range [{dec.min():.3f}, {dec.max():.3f}]'
              f'  codec {codec_bytes / 1024:.1f} KiB ({ratios[-1]:.0f}x vs raw)'
              f'  {times[-1]:.2f}s')

    print(f'done: {len(files)} volume(s), mean {sum(times) / len(times):.2f}s, '
          f'mean compression {sum(ratios) / len(ratios):.0f}x -> {root}')


if __name__ == '__main__':
    main()
