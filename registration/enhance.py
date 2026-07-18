"""GAN enhancement of a (Z, Y, X) stack, with optional latent-space registration.

    python registration/enhance.py --dir $OUT/registration/full1024 \
        --stack corrupted.npy --transforms recovered_latent.json --out latentreg_enh

Encodes the stack once at full slice resolution (the 2D VQ encoder is strictly
per-slice, so Z-chunked encoding is exact), optionally warps every per-scale
latent plane z by the recovered M_z^-1 (--transforms; translation divided by
the latent stride, rotation/scale unchanged — affine.py's centered-coordinate
convention makes them resolution-independent), then decodes the (possibly
registered) 3D latent tile by tile through net_g into the 8x Z-super-resolved
volume. Output is denormalized back to the pre-gamma [-1, 1] scale of the
input stack and saved as enhanced/{out}.npy (float16) + {out}_settings.json.

This is the NEW registration workflow ("register the latent, then decode"),
as opposed to register.py's pixel-space application of the same transforms —
run both stacks through this script so old way and new way share the exact
encode/decode path. Inference is deterministic (train_mode=False).

Pitfalls (see README):
- A bilinear latent warp is in-distribution for the decoder (encode() itself
  bilinearly upsamples the coarse code planes), but the encoder only
  approximately commutes with rotation/subpixel shift — that gap is what the
  enhanced-volume comparison measures.
- Latent value 0 is not "background"; --fill background instead fills warp
  borders with the latent of a constant -1.0 slice. Metrics downstream use a
  central crop either way.
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from registration import affine, features  # noqa: E402

UP = 8   # net_g decode upsampling per axis (skipU: three x2 stages)


def encode_full(eng, vol, zbatch):
    """(Z, Y, X) [-1, 1] stack -> list of per-scale (1, 4, Y/8, X/8, Z) float32
    CPU latents (full latent resolution, quantized VAR contributions)."""
    Z = vol.shape[0]
    normed = eng.normalize(vol).astype(np.float32)
    chunks = None
    with torch.inference_mode():
        for a in range(0, Z, zbatch):
            x = torch.from_numpy(normed[a:a + zbatch]).permute(1, 2, 0)[None, None]
            scale_latents, _ = eng.encode(x.to(eng.device))
            if chunks is None:
                chunks = [[] for _ in scale_latents]
            for k, s in enumerate(scale_latents):
                chunks[k].append(s.float().cpu())
    return [torch.cat(c, dim=4) for c in chunks]


def load_inv_mats(dir_, name, Z):
    """Transforms json (or 'identity') -> (Z, 2, 3) inverse matrices M_z^-1,
    pixel units (same application direction as register.py)."""
    if name == 'identity':
        return np.stack([affine.identity()] * Z)
    with open(os.path.join(dir_, name)) as f:
        mats = np.array(json.load(f)['abs_mats'])
    assert len(mats) == Z, f'{name} has {len(mats)} transforms for {Z} slices'
    return np.stack([affine.invert(m) for m in mats])


def background_fill(eng):
    """Per-scale (1, 4, 1, 1) latent fill = the latent of a constant -1.0
    (pixel background) slice, sampled at the plane center (away from the
    encoder's border effects)."""
    const = -np.ones((1, 256, 256), dtype=np.float32)
    x = torch.from_numpy(eng.normalize(const).astype(np.float32))
    x = x.permute(1, 2, 0)[None, None].to(eng.device)
    with torch.inference_mode():
        scale_latents, _ = eng.encode(x)
    fills = []
    for s in scale_latents:                     # (1, 4, 32, 32, 1)
        h, w = s.shape[2], s.shape[3]
        fills.append(s[0, :, h // 2, w // 2, 0].reshape(1, -1, 1, 1).float().cpu())
    return fills


def warp_latents(scale_latents, inv_mats, fills, device):
    """Warp every per-scale latent plane z by inv_mats[z] (global warp, before
    any tiling). Only the translation is rescaled to latent px."""
    lat_mats = inv_mats.copy()
    lat_mats[:, :, 2] /= features.STRIDE
    warped = []
    with torch.no_grad():
        for s, fill in zip(scale_latents, fills):
            planes = s[0].permute(3, 0, 1, 2).to(device)          # (Z, 4, h, w)
            f = fill.to(device) if torch.is_tensor(fill) else fill
            w = affine.warp_slices(planes, lat_mats, mode='bilinear', fill=f)
            warped.append(w.permute(1, 2, 3, 0)[None].cpu())      # (1, 4, h, w, Z)
    return warped


def decode_tiled(eng, scale_latents, tile, ztile, dbatch, verbose=True):
    """Tile the full-volume latent and decode each tile -> (Z*8, Y, X) float16
    volume on the pre-gamma [-1, 1] scale."""
    _, _, H, W, Zl = scale_latents[0].shape
    assert H % tile == 0 and W % tile == 0 and Zl % ztile == 0, \
        f'latent ({H}, {W}, {Zl}) not divisible by tile {tile}/{ztile}'
    out = np.empty((Zl * UP, H * UP, W * UP), dtype=np.float16)
    coords = [(y, x, z) for z in range(0, Zl, ztile)
              for y in range(0, H, tile) for x in range(0, W, tile)]
    with torch.inference_mode():
        for i in range(0, len(coords), dbatch):
            batch = coords[i:i + dbatch]
            tiles = [torch.cat([s[:, :, y:y + tile, x:x + tile, z:z + ztile]
                                for (y, x, z) in batch], dim=0).to(eng.device)
                     for s in scale_latents]
            dec = eng.denormalize(eng.decode(tiles)).clamp(-1, 1)
            for b, (y, x, z) in enumerate(batch):
                out[z * UP:(z + ztile) * UP, y * UP:(y + tile) * UP,
                    x * UP:(x + tile) * UP] = \
                    dec[b, 0].permute(2, 0, 1).half().cpu().numpy()
            if verbose and (i // dbatch) % 10 == 0:
                print(f'  decode tile {i + len(batch)}/{len(coords)}')
    return out


def main():
    parser = argparse.ArgumentParser(
        description='GAN-enhance a stack, optionally registering in latent space')
    parser.add_argument('--dir', required=True, help="perturb.py's experiment dir")
    parser.add_argument('--stack', required=True, help='Input .npy filename in --dir')
    parser.add_argument('--out', default=None,
                        help='Output stem -> enhanced/{out}.npy (required unless --self_test)')
    parser.add_argument('--transforms', default=None,
                        help="recovered_*.json in --dir (latents warped by M_z^-1), "
                             "or 'identity' (sanity: warp by identity)")
    parser.add_argument('--model', default='skipU', help='Registry model (default skipU)')
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--zbatch', type=int, default=2,
                        help='Slices per encoder pass (2 fits 1024^2 in 24 GB)')
    parser.add_argument('--tile', type=int, default=32,
                        help='XY latent px per decode tile (32 -> 256 px out)')
    parser.add_argument('--ztile', type=int, default=24,
                        help='Z latent slices per decode tile (24 -> 192 out)')
    parser.add_argument('--dbatch', type=int, default=2, help='Tiles per decode pass')
    parser.add_argument('--fill', choices=('zero', 'background'), default='zero',
                        help='Latent warp fill: 0.0, or the encoded latent of a '
                             'constant -1.0 slice (center value, per scale/channel)')
    parser.add_argument('--self_test', action='store_true',
                        help='Check identity-warp == no-warp on the first z-chunk; '
                             'prints max|diff|, saves nothing')
    args = parser.parse_args()
    if not args.self_test and args.out is None:
        parser.error('--out is required unless --self_test')

    t0 = time.time()
    vol = np.load(os.path.join(args.dir, args.stack))
    Z, Y, X = vol.shape
    print(f'{args.stack}: ({Z}, {Y}, {X}), transforms={args.transforms}')

    eng = features.get_engine(args.model, args.epoch, args.device)
    for name in eng.gan.netg_names:              # deterministic inference
        getattr(eng.gan, name).eval()
    eng.train_mode = False

    scale_latents = encode_full(eng, vol, args.zbatch)
    print(f'latents: {len(scale_latents)} scales x {tuple(scale_latents[0].shape)} '
          f'({time.time() - t0:.0f}s)')

    fills = (background_fill(eng) if args.fill == 'background'
             else [0.0] * len(scale_latents))

    if args.self_test:
        chunk = [s[:, :, :, :, :args.ztile] for s in scale_latents]
        inv = load_inv_mats(args.dir, 'identity', args.ztile)
        warped = warp_latents(chunk, inv, fills, eng.device)
        a = decode_tiled(eng, chunk, args.tile, args.ztile, args.dbatch, verbose=False)
        b = decode_tiled(eng, warped, args.tile, args.ztile, args.dbatch, verbose=False)
        diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).max()
        print(f'self test: identity-warp vs no-warp max|diff| = {diff:.2e} '
              f'({"OK" if diff < 1e-2 else "SUSPICIOUS"})')
        return

    if args.transforms:
        inv = load_inv_mats(args.dir, args.transforms, Z)
        scale_latents = warp_latents(scale_latents, inv, fills, eng.device)
        print('latents warped by M_z^-1')

    enhanced = decode_tiled(eng, scale_latents, args.tile, args.ztile, args.dbatch)

    out_dir = os.path.join(args.dir, 'enhanced')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'{args.out}.npy'), enhanced)
    settings = dict(stack=args.stack, out=args.out, transforms=args.transforms,
                    model=args.model, epoch=args.epoch, train_mode=False,
                    spec=eng.spec, zbatch=args.zbatch, tile=args.tile,
                    ztile=args.ztile, dbatch=args.dbatch, fill=args.fill,
                    in_shape=[Z, Y, X], out_shape=list(enhanced.shape),
                    seconds=round(time.time() - t0, 1))
    with open(os.path.join(out_dir, f'{args.out}_settings.json'), 'w') as f:
        json.dump(settings, f, indent=2)
    print(f"-> enhanced/{args.out}.npy {enhanced.shape} "
          f"({settings['seconds']:.0f}s total)")


if __name__ == '__main__':
    main()
