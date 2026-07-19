"""regis2 random-walk experiment on a 3D TIFF patch: corrupt -> register from
the latent -> warp the latent -> decode (the registration/enhance.py "new
way"), then write side-by-side 3D TIFFs for visualization.

    python regis2/run_walkbig.py                    # walk_big on 007003005.tif
    python regis2/run_walkbig.py --perturb walk     # registration/ defaults

Steps (everything imported from registration/, nothing there modified):
1. Similarity-random-walk corruption, --perturb selects the strength:
   'walk'     = registration/perturb.py defaults (rel rot ~U(+-0.5 deg),
                shift ~U(+-3 px), scale ~U(1+-0.005); seed 0)
   'walk_big' = 2x steps (1 deg, 6 px, 0.01; seed 1) — the same chains as the
                matching perturb_options.py panels. GT saved.
   'walk_drift' = walk composed with perturb_options' smooth deterministic
                drift (sinusoidal +-10 px translation / +-1 deg rotation, one
                cycle): random tearing ON TOP of pure low-frequency motion.
2. register.register_features on 'latent' feature planes (phase-corr +
   regularized refine + anchored graph solve, tuned defaults).
3. enhance.encode_full -> enhance.warp_latents (M_z^-1, translation/8) ->
   enhance.decode_tiled in ONE 32x32x32-latent tile (256^3 out, no seams);
   plus unwarped decodes of original and corrupted, and the pixel-space
   registration of the input by the same recovered transforms (old way).
4. One uint8 3D tif of ZX pages (repo convention for enhancement viewing:
   page = Y, rows = super-resolved Z, cols = X; inputs windowed [-1,1],
   enhanced [-0.8,1] = the 11g floor):
   {tag}_compare7.tif            original | corrupted | original enh |
                                 corrupted enh | latent-reg enh |
                                 latent-reg input (pixel warp, trilinear) |
                                 latent-reg input enhanced (old way)
   (tag = walk | walkbig | ...; .npy/.json side outputs share the tag)
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from registration import affine, enhance, features, register  # noqa: E402
from registration.evaluate import transform_errors  # noqa: E402
from registration.perturb import sample_transforms  # noqa: E402
from regis2.graph_tv import solve_graph_tv  # noqa: E402
from regis2.perturb_options import (DEFAULT_TIF, DEFAULT_OUT, apply,  # noqa: E402
                                    chunk_transforms, drift_transforms, upz)


def encode_h_planes(eng, stack, zbatch=8):
    """(Z, Y, X) [-1, 1] stack -> continuous PRE-QUANT encoder output h as
    (Z, z_channels, H, W) on the engine device (features.py's 'latent')."""
    normed = eng.normalize(stack).astype(np.float32)
    outs = []
    with torch.inference_mode():
        for a in range(0, len(normed), zbatch):
            s = torch.from_numpy(normed[a:a + zbatch])[:, None].to(eng.device)
            outs.append(eng.gan.encoder(s)[0].float())
    return torch.cat(outs)


def quantize_planes(eng, h):
    """Run MSclean.encode's per-scale VAR quantization loop (steps 1-6 of its
    docstring) on already-computed h planes (Z, z_channels, H, W) -> the
    scale_latents list in volume form (1, z_channels, H, W, Z), CPU. Mirrors
    models/MSclean.py encode() exactly, minus the encoder pass."""
    gan = eng.gan
    H, W = h.shape[-2:]
    residual, lats = h, []
    with torch.inference_mode():
        for k, scale in enumerate(gan.scale_factors):
            r_k = residual
            if scale > 1:
                r_k = F.interpolate(r_k, size=(H // scale, W // scale),
                                    mode='bilinear', align_corners=False)
            code_k, _, _ = gan.quantizers[k](gan.quant_convs[k](r_k))
            if scale > 1:
                code_k = F.interpolate(code_k, size=(H, W),
                                       mode='bilinear', align_corners=False)
            latent_k = gan.post_quant_convs[k](code_k)
            lats.append(latent_k)
            residual = residual - latent_k
    return [l.permute(1, 2, 3, 0)[None].cpu() for l in lats]


def main():
    parser = argparse.ArgumentParser(description='regis2 walk_big latent->decode run')
    parser.add_argument('--tif', default=DEFAULT_TIF, help='(Z, Y, X) [-1, 1] tiff')
    parser.add_argument('--out_base', default=DEFAULT_OUT)
    parser.add_argument('--perturb',
                        choices=('walk_big', 'walk', 'walk_drift', 'drift', 'chunk'),
                        default='walk_big',
                        help="Corruption model (default walk_big; 'walk' = "
                             "registration/ defaults; 'walk_drift' = walk + "
                             "smooth sinusoidal drift; 'drift' = the sinusoid "
                             "alone, scaled by --drift_scale; 'chunk' = one "
                             'walk_big-magnitude step at Z/3 or 2Z/3 — inter-'
                             'chunk drift)')
    parser.add_argument('--drift_scale', type=float, default=1.0,
                        help='drift mode: amp = 10*scale px, rot = 1*scale deg '
                             '(!= 1 goes into the output tag)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Walk seed (default: the matching perturb_options '
                             'panel — 1 for walk_big, 0 for walk)')
    parser.add_argument('--pairs', default='1,2,4')
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--reg_t', type=float, default=5.0)
    parser.add_argument('--reg_rs', type=float, default=10.0)
    parser.add_argument('--anchor', type=float, default=0.05)
    parser.add_argument('--solver', choices=('anchor', 'tv'), default='anchor',
                        help="Graph solve: 'anchor' = affine.solve_graph "
                             "(identity prior); 'tv' = regis2.graph_tv fused-"
                             'lasso on increments — one prior for drift, chunk '
                             "tears and walks alike (tag suffix 'tv')")
    parser.add_argument('--tv', type=float, default=4.0,
                        help='tv solver: translation-increment penalty (px)')
    parser.add_argument('--tv_lin', type=float, default=2.0,
                        help='tv solver: rot/scale-increment penalty (px-equiv)')
    parser.add_argument('--model', default='skipU')
    parser.add_argument('--latent_interp', choices=('bilinear', 'bicubic'),
                        default='bilinear',
                        help="warp_latents kernel; 'bicubic' appends 'bc' to the "
                             'output tag so bilinear results are kept')
    parser.add_argument('--no_rot', action='store_true',
                        help='Zero the rotation in BOTH the walk and the drift '
                             "(translation/scale unchanged; tag suffix 'nr')")
    parser.add_argument('--warp_space', choices=('postquant', 'prequant'),
                        default='postquant',
                        help='postquant = warp the quantized scale_latents '
                             '(enhance.warp_latents); prequant = warp the '
                             "continuous encoder output h, THEN quantize (tag "
                             "suffix 'pq')")
    args = parser.parse_args()

    vol = tifffile.imread(args.tif).astype(np.float32)
    Z, Y, X = vol.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    name = os.path.splitext(os.path.basename(args.tif))[0]
    out_dir = os.path.join(args.out_base, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'{args.tif}: ({Z}, {Y}, {X}) -> {out_dir}')

    # 1. corrupt (similarity random walk) + save GT
    rot, trans, scale, seed0 = ((1.0, 6.0, 0.01, 1)
                                if args.perturb in ('walk_big', 'chunk')
                                else (0.5, 3.0, 0.005, 0))
    drift_rot = 1.0
    if args.no_rot:
        rot = drift_rot = 0.0
    seed = seed0 if args.seed is None else args.seed
    tag = args.perturb.replace('_', '')
    if args.perturb == 'drift' and args.drift_scale != 1.0:
        tag += f'{args.drift_scale:g}'
    if args.latent_interp == 'bicubic':
        tag += 'bc'
    if args.warp_space == 'prequant':
        tag += 'pq'
    if args.no_rot:
        tag += 'nr'
    if args.solver == 'tv':
        tag += 'tv'
    break_z = None
    if args.perturb == 'drift':
        gt_mats = drift_transforms(Z, amp=10.0 * args.drift_scale,
                                   rot=drift_rot * args.drift_scale)
    elif args.perturb == 'chunk':
        gt_mats, break_z = chunk_transforms(Z, rot=rot, trans=trans,
                                            scale=scale, seed=seed)
        print(f'chunk break at z={break_z}')
    else:
        _, gt_mats = sample_transforms(Z, rot, trans, scale, seed)
        gt_mats = np.stack(gt_mats)
        if args.perturb == 'walk_drift':
            gt_mats = np.stack([affine.compose(d, w) for d, w
                                in zip(drift_transforms(Z, rot=drift_rot), gt_mats)])
    corrupted = apply(vol, gt_mats, device)
    np.save(os.path.join(out_dir, f'corrupted_{tag}.npy'), corrupted)
    with open(os.path.join(out_dir, f'gt_{tag}.json'), 'w') as f:
        json.dump(dict(tif=args.tif, perturb=args.perturb, rot=rot, trans=trans,
                       scale=scale, drift_scale=args.drift_scale, seed=seed,
                       break_z=break_z,
                       abs_params=[affine.mat_to_params(m) for m in gt_mats],
                       abs_mats=gt_mats.tolist()), f, indent=2)

    # 2. register from the latent
    eng = features.get_engine(args.model)
    for n in eng.gan.netg_names:
        getattr(eng.gan, n).eval()
    feats, stride = features.extract(corrupted, 'latent', eng=eng, batch=8)
    print(f'latent feats: {feats.shape} (stride {stride})')
    offsets = [int(o) for o in args.pairs.split(',')]
    abs_mats, pairs, refined = register.register_features(
        torch.from_numpy(feats).to(device), stride, offsets, args.iters,
        reg_t=args.reg_t, reg_rs=args.reg_rs, anchor=args.anchor)
    if args.solver == 'tv':
        meas = [(i, j, affine.params_to_mat(r, tx * stride, ty * stride, s))
                for (i, j), (r, tx, ty, s) in zip(pairs, refined)]
        abs_mats = solve_graph_tv(meas, Z, tv=args.tv, tv_lin=args.tv_lin)
        print(f'tv solve (tv {args.tv}, tv_lin {args.tv_lin})')
    with open(os.path.join(out_dir, f'recovered_{tag}.json'), 'w') as f:
        json.dump(dict(abs_params=[affine.mat_to_params(m) for m in abs_mats],
                       abs_mats=[m.tolist() for m in abs_mats]), f, indent=2)

    err = transform_errors(abs_mats, gt_mats)
    err0 = transform_errors(np.stack([affine.identity()] * Z), gt_mats)
    print(f"corrupted: trans err mean {err0['trans'].mean():.2f} px, "
          f"rot {err0['rot'].mean():.3f} deg")
    print(f"recovered: trans err mean {err['trans'].mean():.2f} px, "
          f"rot {err['rot'].mean():.3f} deg "
          f"(rel {err['rel_trans'].mean():.2f} px / {err['rel_rot'].mean():.3f} deg)")

    # 3. decodes: original / corrupted (unwarped) + latent-registered; and the
    #    pixel-space registration of the input by the same transforms (old way)
    def decode(stack, inv_mats=None):
        if inv_mats is not None and args.warp_space == 'prequant':
            h = encode_h_planes(eng, stack)              # continuous pre-quant
            lm = inv_mats.copy()
            lm[:, :, 2] /= features.STRIDE
            with torch.inference_mode():
                h = affine.warp_slices(h, lm, mode=args.latent_interp, fill=0.0)
            sl = quantize_planes(eng, h)                 # quantize AFTER warping
        else:
            sl = enhance.encode_full(eng, stack, zbatch=8)
            if inv_mats is not None:
                sl = enhance.warp_latents(sl, inv_mats, [0.0] * len(sl),
                                          eng.device, mode=args.latent_interp)
        return enhance.decode_tiled(eng, sl, tile=feats.shape[-1], ztile=Z,
                                    dbatch=1, verbose=False)

    inv = np.stack([affine.invert(m) for m in abs_mats])
    enhanced = decode(corrupted, inv)
    np.save(os.path.join(out_dir, f'enhanced_{tag}.npy'), enhanced)
    original_enh = decode(vol)
    corrupted_enh = decode(corrupted)
    registered_input = apply(corrupted, inv, device)     # old way, pixel warp
    registered_input_enh = decode(registered_input)      # old way, then enhance
    print(f'enhanced: {enhanced.shape}')

    from registration.evaluate import ncc
    c = np.s_[:, Y // 8:Y - Y // 8, X // 8:X - X // 8]   # central 3/4 crop
    for lbl, v in [('corrupted enh', corrupted_enh), ('latent-reg enh', enhanced),
                   ('pixel-reg enh', registered_input_enh)]:
        a = v[c].astype(np.float32)
        adj = np.mean([ncc(a[k], a[k + 1]) for k in range(len(a) - 1)])
        print(f'{lbl:15s} ncc vs original enh {ncc(a, original_enh[c].astype(np.float32)):.4f}'
              f'  adj_ncc {adj:.4f}')

    # 4. side-by-side 3D tifs, ZX pages (page = Y, rows = Z*8, cols = X)
    u8 = lambda v, lo, hi: (np.clip((v.astype(np.float32) - lo) / (hi - lo), 0, 1)
                            * 255).astype(np.uint8)

    def zx_tif(fname, cols, note):
        pages = np.concatenate(cols, axis=2).transpose(1, 0, 2)
        path = os.path.join(out_dir, fname)
        tifffile.imwrite(path, np.ascontiguousarray(pages))
        print(f'-> {path}  ({pages.shape} uint8, pages = Y, ZX view, {note})')

    tri = upz(corrupted)
    zx_tif(f'{tag}_compare7.tif',
           [u8(upz(vol), -1, 1), u8(tri, -1, 1), u8(original_enh, -0.8, 1),
            u8(corrupted_enh, -0.8, 1), u8(enhanced, -0.8, 1),
            u8(upz(registered_input), -1, 1), u8(registered_input_enh, -0.8, 1)],
           'original | corrupted | original enh | corrupted enh | '
           'latent-reg enh | latent-reg input | latent-reg input enh')


if __name__ == '__main__':
    main()
