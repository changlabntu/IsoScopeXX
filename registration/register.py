"""Step 2 — recover per-slice transforms by aligning latent planes.

    python registration/register.py --dir /home/cheese/workspace/Output/registration/reg1024z120s0 --feat latent

Loads corrupted.npy, extracts per-slice feature planes (--feat latent | codec |
pixel, see features.py), then:

1. COARSE: for every slice pair (z, z+o), o in --pairs, channel-summed FFT
   phase correlation between the two feature planes -> integer + quadratic-
   subpixel translation init (in feature px; rotation/scale init 0/1 — the
   relative angles are small by construction).
2. REFINE: all pairs batched, per-pair similarity params (rot, tx, ty, log_s)
   optimized with Adam; the moving plane is warped differentiably
   (affine.theta_from_params + grid_sample) and matched to the fixed plane
   under a valid-region-masked L1 loss.
3. GRAPH SOLVE: pairwise measurements D_(z,z+o) ~ M_(z+o) o M_z^-1 (translation
   x8 back to pixel units) -> least-squares absolute transforms M_z with the
   gauge fixed at M_0 = I (affine.solve_graph; --solver tv swaps in
   graph_tv.solve_graph_tv's fused-lasso increment prior). Multiple offsets
   (default 1,2,4) suppress random-walk error accumulation of a pure
   neighbor chain.
4. APPLY: M_z^-1 warps each corrupted slice back (bicubic, full res) ->
   registered_{label}.npy + recovered_{label}.json.

--method xcorr instead runs the repo's translation-only image-space baseline
(utils/alignments.py align_stack_xy against slice 0).
--noise SIGMA / --gain G corrupt the stack further (additive Gaussian noise,
per-slice intensity gain jitter) before feature extraction — robustness
ablation; ground truth is unchanged and the label gets a _noise suffix.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from registration import affine, features  # noqa: E402


def phase_corr_shift(fixed, moving):
    """Translation d (dx, dy in px) such that shifting `moving` content by d
    aligns it to `fixed`. Both (C, H, W) torch. Channel-summed normalized
    cross-power spectrum; quadratic subpixel peak with wrap-around."""
    Fm = torch.fft.fft2(moving.float())
    Ff = torch.fft.fft2(fixed.float())
    cross = torch.conj(Fm) * Ff                       # peak of ifft at +d
    r = torch.fft.ifft2(cross / (cross.abs() + 1e-6)).real.sum(0)
    H, W = r.shape
    peak = int(torch.argmax(r))
    py, px = peak // W, peak % W

    def sub(vm, v0, vp):
        den = vm - 2 * v0 + vp
        return 0.5 * (vm - vp) / den if abs(den) > 1e-12 else 0.0

    dy = py + sub(r[(py - 1) % H, px].item(), r[py, px].item(), r[(py + 1) % H, px].item())
    dx = px + sub(r[py, (px - 1) % W].item(), r[py, px].item(), r[py, (px + 1) % W].item())
    return ((dx + W / 2) % W) - W / 2, ((dy + H / 2) % H) - H / 2


def refine_pairs(feats, pairs, init_txy, iters=200, reg_t=0.0, reg_rs=0.0,
                 verbose=True):
    """Batched similarity refinement of all (i, j) pairs on feature planes.

    feats: (Z, C, h, w) torch on device; pairs: list of (i, j); init_txy:
    (P, 2) coarse translations (feature px). Returns (P, 4) numpy
    [rot_deg, tx, ty, scale] (feature-px translations).

    Adjacent slices differ by REAL structural change (anisotropic Z), which a
    free L1 fit absorbs into the transform as spurious coherent motion; phase
    correlation is far less biased by it. reg_t pulls the translation toward
    the phase-corr init (quadratic, feat-px units) and reg_rs pulls rot (rad)
    / log-scale toward 0, so the refine only claims what the image evidence
    clearly supports beyond the coarse estimate."""
    dev = feats.device
    moving = feats[[i for i, _ in pairs]]
    fixed = feats[[j for _, j in pairs]]
    P, C, h, w = moving.shape
    rot = torch.zeros(P, device=dev, requires_grad=True)
    txy = torch.tensor(init_txy, dtype=torch.float32, device=dev, requires_grad=True)
    init = txy.detach().clone()
    log_s = torch.zeros(P, device=dev, requires_grad=True)
    ones = torch.ones(P, 1, h, w, device=dev)
    opt = torch.optim.Adam([{'params': [txy], 'lr': 0.1},
                            {'params': [rot], 'lr': 0.01},
                            {'params': [log_s], 'lr': 0.005}])
    for it in range(iters):
        opt.zero_grad()
        theta = affine.theta_from_params(rot, txy[:, 0], txy[:, 1], log_s, h, w)
        warped = affine.warp_slices(moving, theta, is_theta=True)
        mask = affine.warp_slices(ones, theta, is_theta=True) > 0.99
        loss = ((warped - fixed).abs() * mask).sum() / (mask.sum() * C + 1)
        loss = loss + reg_t * ((txy - init) ** 2).mean() \
                    + reg_rs * (rot ** 2 + log_s ** 2).mean()
        loss.backward()
        opt.step()
        if verbose and (it % 50 == 0 or it == iters - 1):
            print(f'  refine iter {it:4d}  loss {loss.item():.5f}')
    return np.stack([np.rad2deg(rot.detach().cpu().numpy()),
                     txy.detach().cpu().numpy()[:, 0],
                     txy.detach().cpu().numpy()[:, 1],
                     np.exp(log_s.detach().cpu().numpy())], axis=1)


def register_features(feats, stride, offsets, iters, reg_t=0.0, reg_rs=0.0,
                      anchor=0.0, solver='anchor', tv=4.0, tv_lin=2.0):
    """Feature planes -> absolute per-slice transforms (Z, 2, 3), pixel units.

    solver='anchor' = affine.solve_graph (identity prior, weight `anchor`);
    solver='tv' = graph_tv.solve_graph_tv (group fused-lasso on increments,
    penalties `tv` px / `tv_lin` px-equiv — one prior for drift, chunk tears
    and walks alike)."""
    Z = feats.shape[0]
    pairs = [(z, z + o) for o in offsets for z in range(Z - o)]
    init = np.array([phase_corr_shift(feats[j], feats[i]) for i, j in pairs],
                    dtype=np.float32)
    print(f'{len(pairs)} pairs (offsets {offsets}), coarse |t| '
          f'median {np.median(np.abs(init)):.2f} feat-px')
    refined = refine_pairs(feats, pairs, init, iters=iters, reg_t=reg_t, reg_rs=reg_rs)
    measurements = [(i, j, affine.params_to_mat(r, tx * stride, ty * stride, s))
                    for (i, j), (r, tx, ty, s) in zip(pairs, refined)]
    if solver == 'tv':
        from registration.graph_tv import solve_graph_tv
        mats = solve_graph_tv(measurements, Z, tv=tv, tv_lin=tv_lin)
    else:
        mats = affine.solve_graph(measurements, Z, anchor=anchor)
    return mats, pairs, refined


def main():
    parser = argparse.ArgumentParser(description='Latent-based per-slice registration')
    parser.add_argument('--dir', required=True, help="perturb.py's experiment dir")
    parser.add_argument('--feat', choices=('latent', 'codec', 'pixel'), default='latent')
    parser.add_argument('--method', choices=('feat', 'xcorr'), default='feat',
                        help="feat = latent-plane registration; xcorr = repo's "
                             'translation-only image-space baseline')
    parser.add_argument('--pairs', default='1,2,4',
                        help='Comma-separated slice-pair offsets (default 1,2,4)')
    parser.add_argument('--iters', type=int, default=200, help='Refine steps (default 200)')
    parser.add_argument('--reg_t', type=float, default=5.0,
                        help='Refine: pull translation toward the phase-corr init '
                             '(0 = free; see refine_pairs, default 5.0 — tuned on an '
                             'identity-perturbation run to a ~1 px bias floor)')
    parser.add_argument('--reg_rs', type=float, default=10.0,
                        help='Refine: pull rotation/log-scale toward 0 (default 10.0)')
    parser.add_argument('--anchor', type=float, default=0.05,
                        help='Graph solve: weak minimum-norm prior pulling each M_z '
                             'toward identity — damps the unobservable smooth drift '
                             'that structural change through Z leaks into long chains. '
                             '~1/(expected drift px); 0 = off (see affine.solve_graph)')
    parser.add_argument('--solver', choices=('anchor', 'tv'), default='anchor',
                        help="Graph solve: 'anchor' = affine.solve_graph "
                             "(identity prior); 'tv' = graph_tv fused-lasso on "
                             'increments — one prior for drift, chunk tears and '
                             "walks alike (label suffix '_tv')")
    parser.add_argument('--tv', type=float, default=4.0,
                        help='tv solver: translation-increment penalty (px)')
    parser.add_argument('--tv_lin', type=float, default=2.0,
                        help='tv solver: rot/scale-increment penalty (px-equiv)')
    parser.add_argument('--cache_feats', action='store_true',
                        help='Save/reuse features_{label}.npy in --dir (for sweeps)')
    parser.add_argument('--model', default='skipU', help='Registry model (default skipU)')
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--batch', type=int, default=8, help='Slices per encoder pass')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Additive Gaussian noise sigma (in [-1,1] units)')
    parser.add_argument('--gain', type=float, default=0.0,
                        help='Per-slice intensity gain jitter, U(1 +- gain)')
    parser.add_argument('--noise_seed', type=int, default=0)
    parser.add_argument('--save_codec', action='store_true',
                        help="codec feat: also save the indices npz ('codec_stack.npz')")
    args = parser.parse_args()

    corrupted = np.load(os.path.join(args.dir, 'corrupted.npy'))
    Z, Y, X = corrupted.shape
    label = 'xcorr' if args.method == 'xcorr' else args.feat
    if args.noise or args.gain:
        rng = np.random.default_rng(args.noise_seed)
        gains = 1 + rng.uniform(-args.gain, args.gain, size=Z).astype(np.float32)
        corrupted = ((corrupted + 1) * gains[:, None, None] - 1
                     + rng.normal(0, args.noise, corrupted.shape).astype(np.float32))
        label += '_noise'
    if args.method != 'xcorr' and args.solver == 'tv':
        label += '_tv'
    print(f'{args.dir}: ({Z}, {Y}, {X}), method={args.method}, label={label}')

    settings = dict(vars(args), label=label)
    if args.method == 'xcorr':
        from utils.alignments import align_stack_xy
        # Used as-is on the [-1, 1] stack. Known inadequacies on this data:
        # the unnormalized correlation's overlap bias pins the shifts near 0
        # (and a zero-mean variant instead locks onto large false peaks once
        # rotation + content change degrade the true peak) — it stands as the
        # naive translation-only image-space baseline.
        registered, shifts = align_stack_xy(corrupted, reference='first',
                                            subpixel=True, fill_value=-1.0)
        # shift (dy, dx) applied to slice z is the correction = M_z^-1 translation
        abs_mats = np.stack([affine.params_to_mat(0.0, -dx, -dy)
                             for dy, dx in shifts])
        pair_dump = None
    else:
        cache = os.path.join(args.dir, f'features_{label}.npy')
        if args.cache_feats and os.path.exists(cache):
            feats, stride = np.load(cache), features.STRIDE
            print(f'features from cache {cache}')
        else:
            eng = None
            if args.feat != 'pixel':
                eng = features.get_engine(args.model, args.epoch, args.device, args.half)
            codec_out = (os.path.join(args.dir, 'codec_stack.npz')
                         if args.save_codec and args.feat == 'codec' else None)
            feats, stride = features.extract(corrupted, args.feat, eng=eng,
                                             batch=args.batch, half=args.half,
                                             codec_out=codec_out)
            if args.cache_feats:
                np.save(cache, feats)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'features: {feats.shape} (stride {stride})')
        feats_t = torch.from_numpy(feats).to(device)
        offsets = [int(o) for o in args.pairs.split(',')]
        abs_mats, pairs, refined = register_features(feats_t, stride, offsets, args.iters,
                                                     reg_t=args.reg_t, reg_rs=args.reg_rs,
                                                     anchor=args.anchor, solver=args.solver,
                                                     tv=args.tv, tv_lin=args.tv_lin)
        pair_dump = [dict(i=i, j=j, rot_deg=float(r), tx=float(tx * stride),
                          ty=float(ty * stride), scale=float(s))
                     for (i, j), (r, tx, ty, s) in zip(pairs, refined)]
        inv = np.stack([affine.invert(m) for m in abs_mats])
        t = torch.from_numpy(corrupted)[:, None].to(device)
        with torch.no_grad():
            registered = affine.warp_slices(t, inv, mode='bicubic',
                                            fill=-1.0)[:, 0].cpu().numpy()

    np.save(os.path.join(args.dir, f'registered_{label}.npy'), registered)
    out = dict(label=label, settings={k: v for k, v in settings.items()
                                      if isinstance(v, (str, int, float, bool, type(None)))},
               abs_params=[affine.mat_to_params(m) for m in abs_mats],
               abs_mats=[m.tolist() for m in abs_mats])
    if pair_dump is not None:
        out['pair_measurements'] = pair_dump
    with open(os.path.join(args.dir, f'recovered_{label}.json'), 'w') as f:
        json.dump(out, f, indent=2)

    tmag = np.linalg.norm(np.stack(
        [np.asarray(m)[:, 2] for m in abs_mats]), axis=1)
    print(f'recovered |t|: median {np.median(tmag):.1f} px, max {tmag.max():.1f} px')
    print(f'-> registered_{label}.npy, recovered_{label}.json')


if __name__ == '__main__':
    main()
