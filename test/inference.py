"""The core inference engine — every test/ scenario runs through this script.

Loads any experiment's checkpoint (see test/load.py) and runs one of three
modes; the .sh files in test/ are scenario drivers on top of it
(inference3d.sh = mean-output sweep, inferencestd.sh = boundary-uncertainty
sweep, inference2d.sh = 2D VQ-head recon sweep).

--mode 3d (default): per-file isotropic inference, no stitching — each .tif is
    one patch pushed through the model's generation() path; the isotropic
    XupX is saved as float32 .tif.
--mode 2d: slice-wise 2D VQ-autoencoder reconstruction (encoder -> quantizer
    -> decoder head only, no 3D net_g, no Z upsampling) of ONE volume; output
    has the same (Z, Y, X) size as the input. Isolates how much detail
    survives the VQ bottleneck. Ignores --tta/--mc/--std_trd (the 2D head has
    no dropout — GroupNorm, dropout 0 — so passes would be identical).
--mode row: inference over a directory of adjacent thAAABBBCCC.tif patches
    (cut by tif_to_patches, global [-1, 1] intensity map) that form a
    contiguous run on exactly one index (AAA = Y, BBB = X, CCC = Z), then the
    outputs are concatenated along that axis into one strip that overlays the
    patch source's original.tif (XY page order always; --save_zx ignored).

Shared across modes: normalization mirrors training (the run's --nm; nm='11g'
applies the config's --gamma/--gamma_lo and by default INVERTS outputs back
to the pre-gamma [-1, 1] scale; --gamma_dec/--gamma_lo_dec re-tone the decode
only), and --tta/--mc pass pooling with std/maskstd maps (3d and row modes).

Usage:
    python test/inference.py \
        --checkpoint $LOGS/thx10-071226/vqcleanM0aMSskipU/roiD192gf/max5skip4 \
        --source /path/to/patches_dir_or_file.tif \
        --destination /home/gary/workspace/Data/THX10SDM20xw/out/skipU
    # --destination defaults to {DEFAULT_OUT}/{experiment name}
    # --dsp 1 to feed a real anisotropic volume as-is (default: config's dsp,
    #   which mimics training by Z-subsampling an isotropic input)
    python test/inference.py --mode 2d --checkpoint ... --source vol.tif \
        --out out/summary2d/skipU.tif
    python test/inference.py --mode row --checkpoint ... --source .../aRow \
        --tag skipU
"""

import argparse
import inspect
import os
import re
import sys
import time
from glob import glob
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Default base for all inference outputs (per-run subdir appended when
# --destination is not given).
DEFAULT_OUT = '/home/gary/workspace/Data/THX10SDM20xw/out'

import json  # noqa: E402
import numpy as np  # noqa: E402
import tifffile as tiff  # noqa: E402
import torch  # noqa: E402

from test.load import load_model  # noqa: E402


# --------------------------------------------------------------------------
# Engine: normalization and the forward passes
# --------------------------------------------------------------------------

def normalize(vol, nm, norm_stats=None, key=None, gamma=0.25, gamma_lo=-1.0):
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
    elif nm == '11g':  # floor + compressive gamma; input must already be in [-1, 1]
        vol = np.clip((vol - gamma_lo) / (1 - gamma_lo), 0, 1)
        vol = vol ** gamma * 2 - 1
    return vol  # '00': untouched


def invert_gamma(vol, gamma, gamma_lo):
    """Map an nm='11g' gamma-space volume back to the pre-gamma [-1, 1] scale.

    Exact inverse of normalize(..., '11g') for values above the --gamma_lo
    noise floor; the forward clip flattened the floor band, so it returns as
    a constant gamma_lo. Model outputs slightly outside [-1, 1] are clipped.
    """
    u = np.clip((vol + 1) / 2, 0, 1) ** (1.0 / gamma)
    return u * (1 - gamma_lo) + gamma_lo


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
def infer_volume(gan, vol, device, tta=False, mc=1, std_trd=None):
    """Run one (Y, X, Z) float32 volume through gan.generation().

    tta=True adds an XY-transposed pass (input Y<->X, output transposed back)
    per draw; mc=N repeats everything N times, so mc * (2 if tta else 1)
    passes are pooled (--tta --mc 5 = 10 runs). Under the default train-mode
    generator every pass is an independent MC-dropout draw. The output is the
    voxelwise mean over all passes and std their population std (None for a
    single pass) — both aggregated in model-output space, before any gamma
    inversion.

    std_trd (model-output space; the CLI maps its output-scale value here)
    additionally binarizes every pass at the threshold and returns the mask's
    across-pass std sqrt(p*(1-p)), p = fraction of passes >= std_trd — zero
    where all passes agree, 0.5 at a 50/50 split, so the nonzero band is the
    uncertain foreground boundary.

    Returns (xupx, xup, std, mask_std): mean isotropic output, trilinear-
    upsampled input, across-pass std map, and boundary-uncertainty map (the
    last two None for a single pass / no std_trd), all (Y, X, Z) numpy arrays.
    """
    t = torch.from_numpy(vol.astype(np.float32))[None, None].to(device)  # (1,1,Y,X,Z)
    deterministic = 'deterministic' in inspect.signature(gan.generation).parameters

    def run(x):
        # fresh dict per pass: generation() reassigns batch['img'] in place
        if deterministic:
            gan.generation({'img': [x]}, deterministic=True)
        else:
            gan.generation({'img': [x]})
        return gan.XupX

    t_swap = t.permute(0, 1, 3, 2, 4).contiguous() if tta else None
    mean = m2 = xup = mask_sum = None  # Welford: stable where sum-of-squares cancels
    n = 0
    for _ in range(max(mc, 1)):
        passes = [run(t)]
        if xup is None:
            xup = gan.Xup  # from the first original-orientation pass
        if tta:
            passes.append(run(t_swap).permute(0, 1, 3, 2, 4))
        for x in passes:
            x = x.float()
            n += 1
            if mean is None:
                mean, m2 = x.clone(), torch.zeros_like(x)
            else:
                delta = x - mean
                mean.add_(delta, alpha=1.0 / n)
                m2.addcmul_(delta, x - mean)
            if std_trd is not None:
                m = (x >= std_trd).float()
                mask_sum = m if mask_sum is None else mask_sum.add_(m)
    std = m2.div_(n).sqrt_()[0, 0].cpu().numpy() if n > 1 else None
    mask_std = None
    if mask_sum is not None and n > 1:
        p = mask_sum.div_(n)  # Bernoulli: std is exactly sqrt(p*(1-p))
        mask_std = (p * (1 - p)).sqrt_()[0, 0].cpu().numpy()
    return mean[0, 0].cpu().numpy(), xup[0, 0].float().cpu().numpy(), std, mask_std


@torch.no_grad()
def recon_volume(gan, vol, device):
    """(Y, X, Z) float32 volume -> slice-wise 2D VQ-head reconstruction, (Z, Y, X).

    The head every model runs before the 3D generator: `self.reconstructions
    = self.forward(input_slice)[0]` inside generation(). No net_g, no Z
    upsampling — output size == input size.
    """
    t = torch.from_numpy(vol.astype(np.float32))[None, None].to(device)  # (1,1,Y,X,Z)
    input_slice = t.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]                # (Z,C,Y,X)
    dec = gan.forward(input_slice)[0]                                    # (Z,C,Y,X)
    return dec[:, 0].float().cpu().numpy()                               # (Z,Y,X)


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


# --------------------------------------------------------------------------
# CLI and shared setup
# --------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description='Patch-inference engine: 3d isotropic / 2d VQ-head recon / row strip')
    parser.add_argument('--mode', choices=['3d', '2d', 'row'], default='3d',
                        help='3d = per-file isotropic inference (default); 2d = slice-wise '
                             'VQ-head reconstruction of one volume; row = row-of-patches '
                             'inference + concatenated strip')
    parser.add_argument('--checkpoint', required=True,
                        help='Timestamped checkpoint dir, its checkpoints/ parent, or the experiment root')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch to load (default: latest)')
    parser.add_argument('--source', required=True,
                        help='3d: a .tif file or a directory of .tif files; 2d: one .tif; '
                             'row: a directory of thAAABBBCCC.tif patches')
    parser.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')
    parser.add_argument('--eval', dest='train_mode', action='store_false',
                        help='Use .eval() for the generator components (BatchNorm running stats, '
                             'dropout off -> deterministic). Default is .train() for MC dropout: '
                             'batch-stat BN and active dropout, stochastic per run. The GAN '
                             'module itself always stays eval so generation() still skips the '
                             'cropz training crop.')
    # normalization (shared)
    parser.add_argument('--nm', default=None, help="Normalization override (default: config's --nm)")
    parser.add_argument('--norm_stats', default=None,
                        help="norm_stats.json path for nm='11p' (key = tif parent dir name)")
    parser.add_argument('--gamma', type=float, default=None,
                        help="nm='11g' gamma override (default: config's --gamma)")
    parser.add_argument('--gamma_lo', type=float, default=None,
                        help="nm='11g' noise-floor override (default: config's --gamma_lo)")
    parser.add_argument('--gamma_dec', type=float, default=None,
                        help='Decode-only gamma for the output inversion (default: --gamma). '
                             'Smaller than --gamma darkens midtones; pure tone remap, the '
                             'model input keeps the trained transform')
    parser.add_argument('--gamma_lo_dec', type=float, default=None,
                        help='Decode-only floor for the output inversion (default: --gamma_lo)')
    parser.add_argument('--no_invert', action='store_true',
                        help="nm='11g': keep outputs in gamma space instead of inverting "
                             'them back to the pre-gamma [-1, 1] intensity scale')
    # pass pooling (3d + row)
    parser.add_argument('--tta', action='store_true',
                        help='Test-time augmentation: average the original pass with an '
                             'XY-transposed pass (transposed back). With the default train '
                             'mode each pass is also an independent MC-dropout draw.')
    parser.add_argument('--mc', type=int, default=1,
                        help='Monte Carlo repeats; combines with --tta (--tta --mc 5 = 10 '
                             'pooled runs). Needs the default train mode to actually vary. '
                             'When total passes > 1, the across-run std is saved (3d: '
                             'std/{model}/{stem}_std.tif next to --destination; row: '
                             '{tag}_std.tif strip). Model-output space, no gamma inversion.')
    parser.add_argument('--std_trd', type=float, default=None,
                        help='Foreground threshold for the pass-agreement mask, in the SAVED '
                             'output intensity scale (mapped through the forward gamma when '
                             "nm='11g' inversion is on). Each pass is binarized at it; the "
                             'mask std sqrt(p*(1-p)) — the uncertain-boundary map — is saved '
                             'as std/{model}/{stem}_maskstd.tif (row: {tag}_maskstd.tif '
                             'strip). Needs total passes > 1.')
    # 3d
    parser.add_argument('--limit', type=int, default=0,
                        help='3d, directory source: only process the first N files, sorted (0 = all)')
    parser.add_argument('--skip', type=int, default=0,
                        help='3d, directory source: skip the first N files before --limit applies')
    parser.add_argument('--destination', default=None,
                        help=f'3d: output dir (default: {DEFAULT_OUT}/{{experiment name}})')
    parser.add_argument('--dsp', type=int, default=None,
                        help="Z-subsample override; 1 = feed anisotropic volume as-is (default: config's --dsp)")
    parser.add_argument('--cropsize', type=int, default=0, help='3d: center-crop Y/X (0 = full)')
    parser.add_argument('--cropz', type=int, default=0, help='3d: center-crop Z before dsp (0 = full)')
    parser.add_argument('--save_input', action='store_true',
                        help='3d: also save the trilinear-upsampled input as {stem}.tif in an '
                             'input/ dir next to --destination (model dirs hold outputs only). '
                             "2d: save the round-tripped input as input.tif next to --out")
    parser.add_argument('--save_zx', action='store_true',
                        help='3d: save volumes in ZX page order (page y=k: rows Z, cols X) '
                             'instead of the default XY page order (page z=k) — puts the '
                             'synthesized Z axis in-plane for direct inspection')
    # 2d
    parser.add_argument('--out', default=None,
                        help=f'2d: output .tif path (default: {DEFAULT_OUT}/summary2d/{{experiment name}}.tif)')
    # row
    parser.add_argument('--tag', default=None,
                        help='row: output name — patches -> {source}/{tag}/, strip -> {source}/{tag}.tif')
    return parser


def setup(args):
    """Load the checkpoint and resolve the run's normalization; shared by all modes."""
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
    gamma = args.gamma if args.gamma is not None else (getattr(cfg, 'gamma', None) or 0.25)
    gamma_lo = args.gamma_lo if args.gamma_lo is not None else (getattr(cfg, 'gamma_lo', None) or -1.0)
    gamma_dec = args.gamma_dec if args.gamma_dec is not None else gamma
    gamma_lo_dec = args.gamma_lo_dec if args.gamma_lo_dec is not None else gamma_lo
    invert = nm == '11g' and not args.no_invert
    if nm == '11g':
        print(f"nm='11g': encode gamma={gamma}, gamma_lo={gamma_lo}; "
              f"decode gamma={gamma_dec}, gamma_lo={gamma_lo_dec}"
              + ('' if invert else ' (kept in gamma space)'))
    std_trd = args.std_trd
    if std_trd is not None and invert:
        # --std_trd is given in the saved (inverted) scale; passes are pooled in
        # model space, so map it through the forward gamma (inverse of invert_gamma)
        u = np.clip((std_trd - gamma_lo_dec) / (1 - gamma_lo_dec), 0, 1)
        std_trd = u ** gamma_dec * 2 - 1
        print(f'--std_trd {args.std_trd} (output scale) -> {std_trd:.4f} (model space)')

    norm_stats = None
    if nm == '11p':
        if args.norm_stats and os.path.isfile(args.norm_stats):
            with open(args.norm_stats) as f:
                raw = json.load(f)
            stats = raw.get('stats', raw)
            norm_stats = {k: (float(v[0]), float(v[1])) for k, v in stats.items()}
        else:
            print("WARNING: nm='11p' without --norm_stats — will fall back to '11'")

    return dict(gan=gan, cfg=cfg, device=device, nm=nm, norm_stats=norm_stats,
                gamma=gamma, gamma_lo=gamma_lo, gamma_dec=gamma_dec,
                gamma_lo_dec=gamma_lo_dec, invert=invert, std_trd=std_trd)


def load_normalized(path, ctx):
    """Read a .tif as (Y, X, Z) and apply the run's normalization."""
    vol = tiff.imread(path).astype(np.float32)          # (Z, Y, X)
    vol = np.transpose(vol, (1, 2, 0))                  # -> (Y, X, Z), Z last
    return normalize(vol, ctx['nm'], ctx['norm_stats'], key=Path(path).parent.name,
                     gamma=ctx['gamma'], gamma_lo=ctx['gamma_lo'])


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def run_3d(args, ctx):
    gan, device, invert = ctx['gan'], ctx['device'], ctx['invert']
    if os.path.isdir(args.source):
        files = sorted(glob(os.path.join(args.source, '*.tif'))
                       + glob(os.path.join(args.source, '*.tiff')))
        files = files[args.skip:]
        if args.limit > 0:
            files = files[:args.limit]
    else:
        files = [args.source]
    if not files:
        raise FileNotFoundError(f'No .tif files found at {args.source}')

    dest = args.destination or os.path.join(DEFAULT_OUT,
                                            os.path.basename(args.checkpoint.rstrip('/')))
    os.makedirs(dest, exist_ok=True)
    input_dir = os.path.join(os.path.dirname(dest.rstrip('/')) or '.', 'input')
    if args.save_input:
        os.makedirs(input_dir, exist_ok=True)
    # std maps go to a std/{model} dir next to --destination (like input/),
    # keeping the model dirs mean-outputs-only for concat_views/perceptual.
    std_dir = os.path.join(os.path.dirname(dest.rstrip('/')) or '.', 'std',
                           os.path.basename(dest.rstrip('/')))
    n_passes = max(args.mc, 1) * (2 if args.tta else 1)
    if n_passes > 1:
        os.makedirs(std_dir, exist_ok=True)
    print(f'{len(files)} file(s) -> {dest}  (nm={ctx["nm"]}, dsp={gan.hparams.dsp})'
          + (f'  pooling {n_passes} passes/volume (tta={args.tta}, mc={args.mc})'
             if n_passes > 1 else ''))

    times = []
    for path in files:
        vol = tiff.imread(path).astype(np.float32)          # (Z, Y, X)
        vol = np.transpose(vol, (1, 2, 0))                  # -> (Y, X, Z), Z last
        vol = center_crop(vol, args.cropsize, args.cropz)
        vol = normalize(vol, ctx['nm'], ctx['norm_stats'], key=Path(path).parent.name,
                        gamma=ctx['gamma'], gamma_lo=ctx['gamma_lo'])

        t0 = time.perf_counter()
        xupx, xup, xstd, xmstd = infer_volume(gan, vol, device, tta=args.tta,
                                              mc=args.mc, std_trd=ctx['std_trd'])
        times.append(time.perf_counter() - t0)          # .cpu() copy syncs the GPU
        if invert:
            xupx = invert_gamma(xupx, ctx['gamma_dec'], ctx['gamma_lo_dec'])
            xup = invert_gamma(xup, ctx['gamma_dec'], ctx['gamma_lo_dec'])

        stem = Path(path).stem
        # (Y, X, Z) -> tif page order: XY pages (Z, Y, X) or ZX pages (Y, Z, X)
        pages = (0, 2, 1) if args.save_zx else (2, 0, 1)
        tiff.imwrite(os.path.join(dest, stem + '.tif'),
                     np.transpose(xupx, pages).astype(np.float32))
        if xstd is not None:
            tiff.imwrite(os.path.join(std_dir, stem + '_std.tif'),
                         np.transpose(xstd, pages).astype(np.float32))
        if xmstd is not None:
            tiff.imwrite(os.path.join(std_dir, stem + '_maskstd.tif'),
                         np.transpose(xmstd, pages).astype(np.float32))
        if args.save_input:
            tiff.imwrite(os.path.join(input_dir, stem + '.tif'),
                         np.transpose(xup, pages).astype(np.float32))
        print(f'  {stem}: in (Z,Y,X)={vol.shape[2]}x{vol.shape[0]}x{vol.shape[1]}'
              f' -> out {xupx.shape[2]}x{xupx.shape[0]}x{xupx.shape[1]}'
              f'  range [{xupx.min():.3f}, {xupx.max():.3f}]'
              + (f'  std mean {xstd.mean():.4f}' if xstd is not None else '')
              + f'  {times[-1]:.2f}s')

    if times:
        steady = times[1:] if len(times) > 1 else times
        print(f'inference time: mean {sum(times) / len(times):.2f}s/vol over {len(times)}'
              f' (excl. first/warmup: {sum(steady) / len(steady):.2f}s/vol)')


def run_2d(args, ctx):
    gan, device, invert = ctx['gan'], ctx['device'], ctx['invert']
    if os.path.isdir(args.source):
        raise ValueError('--mode 2d takes a single .tif, not a directory')
    if args.tta or args.mc > 1 or args.std_trd is not None:
        print('NOTE: --mode 2d ignores --tta/--mc/--std_trd (the 2D head is deterministic)')
    vol = load_normalized(args.source, ctx)

    out = args.out or os.path.join(DEFAULT_OUT, 'summary2d',
                                   os.path.basename(args.checkpoint.rstrip('/')) + '.tif')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    if args.save_input:
        # When inverting, run the input through the same round trip so it is
        # directly comparable to the reconstruction (floor band flattened to gamma_lo).
        inp = invert_gamma(vol, ctx['gamma'], ctx['gamma_lo']) if invert else vol
        input_path = os.path.join(os.path.dirname(out) or '.', 'input.tif')
        tiff.imwrite(input_path, np.transpose(inp, (2, 0, 1)).astype(np.float32))
        print(f'input -> {input_path}')

    rec = recon_volume(gan, vol, device)
    if invert:
        rec = invert_gamma(rec, ctx['gamma_dec'], ctx['gamma_lo_dec'])
    tiff.imwrite(out, rec.astype(np.float32))
    print(f'{os.path.basename(out)}: in (Z,Y,X)={vol.shape[2]}x{vol.shape[0]}x{vol.shape[1]}'
          f' -> recon {rec.shape[0]}x{rec.shape[1]}x{rec.shape[2]}'
          f'  range [{rec.min():.3f}, {rec.max():.3f}]')


def run_row(args, ctx):
    gan, device, invert = ctx['gan'], ctx['device'], ctx['invert']
    if not args.tag:
        raise ValueError('--mode row requires --tag')
    files, cat_axis = row_files(args.source)
    dest = os.path.join(args.source, args.tag)
    os.makedirs(dest, exist_ok=True)

    outputs, stds, mstds = [], [], []
    for path in files:
        vol = load_normalized(path, ctx)
        xupx, _, xstd, xmstd = infer_volume(gan, vol, device, tta=args.tta,
                                            mc=args.mc, std_trd=ctx['std_trd'])
        if invert:
            xupx = invert_gamma(xupx, ctx['gamma_dec'], ctx['gamma_lo_dec'])
        outputs.append(xupx.astype(np.float32))
        if xstd is not None:
            stds.append(xstd.astype(np.float32))
        if xmstd is not None:
            mstds.append(xmstd.astype(np.float32))
        name = os.path.basename(path)
        # XY page order always — strips must overlay the patch source's original.tif
        tiff.imwrite(os.path.join(dest, name), np.transpose(xupx, (2, 0, 1)).astype(np.float32))
        print(f'  {name}: {vol.shape} -> {xupx.shape}  range [{xupx.min():.3f}, {xupx.max():.3f}]'
              + (f'  std mean {xstd.mean():.4f}' if xstd is not None else ''))

    for vols, suffix, note in ((outputs, '', 'saved as XY pages (pages Z, rows Y, cols X)'),
                               (stds, '_std', 'across-pass std strip (model-output space)'),
                               (mstds, '_maskstd', 'boundary-uncertainty strip')):
        if not vols:
            continue
        strip = np.concatenate(vols, axis=cat_axis)     # (Y, X, Z) with one long axis
        strip_path = os.path.join(args.source, args.tag + suffix + '.tif')
        tiff.imwrite(strip_path, np.transpose(strip, (2, 0, 1)).astype(np.float32))
        print(f'{strip_path}: (Y, X, Z) = {strip.shape}, {note}')


def main():
    args = build_parser().parse_args()
    ctx = setup(args)
    {'3d': run_3d, '2d': run_2d, 'row': run_row}[args.mode](args, ctx)


if __name__ == '__main__':
    main()
