# MScleanSup1 — MSclean for the fused multi-view data with the MEASURED forward model:
# Gaussian axial PSF instead of box/max pooling, in the main Z projection AND the side-view
# (xcube/ycube) supervision, plus per-sample gain and a small registration tolerance
# for the side views.
#
# Why (E2507218fuse examination + PSF regression, 2026-08-17; see memory
# e2507218fuse-view-facts and the exam report):
#   * Each view's blurred axis is a Gaussian-like PSF of sigma ~12 vox (FWHM ~28; bracketed
#     8..16 by two estimators, held-out agreement flat for sigma>=12), NOT the 8-voxel
#     box/subsample that `--dsp 8`, `--l1how max/mean` and Sup0's pool8 imply. Under
#     the box model a *correct* sharp output can never match the (blurrier) data, so
#     every projection loss pushes the generator to blur itself, against the D. Held-out
#     cross-view agreement: no blur 0.45, box8 0.50, gauss8 0.58, gauss12 0.60.
#   * Patches are min-max'd per view -> per-patch cross-view gain 0.5-1.8x.
#   * Views are offset ~2 vox systematically along the blurred axis and +-2-3 vox per patch.
#   * Voxel agreement between views holds down to ~16 vox scales, not below.
#
# What this model does:
#   main l1 (--l1how psf): blur XupX along Z with G(sigma=--psf_sigma) and sample the SAME
#     planes the input came from (z_init + dsp*j), then L1 vs oriX. Coarse heads: sigma and
#     stride scaled by their uprate. Any other --l1how falls back to MSclean's behaviour.
#   side losses (--lamb_xy > 0, needs --direction zcube_xcube_ycube): blur XupX along X
#     with G(sigma) -> compare with xcube on the full grid (xcube is PSF-blurred + interpolated;
#     with sigma>>8 the interpolation is immaterial); same along Y for ycube.
#     --side_gain 1: closed-form per-sample affine (a,b) of the blurred prediction onto the
#     view (a,b detached), removes the per-patch gain scatter.
#     --side_shift s: L1 minimised over target shifts in {-s,0,+s} along all three axes
#     (27 combos; gradient flows through the selected one).
#   Metrics: 'l1_x'/'l1_y' = these side losses; 'l1_x_raw'/'l1_y_raw' and val_l1_x/y stay
#   the Sup0 pool8 quantities (l1how_xy) for comparability. Expect the raw ones ~0.06 always.
#
# Geometry as Sup0: cropz == cropsize, dsp == aniso == 8, --downbranch 1. No new network
# parameters — MSclean/Sup0 checkpoints load directly.
#
# Recipe: ... --models MScleanSup1 --direction zcube_xcube_ycube --l1how psf --psf_sigma 12
#         --lamb 5 --lamb_xy <w> --side_gain 1 --side_shift 2 --aniso 8 --dsp 8 --cropz 192 ...
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MScleanSup0 import GAN as Sup0GAN


class GAN(Sup0GAN):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = Sup0GAN.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MScleanSup1")
        parser.add_argument("--psf_sigma", type=float, default=12.0,
                            help="Gaussian axial PSF sigma in full-res voxels (fitted 12, plausible 10-16); "
                                 "used by --l1how psf and by the side losses")
        parser.add_argument("--side_gain", type=int, default=1,
                            help="1: fit a per-sample affine (a,b) of the blurred prediction onto each side "
                                 "view before the L1 (a,b detached)")
        parser.add_argument("--side_shift", type=int, default=2,
                            help="registration tolerance: min L1 over target shifts in {-s,0,+s} vox on all "
                                 "3 axes (0 = off)")
        return parent_parser

    # ---------------------------------------------------------------- generation (records z_init)
    def generation(self, batch, deterministic=False):
        # Copy of MSclean.generation with one addition: the dsp phase `self._z_init` is kept
        # so the PSF projection can sample XupX on exactly the planes the input came from.
        if self.hparams.cropz > 0 and self.training:
            z_init = 0 if deterministic else np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            for b in range(len(batch['img'])):
                batch['img'][b] = batch['img'][b][:, :, :, :, z_init:z_init + self.hparams.cropz]

        self.aux_views = [batch['img'][b] for b in range(1, len(batch['img']))]

        self._z_init = 0
        if self.hparams.dsp > 1:
            z_init = 0 if deterministic else np.random.randint(self.hparams.dsp)
            self._z_init = int(z_init)
            for b in range(len(batch['img'])):
                batch['img'][b] = batch['img'][b][:, :, :, :, z_init::self.hparams.dsp]

        if self.hparams.usp != 1:
            for b in range(len(batch['img'])):
                batch['img'][b] = nn.Upsample(scale_factor=(1, 1, self.hparams.usp),
                                              mode='trilinear')(batch['img'][b])

        self.oriX = batch['img'][0]
        slices = self.vol_to_slices(self.oriX)
        if self.training:
            slices = slices.requires_grad_(True)
        scale_latents, self.qloss, _ = self.encode(slices)
        self.reconstructions = self.decode(sum(scale_latents))
        self.scale_latents = scale_latents
        out = self._netg_decode(scale_latents, batch_size=self.oriX.shape[0])
        self.XupX = out['out0']
        self.XupX128 = out['out128']
        self.XupX64 = out['out64']
        self.gif_scales = [self.XupX128, self.XupX64]
        self.Xup = F.interpolate(self.oriX, size=self.XupX.shape[2:], mode='trilinear')

    # ---------------------------------------------------------------- PSF blur along one axis
    def _gauss1d(self, sigma, device, dtype):
        r = max(1, int(math.ceil(3 * sigma)))
        t = torch.arange(-r, r + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (t / max(sigma, 1e-3)) ** 2)
        return k / k.sum()

    def blur_axis(self, x, sigma, axis):
        """Gaussian blur of (B,C,Y,X,Z) along `axis` (2=Y, 3=X, 4=Z), replicate padding."""
        if sigma <= 0:
            return x
        B, C = x.shape[:2]
        k = self._gauss1d(sigma, x.device, x.dtype)
        r = (k.numel() - 1) // 2
        w = k.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1)          # blur along last dim
        xm = x.movedim(axis, 4)                                    # bring `axis` last
        pad = [r, r] + [0, 0] * 2                                  # F.pad pads last dim first
        xm = F.pad(xm, pad, mode='replicate')
        y = F.conv3d(xm, w, groups=C)
        return y.movedim(4, axis)

    # ---------------------------------------------------------------- projections
    def get_projection(self, x, depth, how='mean', uprate=None, axis=-1):
        if how != 'psf':
            return super().get_projection(x, depth, how=how, uprate=uprate, axis=axis)
        u = self.uprate if uprate is None else uprate
        scale = u / float(self.uprate)                              # 1, 1/2, 1/4 for the heads
        sigma = self.hparams.psf_sigma * scale
        if axis in (-1, 4):
            # main Z projection: blur along Z, then sample the planes the input came from
            # (phase z_init, stride = depth = uprate*skipl1 at this head's resolution)
            # clamp so every head yields the same number of planes as its target (Z - z0 > (n-1)*depth)
            z0 = min(int(round(getattr(self, '_z_init', 0) * scale)), depth - 1)
            xb = self.blur_axis(x, sigma, 4)
            return xb[:, :, :, :, z0::depth]
        # side view: blur along X (3) or Y (2), keep the full grid
        return self.blur_axis(x, sigma, axis)

    def _affine_fit(self, pred, tgt):
        """per-sample closed-form LS of tgt ~ a*pred + b; a,b detached -> (a*pred + b)"""
        B = pred.shape[0]
        p = pred.detach().reshape(B, -1)
        t = tgt.detach().reshape(B, -1)
        pm, tm = p.mean(1, keepdim=True), t.mean(1, keepdim=True)
        var = ((p - pm) ** 2).mean(1, keepdim=True) + 1e-8
        a = ((p - pm) * (t - tm)).mean(1, keepdim=True) / var
        a = a.clamp(0.25, 4.0)                                    # guard against degenerate fits
        b = tm - a * pm
        a = a.view(B, 1, 1, 1, 1); b = b.view(B, 1, 1, 1, 1)
        return a * pred + b

    def _side_l1(self, pred, tgt):
        if self.hparams.side_gain:
            pred = self._affine_fit(pred, tgt)
        s = self.hparams.side_shift
        if s <= 0:
            return self.add_loss_l1(pred, tgt)
        best = None
        for dy in (-s, 0, s):
            for dx in (-s, 0, s):
                for dz in (-s, 0, s):
                    t = torch.roll(tgt, shifts=(dy, dx, dz), dims=(2, 3, 4))
                    l = self.add_loss_l1(pred, t)
                    best = l if best is None else torch.minimum(best, l)
        return best

    def backward_g(self):
        # MSclean's backward_g (main l1 / coarse l1 go through get_projection -> 'psf' path)
        loss_dict = super(Sup0GAN, self).backward_g()

        if self.hparams.lamb_xy > 0 and len(getattr(self, 'aux_views', [])) >= 2:
            xcube, ycube = self.aux_views[0], self.aux_views[1]
            assert self.XupX.shape[2:] == xcube.shape[2:] == ycube.shape[2:], \
                (f"shape mismatch: XupX {tuple(self.XupX.shape)} "
                 f"xcube {tuple(xcube.shape)} ycube {tuple(ycube.shape)}")
            sig = self.hparams.psf_sigma
            px = self.blur_axis(self.XupX, sig, 3)     # xcube is blurred along X (dim3)
            py = self.blur_axis(self.XupX, sig, 2)     # ycube is blurred along Y (dim2)
            loss_l1_x = self._side_l1(px, xcube)
            loss_l1_y = self._side_l1(py, ycube)
            loss_dict['l1_x'] = loss_l1_x
            loss_dict['l1_y'] = loss_l1_y
            loss_dict['sum'] = loss_dict['sum'] + (loss_l1_x + loss_l1_y) * self.hparams.lamb_xy

            # Sup0's pool-based quantities, for comparison across runs only
            with torch.no_grad():
                r = self.hparams.aniso
                how = self.hparams.l1how_xy if self.hparams.l1how_xy != 'psf' else 'mean'
                loss_dict['l1_x_raw'] = self.add_loss_l1(
                    super().get_projection(self.XupX, depth=r, how=how, axis=3),
                    super().get_projection(xcube, depth=r, how=how, axis=3))
                loss_dict['l1_y_raw'] = self.add_loss_l1(
                    super().get_projection(self.XupX, depth=r, how=how, axis=2),
                    super().get_projection(ycube, depth=r, how=how, axis=2))

        return loss_dict
