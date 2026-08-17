# MScleanSup0a — MScleanSup0 with the side-view loss changed from voxel L1 to a
# fine-Z SPECTRUM match (--xy_mode spec, default), with the voxel-L1 variants kept
# as --xy_mode l1 (== Sup0) and --xy_mode hp (L1 on fine-Z detail only).
#
# Why (measured on E2507218fuse, 2026-08-17; run 5a1258ce = Sup0 lambxy10):
#   Sup0's l1_x = L1(pool_X(XupX), pool_X(xcube)) plateaus at ~0.068 from epoch ~7,
#   equal to the score of the Z-interpolated input (0.067): the three views disagree
#   even at COARSE level (blur zcube and xcube to a common resolution: L1 ~0.06,
#   foreground corr 0.3-0.5 — different-direction sectioning + per-patch min-max), and
#   that irreducible term (~94% of the loss) contradicts the main Z projection l1.
#   Worse, the FINE-Z detail is not shared voxel-wise either: corr(hpZ(xcube),
#   hpZ(ycube)) ~ 0.25, and under L1 a volume with NO fine Z (interpolated input) scores
#   20-30% better than a real HR-Z view. So any voxel-wise loss to the side views —
#   plain or high-passed — is minimised by suppressing fine Z detail. That is what Sup0
#   trained: sparse, dim output that never gained Z structure.
#
# What the side views DO reliably carry is the statistics of fine-Z structure: how much
# there is and at what scales. 'spec' matches, per side view, the log power spectrum
# along Z of the fine-Z detail of the projected output to that of the real view:
#     px = pool_X(XupX), tx = pool_X(xcube)          (both (B,C,Y,X/aniso,Z))
#     d  = v - up(pool_Z(v))                          (fine-Z detail)
#     S(v) = log mean_{c,y,x} |rfft_Z(d)|^2           (B, nfreq)
#     l1_x = mean |S(px) - S(tx)|                     (same for y with pool_Y / ycube)
# On real cubes: interpolated input 6.8, input+white noise 5.5, real HR-Z view 1.2 —
# it forbids blur and demands the right texture scale, and is invariant to the 2-4 voxel
# registration offset and per-patch gain. Placement stays the job of the main Z l1 and
# the six-way discriminator, as before.
#
# Metrics: 'l1_x'/'l1_y' = the active side losses (mode-dependent scale: spec is in log
# power units, ~1-7). 'l1_x_raw'/'l1_y_raw' (no grad) and val_l1_x/val_l1_y (base.py)
# stay the unmodified Sup0 voxel L1s for comparability (expect ~0.06 regardless).
#
# Flags: --xy_mode {spec,hp,l1} (default spec), --xy_shift (hp only: min L1 over
# +-s-voxel target shifts along Z and the in-plane sharp axis; default 2). Everything
# else as MScleanSup0 (same geometry rules: cropz == cropsize, dsp == aniso == 8,
# --downbranch 1). No new parameters — MSclean / MScleanSup0 checkpoints load directly.
# --lamb_xy needs re-tuning for spec (different units): start ~1.
import torch
import torch.nn.functional as F
from models.MScleanSup0 import GAN as Sup0GAN


class GAN(Sup0GAN):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = Sup0GAN.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MScleanSup0a")
        parser.add_argument("--xy_mode", type=str, default='spec', choices=['spec', 'hp', 'l1'],
                            help="side-view loss: 'spec' = match fine-Z log power spectrum of the "
                                 "projected output to the real view (default); 'hp' = voxel L1 on "
                                 "fine-Z detail only; 'l1' = plain Sup0 voxel L1")
        parser.add_argument("--xy_shift", type=int, default=2,
                            help="hp mode only: min L1 over target shifts in {-s,0,+s} voxels along Z "
                                 "and the in-plane sharp axis (0 = off)")
        return parent_parser

    def _hp_z(self, v):
        """Fine-Z detail: v minus its Z-blurred (mean-pool by --aniso, interpolated back) copy."""
        r = self.hparams.aniso
        coarse = F.avg_pool3d(v, kernel_size=(1, 1, r), stride=(1, 1, r))
        coarse = F.interpolate(coarse, size=v.shape[2:], mode='trilinear', align_corners=False)
        return v - coarse

    def _zspec(self, v):
        """Log power spectrum along Z of the fine-Z detail, averaged over C, Y, X -> (B, nfreq)."""
        d = self._hp_z(v).float()
        p = torch.fft.rfft(d, dim=4).abs().pow(2)      # (B,C,Y,X,nfreq)
        return torch.log(p.mean(dim=(1, 2, 3)) + 1e-8)

    def _side_loss(self, pred, tgt, plane_axis):
        """pred/tgt: pooled side-view grids (B,C,·,·,Z). plane_axis = the sharp in-plane
        axis (dim2=Y for the x-side, dim3=X for the y-side), used by hp's shift search."""
        mode = self.hparams.xy_mode
        if mode == 'spec':
            return (self._zspec(pred) - self._zspec(tgt.detach())).abs().mean()
        if mode == 'hp':
            pred, tgt = self._hp_z(pred), self._hp_z(tgt)
        s = self.hparams.xy_shift if mode == 'hp' else 0
        if s <= 0:
            return self.add_loss_l1(pred, tgt)
        best = None
        for dz in (-s, 0, s):
            for dp in (-s, 0, s):
                l = self.add_loss_l1(pred, torch.roll(tgt, shifts=(dp, dz), dims=(plane_axis, 4)))
                best = l if best is None else torch.minimum(best, l)
        return best

    def backward_g(self):
        # Sup0's backward_g adds the plain L1s; call MSclean's instead and add ours.
        loss_dict = super(Sup0GAN, self).backward_g()

        if self.hparams.lamb_xy > 0 and len(getattr(self, 'aux_views', [])) >= 2:
            r = self.hparams.aniso
            xcube, ycube = self.aux_views[0], self.aux_views[1]
            assert self.XupX.shape[2:] == xcube.shape[2:] == ycube.shape[2:], \
                (f"shape mismatch: XupX {tuple(self.XupX.shape)} "
                 f"xcube {tuple(xcube.shape)} ycube {tuple(ycube.shape)}")
            how_xy = self.hparams.l1how_xy
            # pooled grids: x-side pools X (dim3) -> (B,C,Y,X/r,Z); y-side pools Y (dim2)
            px = self.get_projection(self.XupX, depth=r, how=how_xy, axis=3)
            tx = self.get_projection(xcube, depth=r, how=how_xy, axis=3)
            py = self.get_projection(self.XupX, depth=r, how=how_xy, axis=2)
            ty = self.get_projection(ycube, depth=r, how=how_xy, axis=2)

            loss_l1_x = self._side_loss(px, tx, plane_axis=2)
            loss_l1_y = self._side_loss(py, ty, plane_axis=3)
            loss_dict['l1_x'] = loss_l1_x
            loss_dict['l1_y'] = loss_l1_y
            loss_dict['sum'] = loss_dict['sum'] + (loss_l1_x + loss_l1_y) * self.hparams.lamb_xy

            # unmodified Sup0 voxel L1s, for comparison only
            with torch.no_grad():
                loss_dict['l1_x_raw'] = self.add_loss_l1(px, tx)
                loss_dict['l1_y_raw'] = self.add_loss_l1(py, ty)

        return loss_dict
