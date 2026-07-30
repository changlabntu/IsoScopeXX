# MScleanSup0 — MSclean + the multi-view projection supervision of vqcleanM0aSup0,
# for fused data (--direction zcube_xcube_ycube): data0=zcube (input, LR in Z),
# data1=xcube (LR in X / HR in Z), data2=ycube (LR in Y / HR in Z). The isotropic
# output XupX is pooled along X (tensor dim3) by --aniso to match xcube and along Y
# (dim2) to match ycube; the pools keep Z full-res and the cubes are HR in Z, so
# XupX's Z is supervised with real high-res structure. Weighted by --lamb_xy.
#
# Everything else (multi-scale heads, per-scale discriminators, EMA, coarse L1,
# VQR options, B>1) is inherited from MSclean unchanged — this adds no parameters,
# so MSclean checkpoints load directly.
#
# --l1how_xy defaults to 'mean', NOT the 'max' of the main Z l1 (and of vqcleanM0aSup0's
# flag default). The views are box-averages of the truth along their blurred axis, and
# mean-pooling commutes with that average, so the loss is minimized at the true isotropic
# volume. Max does not commute: it compares the peak of 8 sharp voxels against the max of
# 8 pre-averaged ones (~their mean), a standing offset the generator can only close by
# flattening its own X/Y peaks. Measured: on a phantom, max scores the ideal volume 0.145
# vs 0.077 for a deliberately X-blurred copy — its minimum is not at the truth — while mean
# puts the ideal lowest and separates the correct axis pairing from a swapped one by 4.7x.
# On the real cubes, blurring a candidate along the pooled axis lowers the max loss 8-17%
# and leaves the mean loss flat. Confirmed in training (run 6bd5b84b, MS0728): under max,
# l1_x/l1_y fell BELOW the floor a Z-honest output can reach while val_spec_xy_hi decayed
# monotonically to half the unsupervised baseline's XY high-frequency retention.
# Max stays correct for the main Z l1, where the pooled axis is the one being
# super-resolved and the output is not asked to stay sharp along it.
#
# Run like the vqcleanM0aSup0 recipe: --direction zcube_xcube_ycube --nm 11p
# --aniso 8 --lamb_xy <w>, with cropz == cropsize and --dsp <aniso> (the shape
# assert below requires XupX and the aux cubes to share the full isotropic grid;
# uprate = aniso must stay divisible by 4 for MSclean's --lamb_coarse).
from models.MSclean import GAN as MScleanGAN


class GAN(MScleanGAN):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = MScleanGAN.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MScleanSup0")
        parser.add_argument("--lamb_xy", type=float, default=2.0,
                            help='weight for xcube/ycube projection supervision (0 disables it)')
        parser.add_argument("--aniso", type=int, default=8,
                            help='anisotropy ratio of the fused views = pool factor for the X/Y projection')
        parser.add_argument("--l1how_xy", type=str, default='mean',
                            help="pooling for the xcube/ycube X/Y projection supervision "
                                 "('mean' or 'max'); must match the views' physical forward "
                                 "model — see the note above, 'max' biases toward blurred X/Y")
        return parent_parser

    def get_projection(self, x, depth, how='mean', uprate=None, axis=-1):
        # MSclean's signature (uprate for the coarse heads' dsp mode) + Sup0's axis
        # argument so mean/max can pool along X (dim3) or Y (dim2), not just Z.
        if how == 'dsp':
            assert axis in (-1, 4), "'dsp' projection only strides the Z axis"
            u = self.uprate if uprate is None else uprate
            x = x[:, :, :, :, (u // 2)::u * self.hparams.skipl1]
        else:
            x = x.unfold(axis, depth, depth)  # pools `axis`; window goes to the last dim
            if how == 'mean':
                x = x.mean(dim=-1)
            elif how == 'max':
                x, _ = x.max(dim=-1)
        return x

    def backward_g(self):
        loss_dict = super().backward_g()

        # Multi-view projection supervision: xcube is LR along tensor dim3 / HR in Z,
        # ycube is LR along dim2 / HR in Z. Projecting XupX along those axes by the
        # anisotropy ratio (pool op = --l1how_xy) and matching the real views supervises
        # XupX's Z with genuine high-res structure (the pools keep Z full-res).
        # aux_views are captured full-Z in MSclean.generation(); --nm 11p puts all
        # views on a shared intensity scale.
        if self.hparams.lamb_xy > 0 and len(getattr(self, 'aux_views', [])) >= 2:
            r = self.hparams.aniso
            xcube, ycube = self.aux_views[0], self.aux_views[1]
            assert self.XupX.shape[2:] == xcube.shape[2:] == ycube.shape[2:], \
                (f"shape mismatch: XupX {tuple(self.XupX.shape)} "
                 f"xcube {tuple(xcube.shape)} ycube {tuple(ycube.shape)}")
            how_xy = self.hparams.l1how_xy
            loss_l1_x = self.add_loss_l1(
                a=self.get_projection(self.XupX, depth=r, how=how_xy, axis=3),
                b=self.get_projection(xcube, depth=r, how=how_xy, axis=3))
            loss_l1_y = self.add_loss_l1(
                a=self.get_projection(self.XupX, depth=r, how=how_xy, axis=2),
                b=self.get_projection(ycube, depth=r, how=how_xy, axis=2))
            loss_dict['l1_x'] = loss_l1_x
            loss_dict['l1_y'] = loss_l1_y
            loss_dict['sum'] = loss_dict['sum'] + (loss_l1_x + loss_l1_y) * self.hparams.lamb_xy

        return loss_dict
