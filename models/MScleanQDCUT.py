from models.MScleanQD2 import GAN as MScleanQD2GAN
import torch
import yaml

from models.CUT import PatchSampleF3D
from networks.networks_cut import init_net, PatchNCELoss
from ldm.modules.ema import LitEma


# MScleanQDCUT: MScleanQD2 + the CUT PatchNCE loss, with the q/k roles SWAPPED
# relative to the historical ae0iso0tccutvqq wiring.
#
# PatchNCELoss detaches feat_k internally, so only the feat_q side carries
# gradient. Here feat_q = encoder features of the GENERATED volume's slices
# (re-encoded in backward_g) and feat_k = features of the real input slices
# (captured during generation()) — true CUT semantics: the NCE gradient flows
# through XupX into net_g, pulling the generated slices toward the input's
# patch-level feature statistics. The old model had q=input/k=generated, which
# only regularized the 2D encoder. --nocut recovers MScleanQD2 exactly.
#
# Tap capture: MSclean.encode() discards the encoder's intermediate features
# (h, _, _ = self.encoder(x)), so a forward hook on self.encoder stashes
# hs[1::2] (per-level res-block outputs, channels ch*ch_mult — [64,128,128,256]
# for vqgan.yaml; the 1::2 pattern assumes num_res_blocks=1). The hook fires on
# every encoder call; consumers read the stash right after the call they own.
# feat_q uses self.encoder(...) directly — no need to run the residual-VQ loop
# on generated slices just to read encoder taps.
#
# Batch support: unlike vqcleanCUT (batch-1), taps are folded to (B, C, H, W, Z)
# volumes via slices_to_vol, PatchSampleF3D's native layout; PatchNCELoss
# groups negatives per volume via opt.batch_size. Upstream limitation kept:
# the same patch_ids are used for every volume in the batch (models/CUT.py).
#
# EMA: MSclean builds LitEma in its __init__, and LitEma.forward KeyErrors on
# any trainable param added afterwards — so when netF has params (--use_mlp),
# model_ema is REBUILT here after netF exists. (The same hazard exists in
# MScleanQD for net_d_prefix with --adv_prefix > 0.) netF MLPs are only
# created under --use_mlp, so without it there are no new params at all.
#
# NOTE (snapshot): train.py copies only models/MScleanQDCUT.py into the
# checkpoint dir; the chain MSclean.py -> MScleanQD.py -> MScleanQD2.py is
# also needed at the run's logged git hash to reproduce from a snapshot.
class GAN(MScleanQD2GAN):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        MScleanQD2GAN.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        self._cut_taps = None   # raw hs[1::2] of the last encoder call (training only)
        self._k_taps = None     # input-side taps in volume form, set by generation()

        if not self.hparams.nocut:
            # Tap channels follow the encoder: hs[1::2] picks each level's
            # res-block output, so channels are ch * ch_mult[i].
            with open('ldm/' + self.hparams.ldmyaml + '.yaml', "r") as f:
                ddconfig = yaml.load(f, Loader=yaml.Loader)['model']['params']['ddconfig']
            feature_shapes = [ddconfig['ch'] * m for m in ddconfig['ch_mult']]

            netF = PatchSampleF3D(
                use_mlp=self.hparams.use_mlp,
                init_type='normal',
                init_gain=0.02,
                gpu_ids=[],
                nc=self.hparams.c_mlp
            )
            self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
            if self.hparams.use_mlp:
                self.netF.create_mlp(feature_shapes)
                # New trainable params after LitEma was built -> rebuild it so
                # LitEma.forward's name lookup covers them (see header).
                self.model_ema = LitEma(self, decay=getattr(hparams, 'ema_decay', 0.9999))
                print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))} (rebuilt for netF).")

            if self.hparams.fWhich is None:
                self.hparams.fWhich = [1 for _ in range(len(feature_shapes))]
            print(self.hparams.fWhich)

            self.criterionNCE = []
            for _ in range(len(feature_shapes)):
                self.criterionNCE.append(PatchNCELoss(opt=hparams))

            self.netg_names['netF'] = 'netF'

            def _tap_hook(module, inputs, output):
                if module.training:
                    self._cut_taps = output[2][1::2]
            self.encoder.register_forward_hook(_tap_hook)

            # Rebuild so opt_ae includes netF's MLPs from the start (PL calls
            # configure_optimizers() again at fit; this keeps self.optimizer_g
            # and the schedulers consistent in the meantime).
            self.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = MScleanQD2GAN.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("CUT")
        parser.add_argument('--nocut', action='store_true',
                            help='disable the NCE loss entirely (recovers MScleanQD2)')
        parser.add_argument('--lbNCE', type=float, default=1.0,
                            help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--num_patches', type=int, default=256,
                            help='number of patches per layer')
        parser.add_argument('--nce_T', type=float, default=0.07,
                            help='temperature for NCE loss')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='include negatives from the other volumes of the '
                                 'minibatch in the contrastive loss')
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument('--c_mlp', dest='c_mlp', type=int, default=256,
                            help='channel of mlp')
        parser.add_argument('--fWhich', nargs='+', type=int, default=None,
                            help='which layers to have NCE loss')
        return parent_parser

    def generation(self, batch, deterministic=False):
        super().generation(batch, deterministic)
        if self.training and not self.hparams.nocut:
            # feat_k source: encoder taps of the real input slices, folded to
            # (B, C, H, W, Z) volumes (detached inside PatchNCELoss).
            B = self.oriX.shape[0]
            self._k_taps = [self.slices_to_vol(f, B) for f in self._cut_taps]

    def backward_g(self):
        loss_dict = super().backward_g()

        if not self.hparams.nocut:
            feat_k = self._k_taps  # grab BEFORE re-encoding overwrites the stash

            # feat_q: encoder taps of the generated slices at the Z positions
            # aligned with the input slices — WITH grad, this is the NCE path
            # into net_g. Direct encoder call: the hook captures the taps, the
            # residual-VQ loop is not needed here.
            B = self.XupX.shape[0]
            x = self.XupX[:, :, :, :, (self.uprate // 2)::self.uprate]
            self.encoder(self.vol_to_slices(x))
            feat_q = [self.slices_to_vol(f, B) for f in self._cut_taps]

            feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches, None)
            feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)

            total_nce_loss = 0.0
            for f_q, f_k, crit, f_w in zip(feat_q_pool, feat_k_pool,
                                           self.criterionNCE, self.hparams.fWhich):
                loss = crit(f_q, f_k) * f_w
                total_nce_loss += loss.mean()
            loss_nce = total_nce_loss / len(feat_q_pool)

            loss_dict['nce'] = loss_nce
            loss_dict['sum'] = loss_dict['sum'] + loss_nce * self.hparams.lbNCE

        return loss_dict

    def configure_optimizers(self):
        opts, scheds = super().configure_optimizers()
        # netF's MLPs (--use_mlp) train with the generator; without --use_mlp
        # netF is parameter-free and this is a no-op.
        if hasattr(self, 'netF'):
            params = [p for p in self.netF.parameters() if p.requires_grad]
            if params:
                opts[0].add_param_group({'params': params})
        return opts, scheds
