from models.vqclean import GAN as VQCleanGAN
import torch
import torch.nn as nn
import yaml

from models.CUT import PatchSampleF3D
from networks.networks_cut import init_net, PatchNCELoss
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


# vqcleanCUT: vqclean + the CUT PatchNCE loss restored from ae0iso0tccutvqq.
#
# The NCE term contrasts encoder features of the real input XY slices (feat_q,
# captured during generation()'s forward) against encoder features of the
# matching slices of the generated volume XupX (feat_k, re-encoded in
# backward_g). Positive pairs are the same spatial patch in both stacks, so it
# pulls the generated slices toward the input's patch-level feature statistics
# — a structure-preservation signal on top of the L1 projection.
#
# Ported 1:1 from ae0iso0tccutvqq (--nocut off there ran this exact math):
#   - feature taps: encoder hs[1::2] -> channels ch*ch_mult per level
#     ([64, 128, 128, 256] for vqgan.yaml)
#   - feat_k slices: XupX at the Z positions aligned with the input slices
#     ((uprate//2)::uprate — the old hardcoded [4::8] at uprate 8)
#   - --nocut recovers vqclean exactly.
# Differences vs the old model:
#   - netF's MLPs (--use_mlp) are added to opt_ae here; the old model never
#     optimized them. With use_mlp off (the historical setting) netF is
#     parameter-free and this changes nothing.
#   - lbNCE defaults to 1 (this model exists to run CUT; old default was 0).
#
# CUT is batch-1 only: the feature stacks are per-volume (B folded nowhere),
# same limitation as vqclean.generation()'s historical batch-1 permutes.
#
# NOTE (snapshot): train.py copies only models/vqcleanCUT.py into the
# checkpoint dir; this file subclasses models/vqclean.py, so reproducing from
# a snapshot also needs vqclean.py at the run's logged git hash.
class GAN(VQCleanGAN):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        VQCleanGAN.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        self._hz = None

        if not self.hparams.nocut:
            if self.hparams.batch_size != 1:
                raise ValueError('vqcleanCUT NCE needs batch_size 1 '
                                 '(got %d); pass --nocut for larger batches'
                                 % self.hparams.batch_size)

            # Tap channels follow the encoder: hs[1::2] picks each level's
            # res-block output, so channels are ch * ch_mult[i] (valid for
            # num_res_blocks=1, as in all current vqgan yamls).
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
            self.netF.create_mlp(feature_shapes)

            if self.hparams.fWhich is None:
                self.hparams.fWhich = [1 for _ in range(len(feature_shapes))]
            print(self.hparams.fWhich)

            self.criterionNCE = []
            for _ in range(len(feature_shapes)):
                self.criterionNCE.append(PatchNCELoss(opt=hparams))

            self.netg_names['netF'] = 'netF'
            # Rebuild so opt_ae includes netF's MLPs from the start (PL calls
            # configure_optimizers() again at fit; this keeps self.optimizer_g
            # and the schedulers consistent in the meantime).
            self.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = VQCleanGAN.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("CUT")
        parser.add_argument('--nocut', action='store_true',
                            help='disable the NCE loss entirely (recovers vqclean)')
        parser.add_argument('--lbNCE', type=float, default=1.0,
                            help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--num_patches', type=int, default=256,
                            help='number of patches per layer')
        parser.add_argument('--nce_T', type=float, default=0.07,
                            help='temperature for NCE loss')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='(single image translation) include negatives from the '
                                 'other samples of the minibatch in the contrastive loss')
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument('--c_mlp', dest='c_mlp', type=int, default=256,
                            help='channel of mlp')
        parser.add_argument('--fWhich', nargs='+', type=int, default=None,
                            help='which layers to have NCE loss')
        return parent_parser

    def encode(self, x):
        """vqclean.encode + stashing the CUT feature taps during training.

        hs[1::2] -> per-level res-block outputs as (1, C, X, Y, Z) volumes,
        the layout PatchSampleF3D expects (input x is a (Z, C, X, Y) slice
        batch of a single volume).
        """
        h, hbranch, hz = self.encoder(x)

        if self.training and not self.hparams.nocut:
            hz = hz[1::2]
            self._hz = [f.permute(1, 2, 3, 0).unsqueeze(0) for f in hz]

        if self.use_vae:
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return posterior, None, None, h, None
        else:
            h = self.quant_conv(h)
            quant, emb_loss, info = self.quantize(h)
            return quant, emb_loss, info, h, None

    def generation(self, batch, deterministic=False):
        super().generation(batch, deterministic)
        if self.training and not self.hparams.nocut:
            self.goutz = self._hz  # feat_q: encoder features of the real input slices

    def backward_g(self):
        loss_dict = super().backward_g()

        if not self.hparams.nocut:
            feat_q = self.goutz

            # feat_k: re-encode the generated slices at the Z positions aligned
            # with the input slices (old hardcoded [4::8], i.e. uprate 8)
            input_slice_k = self.XupX.permute(4, 1, 2, 3, 0)[
                (self.uprate // 2)::self.uprate, :, :, :, 0]  # (Z, C, X, Y)
            self.encode(input_slice_k)  # refreshes self._hz
            feat_k = self._hz

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
        # Unlike the old model, netF's MLPs (--use_mlp) are trained: fold them
        # into opt_ae. Without --use_mlp netF is parameter-free (no-op).
        if hasattr(self, 'netF'):
            params = [p for p in self.netF.parameters() if p.requires_grad]
            if params:
                opts[0].add_param_group({'params': params})
        return opts, scheds
