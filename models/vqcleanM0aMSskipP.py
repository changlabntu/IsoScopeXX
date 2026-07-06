from models.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from ldm.modules.diffusionmodules.modelcut import Encoder, Decoder
from ldm.util import instantiate_from_config
import yaml
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from networks.networks import get_scheduler
import os
from pytorch_msssim import ms_ssim
import tifffile as tiff


# vqcleanM0aMSskipP: vqcleanM0aMSskipE + projection forward-model variants for the L1
# self-consistency loss (MS_refine2 projection probe). Requires --netG ed023emsfpn and
# num_scales >= 3; run with --pyr_detach --adv_ms 0.5.
#
# Inherited from vqcleanM0aMSskipE: full-sum trunk + zero-init lateral skips (step-0 ==
# vqcleanM0aMS), EMA always on for validation/checkpoints, real-data coarse supervision
# (--lamb_coarse, 'l1c').
#
# New here — the projection operator is the acquisition forward model, and the existing
# choices are crude corners (mean: blur pressure; max: gradient to ~1/depth of voxels,
# speckle-tolerant; dsp: pins exact planes). Added:
#   --l1how lse  — log-sum-exp soft-max projection: tau*(logsumexp(x/tau) - log W) over
#        the same windows as 'max' (the -log W term removes the positive bias; tau->0
#        recovers 'max', large tau approaches 'mean'). Gradient reaches every voxel,
#        softmax-weighted. Temperature --lse_tau.
#   --l1how band — band-limited DENSE L1: Gaussian-low-pass (sigma = rate/2) both the
#        prediction and the trilinear-upsampled observation along Z and L1 them at every
#        Z position. Full agreement in the frequency band the input actually measured,
#        complete freedom above it — anchoring without blur pressure on the high-Z
#        frequencies the GAN must synthesize. Applies to the coarse 'l1c' terms too.
#   --l1phase random — for window-based hows (mean/max/lse): shift the window start by a
#        per-step random phase r in [0, skipl1) (prediction by r*rate, observation to
#        obs[..., r::skipl1]) so every voxel is constrained in expectation rather than
#        the same sparse subset each step. One r is shared by the main and coarse losses.
class GAN(BaseModel):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        print('Reading yaml: ' + self.hparams.ldmyaml)
        with open('ldm/' + self.hparams.ldmyaml + '.yaml', "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        ddconfig = config['model']['params']["ddconfig"]

        if self.hparams.tc:
            ddconfig['in_channels'] = 2
            ddconfig['out_ch'] = 1
        self.hparams.netG = self.hparams.netG

        self.hparams.final = 'tanh'
        self.net_g, self.net_d = self.set_networks()

        # Separate per-scale discriminators for the coarse (1/2, 1/4) outputs. The patch
        # discriminator is fully convolutional, so it scores slices at any resolution.
        self.net_d_128 = self.set_networks(net='d')
        self.net_d_64 = self.set_networks(net='d')

        # VQGAN components
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.embed_dim = config['model']['params']['embed_dim']
        self.n_embed = config['model']['params']['n_embed']

        # Multi-scale VQ — scale_factors go coarse to fine, matching VAR Algorithm 1
        # e.g. num_scales=3 → scale_factors=[4, 2, 1]
        # num_scales=1 → scale_factors=[1] (identical to original single VQ)
        self.num_scales = getattr(hparams, 'num_scales', 1)
        if self.num_scales < 3:
            raise ValueError('vqcleanM0aMSskipP needs --num_scales >= 3 (two of the scales '
                             'are laterally injected); got %d' % self.num_scales)
        self.scale_factors = [2 ** (self.num_scales - 1 - i) for i in range(self.num_scales)]
        print('scale_factors: ' + str(self.scale_factors))

        # Per-scale quant_conv: z_channels → embed_dim (applied at downsampled resolution)
        self.quant_convs = nn.ModuleList([
            nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
            for _ in range(self.num_scales)
        ])

        # Per-scale post_quant_conv: embed_dim → z_channels
        # Applied AFTER upsampling back to full resolution, matching VAR ϕk
        self.post_quant_convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)
            for _ in range(self.num_scales)
        ])

        # Quantizers — separate codebook per scale by default
        # Set --shared_codebook to use a single codebook across all scales (VAR paper)
        self.shared_codebook = getattr(hparams, 'shared_codebook', False)
        if self.shared_codebook:
            shared_q = VectorQuantizer(
                self.n_embed, self.embed_dim, beta=0.25,
                remap=getattr(hparams, 'remap', None),
                sane_index_shape=getattr(hparams, 'sane_index_shape', False)
            )
            self.quantizers = nn.ModuleList([shared_q for _ in range(self.num_scales)])
        else:
            self.quantizers = nn.ModuleList([
                VectorQuantizer(
                    self.n_embed, self.embed_dim, beta=0.25,
                    remap=getattr(hparams, 'remap', None),
                    sane_index_shape=getattr(hparams, 'sane_index_shape', False)
                )
                for _ in range(self.num_scales)
            ])

        # Adapters projecting the injected VQ scales (z_channels, latent res) to the
        # net_g decode-stage channel counts (nf follows set_networks' nf=hparams.ngf).
        # net_g resizes them spatially at the injection point, so 1x1 convs suffice here.
        self.inject_convs = nn.ModuleList([
            nn.Conv3d(ddconfig["z_channels"], 4 * hparams.ngf, 1),   # 'pre_up2' (4*nf stage)
            nn.Conv3d(ddconfig["z_channels"], 2 * hparams.ngf, 1),   # 'pre_up1' (2*nf stage)
        ])
        # Zero-init (ControlNet-style): skips contribute nothing at step 0, so the model
        # starts numerically identical to vqcleanM0aMS and can only learn helpful skips.
        for m in self.inject_convs:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # Initialize loss
        self.loss = instantiate_from_config(config['model']['params']["lossconfig"])
        self.discriminator = self.loss.discriminator

        # EMA support
        # MSskipE (F1): EMA is always on — validation and epoch checkpoints use EMA weights.
        self.use_ema = True
        from ldm.modules.ema import LitEma
        self.model_ema = LitEma(self, decay=getattr(hparams, 'ema_decay', 0.9999))
        print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.netg_names = {
            'encoder': 'encoder',
            'decoder': 'decoder',
            'quantizers': 'quantizers',
            'quant_convs': 'quant_convs',
            'post_quant_convs': 'post_quant_convs',
            'inject_convs': 'inject_convs',
            'net_g': 'net_g'
        }
        self.netd_names = {
            'discriminator': 'discriminator',
            'net_d': 'net_d',
            'net_d_128': 'net_d_128',
            'net_d_64': 'net_d_64',
        }

        self.configure_optimizers()

        self.upsample = torch.nn.Upsample(
            size=(hparams.cropsize, hparams.cropsize, hparams.cropsize), mode='trilinear'
        )
        self.uprate = (hparams.cropsize // hparams.cropz) * hparams.dsp / hparams.usp
        self.uprate = int(self.uprate)
        if self.hparams.lamb_coarse > 0 and self.uprate % 4 != 0:
            raise ValueError('vqcleanM0aMSskipP coarse supervision (--lamb_coarse > 0) needs '
                             'uprate divisible by 4; got %d' % self.uprate)

        # Gaussian Z-kernels for l1how='band' (sigma = rate/2), one per output scale.
        # Registered as buffers so they follow the module across devices.
        for rate in sorted({self.uprate, max(self.uprate // 2, 1), max(self.uprate // 4, 1)}):
            sigma = rate / 2.0
            K = int(4 * sigma) // 2 * 2 + 1
            t = torch.arange(K, dtype=torch.float32) - K // 2
            k = torch.exp(-0.5 * (t / sigma) ** 2)
            self.register_buffer('band_k%d' % rate, (k / k.sum()).view(1, 1, 1, 1, K))
        print('uprate: ' + str(self.uprate))
        print('num_scales: ' + str(self.num_scales))

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VQGAN")
        parser.add_argument("--skipl1", type=int, default=4)
        parser.add_argument("--tc", action="store_true", default=False)
        parser.add_argument("--l1how", type=str, default='dsp')
        parser.add_argument("--dsp", type=int, default=1, help='extra downsample rate')
        parser.add_argument("--usp", type=float, default=1.0, help='extra upsample rate')
        parser.add_argument("--downbranch", type=int, default=1)
        parser.add_argument("--resizebranch", type=int, default=1)
        parser.add_argument('--lbm_ms_ssim', type=float, default=0, help='weight for ms_ssim loss')
        # Multi-scale (progressive) output supervision
        parser.add_argument("--lamb_pyr", type=float, default=1.0,
                            help='weight for pyramid consistency loss on the coarse (1/2, 1/4) output heads')
        parser.add_argument("--pyr_detach", action='store_true', default=False,
                            help='detach the avg-pooled full-res target so pyramid consistency only trains the coarse heads (distillation)')
        parser.add_argument("--adv_ms", type=float, default=1.0,
                            help='multiplier on the 1/2 and 1/4 adversarial loss (on top of --adv); adv_ms=0 skips the coarse discriminators entirely')
        # VQ specific arguments
        parser.add_argument("--ldmyaml", type=str, default='vqgan')
        parser.add_argument("--use_ema", action='store_true',
                            help='(ignored — EMA is always on in this model)')
        parser.add_argument("--ema_decay", type=float, default=0.9999,
                            help='EMA decay (LitEma warms up from 0 via its num_updates schedule)')
        parser.add_argument("--lamb_coarse", type=float, default=1.0,
                            help='weight of the real-data coarse L1 (out128/out64 Z-projections vs '
                                 'XY-pooled input), multiplied on top of --lamb; 0 disables')
        parser.add_argument("--lse_tau", type=float, default=0.1,
                            help="temperature for --l1how lse (tau->0 recovers 'max')")
        parser.add_argument("--l1phase", type=str, default='fixed', choices=['fixed', 'random'],
                            help='randomize the projection window phase each step (mean/max/lse only)')
        parser.add_argument("--remap", type=str, default=None, help='remap indices')
        parser.add_argument("--sane_index_shape", action='store_true', help='return indices as bhw')
        parser.add_argument("--lr_g_factor", type=float, default=1.0, help='learning rate factor for generator')
        parser.add_argument("--num_scales", type=int, default=4,
                            help='number of VQ scales coarse-to-fine (this model needs >= 3: '
                                 'coarse trunk + two injected scales)')
        parser.add_argument("--shared_codebook", action='store_true',
                            help='share a single codebook across all scales (VAR paper setting)')
        return parent_parser

    def encode(self, x):
        """
        Residual multi-scale VQ encoding following VAR Algorithm 1.

        For each scale k (coarse to fine):
          1. Downsample residual f to hk × wk
          2. quant_convs[k]: z_channels → embed_dim at hk × wk
          3. quantizers[k]: quantize at hk × wk → quant_k, emb_loss_k, indices_k
          4. Upsample quant_k back to full H × W
          5. post_quant_convs[k]: embed_dim → z_channels at full H × W  (VAR ϕk)
          6. residual = residual − quant_k_up

        Final quant = sum of all quant_k_up  (z_channels space, H × W)
        """
        h, hbranch, hz = self.encoder(x)
        H, W = h.shape[-2], h.shape[-1]

        residual = h          # z_channels space, full H × W
        quants = []
        emb_losses = []
        indices = []

        for k in range(self.num_scales):
            scale = self.scale_factors[k]

            # Step 1: downsample residual to scale k resolution
            if scale > 1:
                h_k = F.interpolate(residual, size=(H // scale, W // scale),
                                    mode='bilinear', align_corners=False)
            else:
                h_k = residual

            # Step 2: project z_channels → embed_dim at scale resolution
            h_k = self.quant_convs[k](h_k)

            # Step 3: quantize at scale resolution
            quant_k, emb_loss_k, info_k = self.quantizers[k](h_k)
            emb_losses.append(emb_loss_k)
            indices.append(info_k[2])   # codebook indices — store these in Zarr

            # Step 4: upsample quantized embeddings back to full H × W
            if scale > 1:
                quant_k = F.interpolate(quant_k, size=(H, W),
                                        mode='bilinear', align_corners=False)

            # Step 5: post_quant_conv at full resolution (VAR ϕk)
            quant_k_up = self.post_quant_convs[k](quant_k)   # embed_dim → z_channels

            quants.append(quant_k_up)

            # Step 6: subtract contribution so next scale encodes the residual
            residual = residual - quant_k_up

        # Sum all scales → final quant in z_channels space, same shape as original h
        quant = sum(quants)
        # Normalize emb_loss to keep qloss magnitude stable regardless of num_scales
        emb_loss = sum(emb_losses) / self.num_scales

        # Also return the per-scale contributions: generation() feeds the coarse partial
        # sum to the net_g trunk and injects the finest scales at matching decoder depths.
        return quant, emb_loss, indices, h, quants

    def decode(self, quant):
        """
        Decode from combined multi-scale latent.
        quant is already in z_channels space (post_quant_convs applied per-scale in encode),
        so feed directly into the decoder — no post_quant_conv needed here.
        """
        dec = self.decoder(quant)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, indices, h, quants = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, indices, h, quants
        return dec, diff, h, quants, quant

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def adv_loss_six_way(self, x, net_d, truth):
        loss = 0
        loss += self.add_loss_adv(a=x.permute(2, 1, 4, 3, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 4, 2, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        loss = loss / 3
        return loss

    def get_xy_plane(self, x):
        return x.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]

    def generation(self, batch, deterministic=False):
        if self.hparams.cropz > 0 and self.training:
            if deterministic:
                z_init = 0
            else:
                z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            for b in range(len(batch['img'])):
                batch['img'][b] = batch['img'][b][:, :, :, :, z_init:z_init + self.hparams.cropz]

        # Keep all non-input views (data1, data2, ...) at full Z — captured before the
        # `dsp` subsample so xcube/ycube retain their real HR-Z. Used for GIF logging
        # (and available for projection supervision). The `dsp`/`usp` reassignments
        # below replace batch['img'][b] with new tensors, so these references stay full-Z.
        self.aux_views = [batch['img'][b] for b in range(1, len(batch['img']))]

        if self.hparams.dsp > 1:
            if deterministic:
                z_init = 0
            else:
                z_init = np.random.randint(self.hparams.dsp)
            for b in range(len(batch['img'])):
                batch['img'][b] = batch['img'][b][:, :, :, :, z_init::self.hparams.dsp]

        if self.hparams.usp != 1:
            for b in range(len(batch['img'])):
                batch['img'][b] = nn.Upsample(scale_factor=(1, 1, self.hparams.usp),
                                              mode='trilinear')(batch['img'][b])

        if self.hparams.tc:
            self.oriX = torch.cat((batch['img'][0], batch['img'][1]), 1)
        else:
            self.oriX = batch['img'][0]

        input_slice = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, Y, X)
        if self.training:
            input_slice = input_slice.requires_grad_(True)

        self.reconstructions, self.qloss, _, quants, _ = self.forward(
            input_slice, return_pred_indices=False
        )
        self.quants = quants   # per-scale contributions, kept for inspection/tests

        # Full sum into the trunk (every code gets full decoder depth, like vqcleanM0aMS);
        # the two finest scales are ADDITIONALLY injected after the out64/out128 taps.
        quant = sum(quants)

        # quant is in z_channels space — feed into decoder.conv_in for net_g
        if self.hparams.downbranch > 1:
            quant = quant.permute(1, 2, 3, 0).unsqueeze(0)
            quant = nn.MaxPool3d((1, 1, self.hparams.downbranch))(quant)
            quant = quant.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]

        if self.hparams.resizebranch != 1:
            quant = quant.permute(1, 2, 3, 0).unsqueeze(0)
            quant = nn.Upsample(scale_factor=(1, 1, self.hparams.resizebranch), mode='trilinear')(quant)
            quant = quant.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]

        quant = self.decoder.conv_in(quant)
        quant = quant.permute(1, 2, 3, 0).unsqueeze(0)

        # Project the injected scales at latent resolution; net_g resizes them spatially
        # at the injection points. Same Z-transforms as the trunk so the grids match.
        inject = {}
        for key, q, conv in (('pre_up2', quants[-2], self.inject_convs[0]),
                             ('pre_up1', quants[-1], self.inject_convs[1])):
            q = q.permute(1, 2, 3, 0).unsqueeze(0)   # (1, z_channels, X, Y, Z)
            if self.hparams.downbranch > 1:
                q = nn.MaxPool3d((1, 1, self.hparams.downbranch))(q)
            if self.hparams.resizebranch != 1:
                q = nn.Upsample(scale_factor=(1, 1, self.hparams.resizebranch), mode='trilinear')(q)
            inject[key] = conv(q)

        out = self.net_g(quant, method='decode', inject=inject)
        self.XupX = out['out0']         # full-res isotropic output
        self.XupX128 = out['out128']    # 1/2-res output head
        self.XupX64 = out['out64']      # 1/4-res output head
        # Extra GIF panels for MLflow logging (base.py upsamples these to the full grid
        # with nearest-neighbor so their coarser resolution stays visible).
        self.gif_scales = [self.XupX128, self.XupX64]
        # self.Xup = self.upsample(self.oriX)
        # Match Xup to the network output's actual size (not a fixed cropsize cube),
        # so validation without cropz (full z-depth) keeps Xup and XupX the same shape
        self.Xup = F.interpolate(self.oriX, size=self.XupX.shape[2:], mode='trilinear')

    def get_projection(self, x, depth, how='mean', uprate=None):
        # `uprate` only matters for how='dsp' (defaults to the full-res rate); the coarse
        # heads pass their own halved/quartered rate.
        if how == 'dsp':
            u = self.uprate if uprate is None else uprate
            x = x[:, :, :, :, (u // 2)::u * self.hparams.skipl1]
        else:
            x = x.unfold(-1, depth, depth)
            if how == 'mean':
                x = x.mean(dim=-1)
            elif how == 'max':
                x, _ = x.max(dim=-1)
            elif how == 'lse':
                # Soft-max projection: -log W removes the positive bias so tau->0 recovers
                # 'max' exactly and large tau approaches 'mean'.
                tau = self.hparams.lse_tau
                x = tau * (torch.logsumexp(x / tau, dim=-1) - np.log(x.shape[-1]))
        return x

    def band_l1(self, pred, obs, rate):
        # l1how='band': dense L1 restricted to the Z-frequency band the observation
        # actually measured — Gaussian-low-pass both sides along Z, compare everywhere.
        k = getattr(self, 'band_k%d' % rate)
        obs_up = F.interpolate(obs, size=pred.shape[2:], mode='trilinear')
        C = pred.shape[1]
        w = k.repeat(C, 1, 1, 1, 1)
        pad = (0, 0, k.shape[-1] // 2)

        def blur(t):
            return F.conv3d(t, w, groups=C, padding=pad)

        return self.add_loss_l1(blur(pred), blur(obs_up))

    def proj_l1(self, pred, obs, rate, r=0):
        # Window-based projection L1 (dsp/mean/max/lse); r is the shared random window
        # phase from --l1phase random (0 = the original fixed grid).
        skip = self.hparams.skipl1
        if r > 0:
            pred = pred[:, :, :, :, r * rate:]
            obs = obs[:, :, :, :, r:]
        a = self.get_projection(pred, depth=rate * skip, how=self.hparams.l1how, uprate=rate)
        b = obs[:, :, :, :, ::skip]
        n = min(a.shape[-1], b.shape[-1])
        return self.add_loss_l1(a[..., :n], b[..., :n])

    def backward_g(self):
        loss_g = 0
        loss_dict = {}

        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)

        # One window phase shared by the main and coarse projection losses.
        r_phase = 0
        if (self.hparams.l1phase == 'random' and self.training
                and self.hparams.l1how in ('mean', 'max', 'lse')):
            r_phase = np.random.randint(self.hparams.skipl1)

        if self.hparams.l1how == 'band':
            loss_l1 = self.band_l1(self.XupX, self.oriX, self.uprate)
        else:
            loss_l1 = self.proj_l1(self.XupX, self.oriX, self.uprate, r=r_phase)

        if self.hparams.lbm_ms_ssim > 0:
            loss_ms_ssim = 1 - ms_ssim(
                self.XupX.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],
                self.Xup.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],
                data_range=2.0, size_average=True, win_size=7,
                weights=[0.0448, 0.2856, 0.6696]
            )
            loss_dict['ms_ssim'] = loss_ms_ssim
            loss_g += loss_ms_ssim * self.hparams.lbm_ms_ssim

        loss_dict['axx'] = axx
        loss_g += axx * self.hparams.adv
        loss_dict['l1'] = loss_l1
        loss_g += loss_l1 * self.hparams.lamb

        # Real-data supervision of the coarse heads (same --l1how forward model as the
        # main l1, with halved/quartered rate) against the XY-avg-pooled input.
        if self.hparams.lamb_coarse > 0:
            ori128 = F.avg_pool3d(self.oriX, (2, 2, 1))
            ori64 = F.avg_pool3d(self.oriX, (4, 4, 1))
            if self.hparams.l1how == 'band':
                loss_l1_128 = self.band_l1(self.XupX128, ori128, self.uprate // 2)
                loss_l1_64 = self.band_l1(self.XupX64, ori64, self.uprate // 4)
            else:
                loss_l1_128 = self.proj_l1(self.XupX128, ori128, self.uprate // 2, r=r_phase)
                loss_l1_64 = self.proj_l1(self.XupX64, ori64, self.uprate // 4, r=r_phase)
            loss_dict['l1c'] = loss_l1_128 + loss_l1_64
            loss_g += (loss_l1_128 + loss_l1_64) * self.hparams.lamb * self.hparams.lamb_coarse

        # --- Multi-scale (progressive) output supervision ---
        # Pyramid consistency: coarse heads match avg-pooled full-res output.
        ref128 = F.avg_pool3d(self.XupX, 2)
        ref64 = F.avg_pool3d(self.XupX, 4)
        if self.hparams.pyr_detach:
            ref128 = ref128.detach()
            ref64 = ref64.detach()
        loss_pyr = self.add_loss_l1(self.XupX128, ref128) + self.add_loss_l1(self.XupX64, ref64)
        loss_dict['pyr'] = loss_pyr
        loss_g += loss_pyr * self.hparams.lamb_pyr

        # Per-scale six-way adversarial loss from the dedicated discriminators.
        # --adv_ms scales it on top of --adv; adv_ms=0 skips the coarse discriminators entirely.
        if self.hparams.adv_ms > 0:
            axx128 = self.adv_loss_six_way(self.XupX128, net_d=self.net_d_128, truth=True)
            axx64 = self.adv_loss_six_way(self.XupX64, net_d=self.net_d_64, truth=True)
            loss_dict['axx128'] = axx128
            loss_dict['axx64'] = axx64
            loss_g += (axx128 + axx64) * self.hparams.adv * self.hparams.adv_ms

        oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        if self.hparams.tc:
            oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :1, :, :, 0]

        aeloss, log_dict_ae = self.loss(
            self.qloss, oriXpermute, self.reconstructions,
            0, self.global_step,
            last_layer=self.get_last_layer(), split="train"
        )
        loss_g += aeloss
        loss_dict['ae'] = aeloss
        loss_dict['vq'] = self.qloss
        loss_g += self.qloss

        loss_dict['sum'] = loss_g
        return loss_dict

    def backward_d(self):
        loss_d = 0
        loss_dict = {}

        dxx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=False)
        dx = self.add_loss_adv(a=self.get_xy_plane(self.oriX), net_d=self.net_d, truth=True)

        loss_dict['dxx_x'] = dxx + dx
        loss_d += (dxx + dx) * self.hparams.adv

        # Per-scale discriminator updates: fakes from the coarse heads, real targets are the
        # real X/Y slices downsampled to the matching resolution. Skipped when --adv_ms=0.
        if self.hparams.adv_ms > 0:
            real_xy = self.get_xy_plane(self.oriX)
            d128 = self.adv_loss_six_way(self.XupX128, net_d=self.net_d_128, truth=False) \
                + self.add_loss_adv(a=F.avg_pool2d(real_xy, 2), net_d=self.net_d_128, truth=True)
            d64 = self.adv_loss_six_way(self.XupX64, net_d=self.net_d_64, truth=False) \
                + self.add_loss_adv(a=F.avg_pool2d(real_xy, 4), net_d=self.net_d_64, truth=True)
            loss_dict['d128'] = d128
            loss_dict['d64'] = d64
            loss_d += (d128 + d64) * self.hparams.adv * self.hparams.adv_ms

        oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        if self.hparams.tc:
            oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :1, :, :, 0]

        discloss, log_dict_disc = self.loss(
            self.qloss, oriXpermute, self.reconstructions,
            1, self.global_step,
            last_layer=self.get_last_layer(), split="train"
        )
        loss_d += discloss
        loss_dict['disc'] = discloss
        loss_dict['sum'] = loss_d
        return loss_dict

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    # MSskipE (F1): run the whole validation loop (metrics, KID, GIFs) under EMA weights.
    def on_validation_start(self):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)

    def on_validation_end(self):
        if self.use_ema:
            self.model_ema.restore(self.parameters())

    def training_epoch_end(self, outputs):
        # The epoch checkpoints (saved by BaseModel.training_epoch_end via netg_names) should
        # hold EMA weights too — swap them in around the base implementation.
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            try:
                super().training_epoch_end(outputs)
            finally:
                self.model_ema.restore(self.parameters())
        else:
            super().training_epoch_end(outputs)

    def configure_optimizers(self):
        lr_d = self.hparams.lr
        lr_g = getattr(self.hparams, 'lr_g_factor', 1.0) * self.hparams.lr
        print("lr_d", lr_d)
        print("lr_g", lr_g)

        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantizers.parameters()) +
            list(self.quant_convs.parameters()) +
            list(self.post_quant_convs.parameters()) +
            list(self.inject_convs.parameters()) +
            list(self.net_g.parameters()),
            lr=lr_g, betas=(0.5, 0.9)
        )

        opt_disc = torch.optim.Adam(
            list(self.loss.discriminator.parameters()) +
            list(self.net_d.parameters()) +
            list(self.net_d_128.parameters()) +
            list(self.net_d_64.parameters()),
            lr=lr_d, betas=(0.5, 0.9)
        )

        if hasattr(self.hparams, 'scheduler_config') and self.hparams.scheduler_config is not None:
            scheduler = instantiate_from_config(self.hparams.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                 'interval': 'step', 'frequency': 1},
                {'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                 'interval': 'step', 'frequency': 1},
            ]
            self.net_g_scheduler = scheduler[0]['scheduler']
            self.net_d_scheduler = scheduler[1]['scheduler']
            return [opt_ae, opt_disc], scheduler

        self.net_g_scheduler = get_scheduler(opt_ae, self.hparams)
        self.net_d_scheduler = get_scheduler(opt_disc, self.hparams)
        return [opt_ae, opt_disc], []

    def save_checkpoint(self, filepath):
        state_dict = {}

        for k, v in self.encoder.state_dict().items():
            state_dict[f'encoder.{k}'] = v
        for k, v in self.decoder.state_dict().items():
            state_dict[f'decoder.{k}'] = v
        for k, v in self.quantizers.state_dict().items():
            state_dict[f'quantizers.{k}'] = v
        for k, v in self.quant_convs.state_dict().items():
            state_dict[f'quant_convs.{k}'] = v
        for k, v in self.post_quant_convs.state_dict().items():
            state_dict[f'post_quant_convs.{k}'] = v
        for k, v in self.inject_convs.state_dict().items():
            state_dict[f'inject_convs.{k}'] = v
        for k, v in self.discriminator.state_dict().items():
            state_dict[f'loss.discriminator.{k}'] = v

        if self.use_ema:
            for k, v in self.model_ema.state_dict().items():
                state_dict[f'model_ema.{k}'] = v

        checkpoint = {
            "state_dict": state_dict,
            "global_step": self.global_step,
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }

        if hasattr(self, 'hparams'):
            checkpoint['hparams'] = self.hparams

        torch.save(checkpoint, filepath)
        print(f"VQGAN model saved to {filepath}")
