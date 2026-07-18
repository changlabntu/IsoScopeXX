from models.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from ldm.modules.diffusionmodules.modelcut import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
import yaml
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from networks.networks import get_scheduler
import os
from pytorch_msssim import ms_ssim
import tifffile as tiff


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        # VQGAN Model
        print('Reading yaml: ' + self.hparams.ldmyaml)
        with open('ldm/' + self.hparams.ldmyaml + '.yaml', "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        ddconfig = config['model']['params']["ddconfig"]

        if self.hparams.tc:
            ddconfig['in_channels'] = 2
            ddconfig['out_ch'] = 1
        self.hparams.netG = self.hparams.netG

        # Bottleneck mode: VQ (default) or VAE (KL)
        self.use_vae = getattr(hparams, 'vae', False)
        if self.use_vae:
            ddconfig['double_z'] = True  # encoder outputs 2*z_channels for mu+logvar

        self.hparams.final = 'tanh'
        self.net_g, self.net_d = self.set_networks()

        # VQGAN/VAE components
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.embed_dim = config['model']['params']['embed_dim']

        if self.use_vae:
            # VAE: quant_conv maps 2*z_channels -> 2*embed_dim (mu + logvar)
            self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * self.embed_dim, 1)
            self.post_quant_conv = nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)
        else:
            self.n_embed = config['model']['params']['n_embed']
            # Vector Quantizer
            self.quantize = VectorQuantizer(
                self.n_embed,
                self.embed_dim,
                beta=0.25,
                remap=getattr(hparams, 'remap', None),
                sane_index_shape=getattr(hparams, 'sane_index_shape', False)
            )
            # Quantization convolutions
            self.quant_conv = nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
            self.post_quant_conv = nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)

        # Initialize loss
        self.loss = instantiate_from_config(config['model']['params']["lossconfig"])
        self.discriminator = self.loss.discriminator

        # EMA support
        self.use_ema = getattr(hparams, 'use_ema', False)
        if self.use_ema:
            try:
                from taming.modules.util import LitEma
                self.model_ema = LitEma(self)
                print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
            except ImportError:
                print("LitEma not available, disabling EMA")
                self.use_ema = False

        # Save model names
        self.netg_names = {
            'encoder': 'encoder',
            'decoder': 'decoder',
            'quant_conv': 'quant_conv',
            'post_quant_conv': 'post_quant_conv',
            'net_g': 'net_g'
        }
        if not self.use_vae:
            self.netg_names['quantize'] = 'quantize'
        self.netd_names = {'discriminator': 'discriminator', 'net_d': 'net_d'}

        # Configure optimizers
        self.configure_optimizers()

        self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropsize), mode='trilinear')
        self.uprate = (hparams.cropsize // hparams.cropz) * hparams.dsp / hparams.usp
        self.uprate = int(self.uprate)
        print('uprate: ' + str(self.uprate))

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
        # VQ specific arguments
        parser.add_argument("--ldmyaml", type=str, default='vqgan')
        parser.add_argument("--use_ema", action='store_true', help='use exponential moving average')
        parser.add_argument("--remap", type=str, default=None, help='remap indices')
        parser.add_argument("--sane_index_shape", action='store_true', help='return indices as bhw')
        parser.add_argument("--lr_g_factor", type=float, default=1.0, help='learning rate factor for generator')
        parser.add_argument("--vae", action='store_true', help='use VAE (KL) instead of VQ bottleneck')
        return parent_parser

    def encode(self, x):
        """Encode input to latent space (VQ or VAE bottleneck)"""
        h, hbranch, hz = self.encoder(x)
        if self.use_vae:
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return posterior, None, None, h, None
        else:
            h = self.quant_conv(h)
            quant, emb_loss, info = self.quantize(h)
            return quant, emb_loss, info, h, None

    def decode(self, quant):
        """Decode from latent space"""
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input, return_pred_indices=False, sample_posterior=True):
        """Forward pass"""
        if self.use_vae:
            posterior, _, _, h, _ = self.encode(input)
            z = posterior.sample() if sample_posterior else posterior.mode()
            dec = self.decode(z)
            return dec, posterior, h, None, z
        else:
            quant, diff, (_, _, ind), h, _ = self.encode(input)
            dec = self.decode(quant)
            if return_pred_indices:
                return dec, diff, ind, h, None
            return dec, diff, h, None, quant

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def adv_loss_six_way(self, x, net_d, truth):
        loss = 0
        loss += self.add_loss_adv(a=x.permute(2, 1, 4, 3, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        #loss += self.add_loss_adv(a=x.permute(2, 1, 3, 4, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 4, 2, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        #loss += self.add_loss_adv(a=x.permute(3, 1, 2, 4, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        #loss += self.add_loss_adv(a=x.permute(4, 1, 3, 2, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        loss = loss / 3
        return loss

    def get_xy_plane(self, x):
        return x.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]

    # Volumes are (B, C, Y, X, Z) — the repo-wide layout (dim4 = Z, the
    # anisotropic axis). Same helpers as models/MSclean.py: the 2D VQ stack
    # runs on XY-slice batches with Z folded into the batch dimension. Used by
    # the stateless inference path only; generation() keeps its historical
    # batch-1 permutes.

    @staticmethod
    def vol_to_slices(v):
        """(B, C, Y, X, Z) volume -> (B*Z, C, Y, X) XY-slice batch."""
        B, C, Y, X, Z = v.shape
        return v.permute(0, 4, 1, 2, 3).reshape(B * Z, C, Y, X)

    @staticmethod
    def slices_to_vol(s, batch_size=1):
        """(B*Z, C, H, W) slice batch -> (B, C, H, W, Z) volume."""
        BZ, C, H, W = s.shape
        return s.reshape(batch_size, BZ // batch_size, C, H, W).permute(0, 2, 3, 4, 1)

    def _z_transform(self, v):
        """Optional latent-Z reshaping (--downbranch / --resizebranch) on a
        (B, C, H, W, Z) volume before it enters net_g."""
        if self.hparams.downbranch > 1:
            v = nn.MaxPool3d((1, 1, self.hparams.downbranch))(v)
        if self.hparams.resizebranch != 1:
            v = nn.Upsample(scale_factor=(1, 1, self.hparams.resizebranch),
                            mode='trilinear')(v)
        return v

    def _netg_decode_single(self, latent_slices, batch_size):
        """Batched twin of generation() lines 233-246 (bit-identical for B=1).

        In:  latent_slices — (B*Z, embed_dim, H, W) post-quantize latent. Feeding
             it to decoder.conv_in works because embed_dim == z_channels in the
             ldm yaml (4 == 4 for vqgan.yaml) — the same coupling generation()
             relies on.
        Out: (B, 1, Y', X', Z') isotropic volume, net_g's 'out0'.
        """
        v = self._z_transform(self.slices_to_vol(latent_slices, batch_size))
        trunk = self.slices_to_vol(self.decoder.conv_in(self.vol_to_slices(v)),
                                   batch_size)
        return self.net_g(trunk, method='decode')['out0']

    def generation_test(self, x, method='full'):
        """Stateless inference API — same staged contract as models/MSclean.py,
        with single-scale length-1 lists so callers (inference.Engine) never
        branch on model type.

        method='full' (default): x = (B, C, Y, X, Z) normalized volume
            -> the full-res isotropic volume (what generation() assigns to
            self.XupX, i.e. net_g's 'out0'). Equals 'encode' then 'decode'.
        method='encode': x = (B, C, Y, X, Z) volume -> (scale_latents, indices):
            scale_latents — [ (B, embed_dim, H, W, Z) ], the post-quantize
                latent (pre post_quant_conv — what BOTH downstream paths eat);
            indices — [ (B, Z, H, W) int64 codebook indices ] (Zarr-storable;
                rebuild the latent exactly with latents_from_indices), or None
                under --vae (no codebook; latent is posterior.mode()).
        method='decode': x = scale_latents list -> the 3D isotropic volume
            (B, 1, Y', X', Z') via decoder.conv_in + net_g.
        method='reconstruction': x = scale_latents list -> the slice-wise 2D
            VQ-head reconstruction (post_quant_conv + decoder) as a
            (B, out_ch, Y, X, Z) volume — no net_g, no Z upsampling (what
            training keeps as self.reconstructions).

        Contract (all methods): inputs already normalized and at the model's
        expected Z resolution (no dsp/usp/cropz prep); stateless; batched (Z
        folded with B into the 2D batches); device/dtype are the caller's job;
        wrap calls in torch.no_grad()/inference_mode().

        Note --vae 'full' uses the deterministic posterior.mode(), unlike
        generation()'s posterior.sample() — only the VQ branch bit-matches
        generation().
        """
        assert not self.training, \
            'generation_test is eval-only (train mode would update BatchNorm running stats)'

        if method in ('full', 'encode'):
            B = x.shape[0]
            slices = self.vol_to_slices(x)
            if self.use_vae:
                posterior, _, _, _, _ = self.encode(slices)
                latent_slices = posterior.mode()
                raw_indices = None
            else:
                quant, _, info, _, _ = self.encode(slices)
                latent_slices = quant
                raw_indices = info[2]
            if method == 'full':
                return self._netg_decode_single(latent_slices, batch_size=B)
            H, W = latent_slices.shape[-2:]
            Z = latent_slices.shape[0] // B
            scale_latents = [self.slices_to_vol(latent_slices, B)]
            # reshape works for every quantizer index layout (flat, (N,1) with
            # remap, or (N, H, W) with --sane_index_shape)
            indices = None if raw_indices is None else [raw_indices.reshape(B, Z, H, W)]
            return scale_latents, indices

        if method == 'decode':
            B = x[0].shape[0]
            return self._netg_decode_single(self.vol_to_slices(x[0]), batch_size=B)

        if method == 'reconstruction':
            B = x[0].shape[0]
            recon = self.decode(self.vol_to_slices(x[0]))
            return self.slices_to_vol(recon, B)

        raise ValueError(f"generation_test: unknown method '{method}'")

    def latents_from_indices(self, indices):
        """Rebuild generation_test(method='encode')'s scale_latents from stored
        codebook indices — the decode side of the compression path (no encoder
        run needed).

        In:  indices — [ (B, Z, H, W) int64 ] (length-1 list).
        Out: scale_latents — [ (B, embed_dim, H, W, Z) ]. Unlike MSclean there
             is no post_quant_conv here: vqclean's stored latent is the raw
             quantized code (post_quant_conv is applied inside decode() for
             reconstruction, decoder.conv_in for the net_g path). Differs from
             the encode-path latent only by straight-through round-off (~1e-7:
             encode carries z + (z_q - z).detach(), this path the exact z_q).
        """
        assert not self.training, 'latents_from_indices is eval-only'
        if self.use_vae:
            raise RuntimeError('latents_from_indices needs a VQ codebook; this run is --vae')
        idx = indices[0]
        B, Z, H, W = idx.shape
        code = self.quantize.get_codebook_entry(
            idx.reshape(-1), shape=(B * Z, H, W, self.embed_dim))
        return [self.slices_to_vol(code, B)]

    def generation(self, batch, deterministic=False):
        # cropz is a training-time crop only (gated on self.training, as in the MS
        # family) — previously unconditional, which silently truncated full-depth
        # validation/inference volumes (test/inference.py worked around it by zeroing
        # hparams.cropz after loading; that workaround remains compatible).
        if self.hparams.cropz > 0 and self.training:
            if deterministic:
                z_init = 0
            else:
                z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            for b in range(len(batch['img'])):
                batch['img'][b] = batch['img'][b][:, :, :, :, z_init:z_init + self.hparams.cropz]

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

        result = self.forward(input_slice, return_pred_indices=False)
        self.reconstructions = result[0]
        if self.use_vae:
            self.posterior = result[1]
        else:
            self.qloss = result[1]
        quant = result[4]

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

        self.XupX = self.net_g(quant, method='decode')['out0']
        # Match Xup to the output's actual size (not a fixed cropsize cube) so full
        # z-depth validation keeps Xup and XupX the same shape — same fix as the MS
        # family; identical to self.upsample(...) during training (crops = cropsize^3).
        self.Xup = F.interpolate(self.oriX, size=self.XupX.shape[2:], mode='trilinear')

    def get_projection(self, x, depth, how='mean'):
        if how == 'dsp':
            x = x[:, :, :, :, (self.uprate // 2)::self.uprate * self.hparams.skipl1]
        else:
            x = x.unfold(-1, depth, depth)
            if how == 'mean':
                x = x.mean(dim=-1)
            elif how == 'max':
                x, _ = x.max(dim=-1)
        return x

    def backward_g(self):
        loss_g = 0
        loss_dict = {}

        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)

        loss_l1 = self.add_loss_l1(
            a=self.get_projection(self.XupX, depth=self.uprate * self.hparams.skipl1,
                                  how=self.hparams.l1how),
            b=self.oriX[:, :, :, :, ::self.hparams.skipl1]
        )

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

        oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        if self.hparams.tc:
            oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :1, :, :, 0]

        if self.use_vae:
            aeloss, log_dict_ae = self.loss(
                oriXpermute, self.reconstructions, self.posterior,
                0, self.global_step,
                last_layer=self.get_last_layer(), split="train"
            )
            loss_g += aeloss
            loss_dict['ae'] = aeloss
        else:
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

        oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        if self.hparams.tc:
            oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :1, :, :, 0]

        if self.use_vae:
            discloss, log_dict_disc = self.loss(
                oriXpermute, self.reconstructions, self.posterior,
                1, self.global_step,
                last_layer=self.get_last_layer(), split="train"
            )
        else:
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

    def configure_optimizers(self):
        lr_d = self.hparams.lr
        lr_g = getattr(self.hparams, 'lr_g_factor', 1.0) * self.hparams.lr
        print("lr_d", lr_d)
        print("lr_g", lr_g)

        ae_params = (
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()) +
            list(self.net_g.parameters())
        )
        if not self.use_vae:
            ae_params += list(self.quantize.parameters())

        opt_ae = torch.optim.Adam(ae_params, lr=lr_g, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(
            list(self.loss.discriminator.parameters()) + list(self.net_d.parameters()),
            lr=lr_d, betas=(0.5, 0.9)
        )

        if hasattr(self.hparams, 'scheduler_config') and self.hparams.scheduler_config is not None:
            scheduler = instantiate_from_config(self.hparams.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1},
                {'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1},
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
        if not self.use_vae:
            for k, v in self.quantize.state_dict().items():
                state_dict[f'quantize.{k}'] = v
        for k, v in self.quant_conv.state_dict().items():
            state_dict[f'quant_conv.{k}'] = v
        for k, v in self.post_quant_conv.state_dict().items():
            state_dict[f'post_quant_conv.{k}'] = v
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
