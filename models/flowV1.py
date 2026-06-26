"""Plane-Aware Conditional Flow Refinement for Latent Isotropic Enhancement.

Trains a 2D rectified flow on XY latent slices (the clean distribution),
then refines XZ/YZ slices toward that distribution via ODE integration.
Tri-plane fusion feeds the corrected latent into net_g for 3D decoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from contextlib import contextmanager

from models.base import BaseModel
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


# ====================================================================
# 2D Flow Matching Components
# ====================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLMResBlock2D(nn.Module):
    """2D residual block with FiLM time conditioning."""
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(time_dim, 2 * channels)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        film_params = self.film(t_emb)
        while film_params.dim() < h.dim():
            film_params = film_params.unsqueeze(-1)
        scale, shift = film_params.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class VelocityNet2D(nn.Module):
    """2D velocity network for rectified flow matching.

    Input:  concat(z_t, condition) in R^(B, 2C, H, W)
    Output: velocity v in R^(B, C, H, W)
    """
    def __init__(self, in_channels, hidden_channels=128, num_blocks=4, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.conv_in = nn.Conv2d(2 * in_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList([
            FiLMResBlock2D(hidden_channels, time_dim) for _ in range(num_blocks)
        ])
        self.norm_out = nn.GroupNorm(32, hidden_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
        # Zero-init so flow starts as identity
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, z_t, t, condition):
        t_emb = self.time_mlp(t)
        h = self.conv_in(torch.cat([z_t, condition], dim=1))
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.act_out(self.norm_out(h))
        return self.conv_out(h)


# ====================================================================
# GAN Model
# ====================================================================

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
            ddconfig['double_z'] = True

        self.hparams.final = 'tanh'
        self.net_g, self.net_d = self.set_networks()

        # VQGAN/VAE components
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.embed_dim = config['model']['params']['embed_dim']

        if self.use_vae:
            self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * self.embed_dim, 1)
            self.post_quant_conv = nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)
        else:
            self.n_embed = config['model']['params']['n_embed']
            self.quantize = VectorQuantizer(
                self.n_embed, self.embed_dim, beta=0.25,
                remap=getattr(hparams, 'remap', None),
                sane_index_shape=getattr(hparams, 'sane_index_shape', False)
            )
            self.quant_conv = nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
            self.post_quant_conv = nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)

        # Initialize loss
        self.loss = instantiate_from_config(config['model']['params']["lossconfig"])
        self.discriminator = self.loss.discriminator

        # Flow matching: 2D velocity network
        # Operates on post-conv_in latent (ch * ch_mult[-1] channels)
        self.latent_ch = ddconfig['ch'] * ddconfig['ch_mult'][-1]
        self.velocity_net = VelocityNet2D(
            in_channels=self.latent_ch,
            hidden_channels=getattr(hparams, 'flow_hidden_ch', 128),
            num_blocks=getattr(hparams, 'flow_num_blocks', 4),
        )
        self.flow_grad_ckpt = True

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

        # Save model names (used by base class for checkpointing / param counting)
        self.netg_names = {
            'encoder': 'encoder',
            'decoder': 'decoder',
            'quant_conv': 'quant_conv',
            'post_quant_conv': 'post_quant_conv',
            'net_g': 'net_g',
            'velocity_net': 'velocity_net',
        }
        if not self.use_vae:
            self.netg_names['quantize'] = 'quantize'
        self.netd_names = {'discriminator': 'discriminator', 'net_d': 'net_d'}

        # Configure optimizers
        self.configure_optimizers()

        self.upsample = torch.nn.Upsample(
            size=(hparams.cropsize, hparams.cropsize, hparams.cropsize), mode='trilinear')
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
        parser = parent_parser.add_argument_group("FlowV1")
        # Inherited from vqclean
        parser.add_argument("--skipl1", type=int, default=4)
        parser.add_argument("--tc", action="store_true", default=False)
        parser.add_argument("--l1how", type=str, default='dsp')
        parser.add_argument("--dsp", type=int, default=1, help='extra downsample rate')
        parser.add_argument("--usp", type=float, default=1.0, help='extra upsample rate')
        parser.add_argument("--downbranch", type=int, default=1)
        parser.add_argument("--resizebranch", type=int, default=1)
        parser.add_argument('--lbm_ms_ssim', type=float, default=0, help='weight for ms_ssim loss')
        parser.add_argument("--ldmyaml", type=str, default='vqgan')
        parser.add_argument("--use_ema", action='store_true', help='use exponential moving average')
        parser.add_argument("--remap", type=str, default=None)
        parser.add_argument("--sane_index_shape", action='store_true')
        parser.add_argument("--lr_g_factor", type=float, default=1.0)
        parser.add_argument("--vae", action='store_true', help='use VAE (KL) instead of VQ bottleneck')
        # Flow-specific
        parser.add_argument("--flow_steps_train", type=int, default=4,
                            help='Euler steps during training (backprop through all)')
        parser.add_argument("--flow_steps", type=int, default=20,
                            help='Euler steps during inference')
        parser.add_argument("--flow_lambda", type=float, default=1.0,
                            help='weight for flow prior MSE loss')
        parser.add_argument("--flow_hidden_ch", type=int, default=128,
                            help='hidden channels in VelocityNet2D')
        parser.add_argument("--flow_num_blocks", type=int, default=4,
                            help='FiLM ResBlocks in VelocityNet2D')
        parser.add_argument("--flow_xy_weight", type=float, default=0.5,
                            help='XY weight in tri-plane fusion (XZ/YZ split remainder)')
        return parent_parser

    # ====================================================================
    # VQGAN encode / decode
    # ====================================================================

    def encode(self, x):
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
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input, return_pred_indices=False, sample_posterior=True):
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

    # ====================================================================
    # Flow matching helpers
    # ====================================================================

    def _flow_velocity_step(self, z_t, t_scalar, condition):
        """Single velocity prediction, optionally with gradient checkpointing."""
        t_tensor = torch.full((z_t.shape[0],), t_scalar,
                              device=z_t.device, dtype=z_t.dtype)
        if self.training and self.flow_grad_ckpt:
            return grad_checkpoint(
                self.velocity_net, z_t, t_tensor, condition,
                use_reentrant=False)
        return self.velocity_net(z_t, t_tensor, condition)

    def _flow_refine_slices(self, slices, num_steps):
        """Integrate rectified flow ODE from noise (t=1) to data (t=0).

        Euler method: z_{k+1} = z_k - v(z_k, t_k) * dt

        Args:
            slices: (S, C, H, W) degraded slices used as condition
            num_steps: number of Euler steps

        Returns:
            (S, C, H, W) refined slices at t=0
        """
        z_t = torch.randn_like(slices)
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t_val = 1.0 - k * dt                       # t: 1.0 → dt
            v = self._flow_velocity_step(z_t, t_val, condition=slices)
            z_t = z_t - v * dt
        return z_t

    def _compute_flow_prior_loss(self, xy_slices):
        """Rectified flow prior MSE loss on XY slices (clean distribution).

        Trains the velocity net to generate XY-quality slices from noise,
        conditioned on the input slice.  Detached so only velocity_net
        receives gradients from this loss.
        """
        x_0 = xy_slices.detach()                        # (Z, C, H, W)
        noise = torch.randn_like(x_0)
        t = torch.rand(x_0.shape[0], device=x_0.device, dtype=x_0.dtype)

        t_expand = t[:, None, None, None]
        z_t = (1 - t_expand) * x_0 + t_expand * noise   # interpolate

        v_pred = self.velocity_net(z_t, t, condition=x_0)
        v_target = noise - x_0                            # rectified flow target
        return F.mse_loss(v_pred, v_target)

    # ====================================================================
    # Loss helpers
    # ====================================================================

    def adv_loss_six_way(self, x, net_d, truth):
        loss = 0
        loss += self.add_loss_adv(a=x.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],
                                  net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 4, 2, 0)[:, :, :, :, 0],
                                  net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0],
                                  net_d=net_d, truth=truth)
        loss = loss / 3
        return loss

    def get_xy_plane(self, x):
        return x.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]

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

    # ====================================================================
    # Generation (forward pass through full pipeline)
    # ====================================================================

    def generation(self, batch, deterministic=False):
        # ---- batch prep (same as vqclean) ----
        if self.hparams.cropz > 0:
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

        # ---- 1. Encode 2D slices → VQ/VAE → quant ----
        input_slice = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)
        if self.training:
            input_slice = input_slice.requires_grad_(True)

        result = self.forward(input_slice, return_pred_indices=False)
        self.reconstructions = result[0]
        if self.use_vae:
            self.posterior = result[1]
        else:
            self.qloss = result[1]
        quant = result[4]

        # ---- 2. downbranch / resizebranch ----
        if self.hparams.downbranch > 1:
            quant = quant.permute(1, 2, 3, 0).unsqueeze(0)
            quant = nn.MaxPool3d((1, 1, self.hparams.downbranch))(quant)
            quant = quant.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]

        if self.hparams.resizebranch != 1:
            quant = quant.permute(1, 2, 3, 0).unsqueeze(0)
            quant = nn.Upsample(scale_factor=(1, 1, self.hparams.resizebranch),
                                mode='trilinear')(quant)
            quant = quant.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]

        # ---- 3. decoder.conv_in → stack to 3D latent ----
        quant = self.decoder.conv_in(quant)                 # (Z', latent_ch, H, W)
        c = quant.permute(1, 2, 3, 0).unsqueeze(0)         # (1, C, X, Y, Z)

        _, C, X, Y, Z = c.shape
        assert X == Y == Z, (
            f"Tri-plane flow requires cubic latent, got ({X}, {Y}, {Z}). "
            f"Adjust cropz / dsp / resizebranch so latent Z equals spatial X=Y={X}.")

        # ---- 4. Tri-plane flow refinement ----
        # Extract orthogonal slices
        xy_slices = c[0].permute(3, 0, 1, 2)               # (Z, C, X, Y)
        xz_slices = c[0].permute(2, 0, 1, 3)               # (Y, C, X, Z)
        yz_slices = c[0].permute(1, 0, 2, 3)               # (X, C, Y, Z)

        if self.training:
            # Flow prior: teach velocity net the XY distribution
            self.flow_prior_loss = self._compute_flow_prior_loss(xy_slices)
            # Refine XZ/YZ via flow ODE (gradients flow through for end-to-end training)
            xz_refined = self._flow_refine_slices(xz_slices, self.hparams.flow_steps_train)
            yz_refined = self._flow_refine_slices(yz_slices, self.hparams.flow_steps_train)
        else:
            self.flow_prior_loss = torch.tensor(0.0, device=c.device)
            with torch.no_grad():
                xz_refined = self._flow_refine_slices(xz_slices, self.hparams.flow_steps)
                yz_refined = self._flow_refine_slices(yz_slices, self.hparams.flow_steps)

        # Reassemble refined planes back to (1, C, X, Y, Z)
        vol_xy = c                                           # XY kept as-is (stable anchor)
        vol_xz = xz_refined.permute(1, 2, 0, 3).unsqueeze(0)
        vol_yz = yz_refined.permute(1, 0, 2, 3).unsqueeze(0)

        # Weighted tri-plane fusion
        w_xy = self.hparams.flow_xy_weight                   # default 0.5
        w_other = (1.0 - w_xy) / 2.0                        # default 0.25 each
        c_fused = w_xy * vol_xy + w_other * vol_xz + w_other * vol_yz

        # ---- 5. Decode fused latent ----
        self.XupX = self.net_g(c_fused, method='decode')['out0']
        self.Xup = self.upsample(self.oriX)

    # ====================================================================
    # Backward passes
    # ====================================================================

    def backward_g(self):
        loss_g = 0
        loss_dict = {}

        # 3D adversarial (six-way PatchGAN)
        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)

        # L1 projection loss
        loss_l1 = self.add_loss_l1(
            a=self.get_projection(self.XupX, depth=self.uprate * self.hparams.skipl1,
                                  how=self.hparams.l1how),
            b=self.oriX[:, :, :, :, ::self.hparams.skipl1])

        # Optional MS-SSIM
        if self.hparams.lbm_ms_ssim > 0:
            loss_ms_ssim = 1 - ms_ssim(
                self.XupX.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],
                self.Xup.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],
                data_range=2.0, size_average=True, win_size=7,
                weights=[0.0448, 0.2856, 0.6696])
            loss_dict['ms_ssim'] = loss_ms_ssim
            loss_g += loss_ms_ssim * self.hparams.lbm_ms_ssim

        loss_dict['axx'] = axx
        loss_g += axx * self.hparams.adv
        loss_dict['l1'] = loss_l1
        loss_g += loss_l1 * self.hparams.lamb

        # Flow prior loss (only velocity_net gets gradients from this)
        loss_dict['flow'] = self.flow_prior_loss
        loss_g += self.flow_prior_loss * self.hparams.flow_lambda

        # VQ/VAE reconstruction loss
        oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]
        if self.hparams.tc:
            oriXpermute = self.oriX.permute(4, 1, 2, 3, 0)[:, :1, :, :, 0]

        if self.use_vae:
            aeloss, log_dict_ae = self.loss(
                oriXpermute, self.reconstructions, self.posterior,
                0, self.global_step,
                last_layer=self.get_last_layer(), split="train")
            loss_g += aeloss
            loss_dict['ae'] = aeloss
        else:
            aeloss, log_dict_ae = self.loss(
                self.qloss, oriXpermute, self.reconstructions,
                0, self.global_step,
                last_layer=self.get_last_layer(), split="train")
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
                last_layer=self.get_last_layer(), split="train")
        else:
            discloss, log_dict_disc = self.loss(
                self.qloss, oriXpermute, self.reconstructions,
                1, self.global_step,
                last_layer=self.get_last_layer(), split="train")
        loss_d += discloss
        loss_dict['disc'] = discloss
        loss_dict['sum'] = loss_d
        return loss_dict

    # ====================================================================
    # Utilities
    # ====================================================================

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
            list(self.net_g.parameters()) +
            list(self.velocity_net.parameters())
        )
        if not self.use_vae:
            ae_params += list(self.quantize.parameters())

        opt_ae = torch.optim.Adam(ae_params, lr=lr_g, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(
            list(self.loss.discriminator.parameters()) + list(self.net_d.parameters()),
            lr=lr_d, betas=(0.5, 0.9))

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
        if not self.use_vae:
            for k, v in self.quantize.state_dict().items():
                state_dict[f'quantize.{k}'] = v
        for k, v in self.quant_conv.state_dict().items():
            state_dict[f'quant_conv.{k}'] = v
        for k, v in self.post_quant_conv.state_dict().items():
            state_dict[f'post_quant_conv.{k}'] = v
        for k, v in self.velocity_net.state_dict().items():
            state_dict[f'velocity_net.{k}'] = v
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
        print(f"FlowV1 model saved to {filepath}")

