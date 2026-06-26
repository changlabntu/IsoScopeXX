from models.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from torch.utils.checkpoint import checkpoint as grad_checkpoint

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
import math


# ============================================================================
# Velocity Network for Rectified Flow in 3D Latent Space
# ============================================================================

class FiLMResBlock3D(nn.Module):
    """3D residual block with FiLM (Feature-wise Linear Modulation) time conditioning.

    Time embedding is injected after the first normalization via learned
    scale and shift parameters, following standard practice in diffusion
    and flow matching architectures.
    """

    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.film = nn.Linear(time_dim, channels * 2)

        # Zero-init FiLM so block starts as identity
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # FiLM: inject time via learned affine transform
        film_params = self.film(F.silu(time_emb))       # (B, 2*C)
        scale, shift = film_params.chunk(2, dim=-1)      # each (B, C)
        scale = scale[:, :, None, None, None]             # (B, C, 1, 1, 1)
        shift = shift[:, :, None, None, None]

        h = self.norm2(h) * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)

        return x + h


class VelocityNet3D(nn.Module):
    """Lightweight 3D velocity network for rectified flow.

    Architecture:
        - Input projection: Conv3d( concat[z_t, condition] ) -> hidden
        - N FiLM-conditioned residual blocks (time-aware)
        - Output projection: Conv3d -> velocity (same shape as z_t)

    The output layer is zero-initialized so the network starts by
    predicting zero velocity, providing a stable training initialization.
    """

    def __init__(self, data_channels, time_dim=256, hidden_channels=128, num_blocks=4):
        super().__init__()

        # Time embedding MLP
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )

        # Input: concat(z_t, condition) along channel dim -> 2 * data_channels
        self.input_conv = nn.Conv3d(data_channels * 2, hidden_channels, 3, padding=1)

        # Residual blocks with FiLM time conditioning
        self.blocks = nn.ModuleList([
            FiLMResBlock3D(hidden_channels, hidden_channels)
            for _ in range(num_blocks)
        ])

        # Output: project back to data_channels (velocity has same shape as z_t)
        self.output_norm = nn.GroupNorm(8, hidden_channels)
        self.output_conv = nn.Conv3d(hidden_channels, data_channels, 3, padding=1)

        # Zero-init output for stable start: v_theta = 0 initially
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def sinusoidal_embedding(self, t):
        """Sinusoidal positional embedding for scalar timestep t.

        Args:
            t: (B,) tensor of timestep values in [0, 1]

        Returns:
            (B, time_dim) embedding
        """
        half_dim = self.time_dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(-emb_scale * torch.arange(half_dim, device=t.device).float())
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, z_t, t, condition):
        """Predict velocity field v(z_t, t, c).

        Args:
            z_t:       (B, C, H, W, D) noisy latent at time t
            t:         (B,) scalar timesteps in [0, 1]
            condition: (B, C, H, W, D) conditioning latent (VQ output)

        Returns:
            (B, C, H, W, D) predicted velocity
        """
        # Time embedding
        t_emb = self.sinusoidal_embedding(t)              # (B, time_dim)
        t_emb = self.time_mlp(t_emb)                      # (B, hidden_channels)

        # Concatenate noisy state and condition
        h = torch.cat([z_t, condition], dim=1)             # (B, 2C, H, W, D)
        h = self.input_conv(h)                             # (B, hidden, H, W, D)

        # Residual blocks with time conditioning
        for block in self.blocks:
            h = block(h, t_emb)

        # Output projection
        h = self.output_norm(h)
        h = F.silu(h)
        v = self.output_conv(h)                            # (B, C, H, W, D)

        return v


# ============================================================================
# Main Model: VQ-GAN + Latent Flow Matching (Single-Stage Training)
# ============================================================================

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
                self.n_embed,
                self.embed_dim,
                beta=0.25,
                remap=getattr(hparams, 'remap', None),
                sane_index_shape=getattr(hparams, 'sane_index_shape', False)
            )
            self.quant_conv = nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
            self.post_quant_conv = nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)

        # Initialize loss
        self.loss = instantiate_from_config(config['model']['params']["lossconfig"])
        self.discriminator = self.loss.discriminator

        # ====================================================================
        # Latent Flow Matching (conditional rectified flow in 3D latent space)
        # ====================================================================
        self.use_flow = getattr(hparams, 'use_flow', False)
        if self.use_flow:
            # Channel count after decoder.conv_in (= ch * ch_mult[-1])
            flow_data_ch = ddconfig['ch'] * ddconfig['ch_mult'][-1]
            flow_hidden  = getattr(hparams, 'flow_hidden_ch', 128)
            flow_blocks  = getattr(hparams, 'flow_num_blocks', 4)

            self.velocity_net = VelocityNet3D(
                data_channels=flow_data_ch,
                time_dim=256,
                hidden_channels=flow_hidden,
                num_blocks=flow_blocks,
            )

            self.flow_steps_train = getattr(hparams, 'flow_steps_train', 4)
            self.flow_steps       = getattr(hparams, 'flow_steps', 20)
            self.flow_lambda      = getattr(hparams, 'flow_lambda', 1.0)
            self.flow_grad_ckpt   = getattr(hparams, 'flow_grad_checkpoint', False)

            vnet_params = sum(p.numel() for p in self.velocity_net.parameters())
            print(f'[Flow] VelocityNet3D: {vnet_params/1e6:.1f}M params, '
                  f'data_ch={flow_data_ch}, hidden={flow_hidden}, '
                  f'blocks={flow_blocks}, train_steps={self.flow_steps_train}')

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

        # Save model names (for checkpoint loading)
        self.netg_names = {
            'encoder': 'encoder',
            'decoder': 'decoder',
            'quant_conv': 'quant_conv',
            'post_quant_conv': 'post_quant_conv',
            'net_g': 'net_g'
        }
        if not self.use_vae:
            self.netg_names['quantize'] = 'quantize'
        if self.use_flow:
            self.netg_names['velocity_net'] = 'velocity_net'

        self.netd_names = {'discriminator': 'discriminator', 'net_d': 'net_d'}

        # Configure optimizers
        self.configure_optimizers()

        self.upsample = torch.nn.Upsample(
            size=(hparams.cropsize, hparams.cropsize, hparams.cropsize),
            mode='trilinear'
        )
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
        # --- existing args (unchanged) ---
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
        parser.add_argument("--remap", type=str, default=None, help='remap indices')
        parser.add_argument("--sane_index_shape", action='store_true', help='return indices as bhw')
        parser.add_argument("--lr_g_factor", type=float, default=1.0, help='learning rate factor for generator')
        parser.add_argument("--vae", action='store_true', help='use VAE (KL) instead of VQ bottleneck')
        # --- flow matching args ---
        parser.add_argument("--use_flow", action='store_true',
                            help='enable latent flow matching for stochastic 3D generation')
        parser.add_argument("--flow_steps_train", type=int, default=4,
                            help='Euler ODE steps during training (backprop through all)')
        parser.add_argument("--flow_steps", type=int, default=20,
                            help='Euler ODE steps during inference')
        parser.add_argument("--flow_lambda", type=float, default=1.0,
                            help='weight for flow anchor MSE loss')
        parser.add_argument("--flow_hidden_ch", type=int, default=128,
                            help='hidden channels in velocity network')
        parser.add_argument("--flow_num_blocks", type=int, default=4,
                            help='number of FiLM ResBlocks in velocity network')
        parser.add_argument("--flow_grad_checkpoint", action='store_true',
                            help='gradient checkpointing for ODE steps (saves memory)')
        return parent_parser

    # ====================================================================
    # Encoder / Decoder / Forward (unchanged from original)
    # ====================================================================

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

    # ====================================================================
    # Adversarial loss helpers (unchanged)
    # ====================================================================

    def adv_loss_six_way(self, x, net_d, truth):
        loss = 0
        loss += self.add_loss_adv(a=x.permute(2, 1, 4, 3, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 4, 2, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0], net_d=net_d, truth=truth)
        loss = loss / 3
        return loss

    def get_xy_plane(self, x):
        return x.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]

    # ====================================================================
    # Flow matching helpers
    # ====================================================================

    def _flow_velocity_step(self, z_t, t_scalar, condition):
        """Single velocity prediction, optionally with gradient checkpointing."""
        t_tensor = torch.full((z_t.shape[0],), t_scalar, device=z_t.device)
        if self.training and self.flow_grad_ckpt:
            return grad_checkpoint(
                self.velocity_net, z_t, t_tensor, condition,
                use_reentrant=False
            )
        return self.velocity_net(z_t, t_tensor, condition)

    def _flow_ode_integrate(self, condition, num_steps, noise=None):
        """Integrate rectified flow ODE from noise (t=1) to data (t=0).

        Uses Euler method: z_{k+1} = z_k - v(z_k, t_k) * dt

        Args:
            condition: (B, C, H, W, D) conditioning latent
            num_steps: number of Euler steps
            noise:     optional starting noise; sampled if None

        Returns:
            z_0: (B, C, H, W, D) generated latent at t=0
        """
        if noise is None:
            z_t = torch.randn_like(condition)
        else:
            z_t = noise

        dt = 1.0 / num_steps
        for k in range(num_steps):
            t_val = 1.0 - k * dt                          # t goes 1.0 -> dt
            v = self._flow_velocity_step(z_t, t_val, condition)
            z_t = z_t - v * dt                             # Euler step toward t=0

        return z_t

    # ====================================================================
    # Generation (main forward pipeline)
    # ====================================================================

    def generation(self, batch, deterministic=False):
        # --- Data preparation (unchanged) ---
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

        # --- 2D Encode + VQ (unchanged) ---
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

        # --- Axial processing (unchanged) ---
        if self.hparams.downbranch > 1:
            quant = quant.permute(1, 2, 3, 0).unsqueeze(0)
            quant = nn.MaxPool3d((1, 1, self.hparams.downbranch))(quant)
            quant = quant.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]

        if self.hparams.resizebranch != 1:
            quant = quant.permute(1, 2, 3, 0).unsqueeze(0)
            quant = nn.Upsample(scale_factor=(1, 1, self.hparams.resizebranch),
                                mode='trilinear')(quant)
            quant = quant.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]

        # --- Project to decoder feature space ---
        quant = self.decoder.conv_in(quant)                # (Z', block_in, H/8, W/8)
        quant = quant.permute(1, 2, 3, 0).unsqueeze(0)     # (1, block_in, H/8, W/8, Z')

        # ================================================================
        # 3D Lifting: either deterministic (original) or flow matching
        # ================================================================
        if self.use_flow:
            c = quant                                       # condition for flow

            if self.training:
                # ---- Loss A: Anchor velocity MSE (stabilizer) ----
                # Teaches velocity_net basic flow behavior using the
                # current-best deterministic solution as pseudo-target.
                # z_0 = anchor target, z_1 = noise
                z_0_anchor = c.detach()
                z_1 = torch.randn_like(c)
                t = torch.rand(c.shape[0], device=c.device)           # (B,)

                # Rectified flow interpolation: z_t = (1-t)*z_0 + t*z_1
                t_expand = t[:, None, None, None, None]                # (B,1,1,1,1)
                z_t = (1.0 - t_expand) * z_0_anchor + t_expand * z_1

                v_pred = self.velocity_net(z_t, t, c)
                v_target = z_1 - z_0_anchor                           # constant velocity
                self.flow_mse = F.mse_loss(v_pred, v_target)

                # ---- Loss B: Full ODE -> self-supervised losses ----
                # Integrate ODE from noise, decode, apply adversarial +
                # L1 + MIP losses. Gradients flow through all Euler steps
                # back to velocity_net (and through c to the encoder).
                z_0_gen = self._flow_ode_integrate(c, self.flow_steps_train)

                # Decode flow output to isotropic volume via net_g
                self.XupX = self.net_g(z_0_gen, method='decode')['out0']

            else:
                # ---- Inference: Euler integration ----
                if deterministic:
                    # Zero noise -> reproduces ~ deterministic solution
                    noise = torch.zeros_like(c)
                else:
                    noise = None                                       # random noise

                z_0_gen = self._flow_ode_integrate(c, self.flow_steps, noise=noise)
                self.XupX = self.net_g(z_0_gen, method='decode')['out0']

        else:
            # ---- Original deterministic path (no flow) ----
            self.XupX = self.net_g(quant, method='decode')['out0']

        self.Xup = self.upsample(self.oriX)

    # ====================================================================
    # Uncertainty estimation via stochastic flow sampling
    # ====================================================================

    def generate_uncertainty(self, batch, n_samples=20):
        """Generate mean and std maps from multiple stochastic flow samples.

        This replaces the dropout-based stochasticity in the original model.
        Each sample uses different initial noise, producing genuinely
        different plausible isotropic volumes.

        Args:
            batch:     input data batch
            n_samples: number of stochastic forward passes

        Returns:
            mean_vol: (B, C, H, W, D) mean of samples
            std_vol:  (B, C, H, W, D) std of samples (uncertainty map)
        """
        assert self.use_flow, "generate_uncertainty requires --use_flow"
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                self.generation(batch, deterministic=False)
                samples.append(self.XupX.detach().clone())

        stacked = torch.stack(samples, dim=0)              # (N, B, C, H, W, D)
        mean_vol = stacked.mean(dim=0)
        std_vol  = stacked.std(dim=0)
        return mean_vol, std_vol

    # ====================================================================
    # Projection helper (unchanged)
    # ====================================================================

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
    # Generator loss (backward_g)
    # ====================================================================

    def backward_g(self):
        loss_g = 0
        loss_dict = {}

        # --- 3D adversarial loss on XupX (from flow or deterministic path) ---
        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)

        # --- L1 projection loss (self-supervised: compare projections to input) ---
        loss_l1 = self.add_loss_l1(
            a=self.get_projection(self.XupX, depth=self.uprate * self.hparams.skipl1,
                                  how=self.hparams.l1how),
            b=self.oriX[:, :, :, :, ::self.hparams.skipl1]
        )

        # --- MS-SSIM loss (optional) ---
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

        # --- 2D reconstruction losses (VQ perceptual + disc, unchanged) ---
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

        # --- Flow anchor MSE loss (only when flow is enabled) ---
        if self.use_flow and self.training and hasattr(self, 'flow_mse'):
            loss_dict['flow_mse'] = self.flow_mse
            loss_g += self.flow_mse * self.flow_lambda

        loss_dict['sum'] = loss_g
        return loss_dict

    # ====================================================================
    # Discriminator loss (unchanged)
    # ====================================================================

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

    # ====================================================================
    # Utilities (unchanged except configure_optimizers and save_checkpoint)
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
            list(self.net_g.parameters())
        )
        if not self.use_vae:
            ae_params += list(self.quantize.parameters())

        # Include velocity_net in the generator optimizer
        if self.use_flow:
            ae_params += list(self.velocity_net.parameters())

        opt_ae = torch.optim.Adam(ae_params, lr=lr_g, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(
            list(self.loss.discriminator.parameters()) + list(self.net_d.parameters()),
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
        if not self.use_vae:
            for k, v in self.quantize.state_dict().items():
                state_dict[f'quantize.{k}'] = v
        for k, v in self.quant_conv.state_dict().items():
            state_dict[f'quant_conv.{k}'] = v
        for k, v in self.post_quant_conv.state_dict().items():
            state_dict[f'post_quant_conv.{k}'] = v
        for k, v in self.discriminator.state_dict().items():
            state_dict[f'loss.discriminator.{k}'] = v

        # Save velocity_net if flow matching is enabled
        if self.use_flow:
            for k, v in self.velocity_net.state_dict().items():
                state_dict[f'velocity_net.{k}'] = v

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
