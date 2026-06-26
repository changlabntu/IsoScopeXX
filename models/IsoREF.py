from models.base import BaseModel
import copy
import torch
import numpy as np
import torch.nn as nn
from models.base import VGGLoss
from networks.networks_cut import Normalize, init_net, PatchNCELoss
from models.CUT import PatchSampleF3D
from typing import Dict


class GAN(BaseModel):
    """GAN model for image-to-image translation with cycle consistency and contrastive learning.
    
    Features:
    - Supports 3D image translation
    - Optional cycle consistency loss
    - Optional contrastive learning (CUT)
    - Multi-scale discriminators
    """
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)
        from networks.EncoderDecoder.edclean import Generator
        
        # Initialize main generator and discriminator
        self.hparams.final = 'tanh'
        self._init_main_networks()
        
        # Initialize cycle networks if needed
        if not self.hparams.nocyc:
            self._init_cycle_networks()
        
        # Initialize CUT networks if needed
        if not self.hparams.nocut:
            self._init_cut_networks()
        
        # Configure optimizers and upsampling
        self.configure_optimizers()
        self._init_upsampling()
    
    def _init_main_networks(self):
        """Initialize main generator and discriminator networks."""
        self.net_g = self._create_generator(encode=self.hparams.gdim[:2], decode=self.hparams.gdim[2:])
        _, self.net_d = self.set_networks()
        self.netg_names = {'net_g': 'net_g'}
        self.netd_names = {'net_d': 'net_d'}
    
    def _init_cycle_networks(self):
        """Initialize networks for cycle consistency."""
        self.net_gback = self._create_generator(encode='3d', decode='1d')
        _, self.net_dzy = self.set_networks()
        _, self.net_dzx = self.set_networks()
        
        # Update network names
        self.netg_names['net_gback'] = 'net_gback'
        self.netd_names.update({'net_dzy': 'net_dzy', 'net_dzx': 'net_dzx'})
    
    def _init_cut_networks(self):
        """Initialize networks for contrastive learning."""
        netF = PatchSampleF3D(use_mlp=self.hparams.use_mlp,
                           init_type='normal',
                           init_gain=0.02,
                           gpu_ids=[],
                           nc=self.hparams.c_mlp)
        self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
        
        # Setup feature extraction
        feature_shapes = [x * self.hparams.ngf for x in [1, 2, 4, 8]]
        self.netF.create_mlp(feature_shapes)
        
        # Initialize feature weights if not specified
        if self.hparams.fWhich is None:
            self.hparams.fWhich = [1] * len(feature_shapes)
        
        # Initialize NCE criteria
        self.criterionNCE = [PatchNCELoss(opt=self.hparams) for _ in range(4)]
        self.netg_names['netF'] = 'netF'
    
    def _create_generator(self, encode, decode):
        """Create a generator with specified encoding/decoding dimensions."""
        from networks.EncoderDecoder.edclean import Generator
        return Generator(
            n_channels=self.hparams.input_nc,
            out_channels=self.hparams.output_nc,
            nf=self.hparams.ngf,
            norm_type=self.hparams.norm,
            final=self.hparams.final,
            mc=self.hparams.mc,
            encode=encode,
            decode=decode
        )
    
    def _init_upsampling(self):
        """Initialize upsampling layer."""
        self.upsample = torch.nn.Upsample(
            size=(
                self.hparams.cropsize,
                self.hparams.cropsize,
                self.hparams.cropsize #// self.hparams.dsp * self.hparams.uprate
            ),
            mode='trilinear'
        )
        self.downsample = torch.nn.Upsample(
            size=(
                self.hparams.cropsize,
                self.hparams.cropsize,
                self.hparams.cropz // self.hparams.dsp
            ),
            mode='trilinear'
        )
        self.upsample2d = torch.nn.Upsample(size=(self.hparams.cropsize, self.hparams.cropsize),
                                            mode='bicubic', align_corners=True)
            #scale_factor=(1, 8), mode='bicubic', align_corners=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--gdim", type=str, default='1d_3d', help='dimension of the generator')
        parser.add_argument("--dsp", type=int, default=1)
        parser.add_argument("--lambB", type=int, default=1)
        parser.add_argument("--l1how", type=str, default='max')
        parser.add_argument("--uprate", type=int, default=8)
        parser.add_argument("--skipl1", type=int, default=1)
        parser.add_argument("--nocyc", action='store_true')
        parser.add_argument("--nocut", action='store_true')
        # PatchNCE losses
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--lbNCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--use_mlp', action='store_true', default=False)
        parser.add_argument("--c_mlp", dest='c_mlp', type=int, default=256, help='channel of mlp')
        parser.add_argument('--fWhich', nargs='+', help='which layers to have NCE loss', type=int, default=None)
        parser.add_argument('--downz', type=int, default=1, help='extra downsample factor')
        parser.add_argument('--cubic', action='store_true', default=False)
        return parent_parser

    def generation(self, batch):
        #if self.hparams.downz > 1:
        #    batch['img'][0] = torch.nn.Upsample(scale_factor=(1, 1, 1 / self.hparams.downz), mode='trilinear')(batch['img'][0])
        #    batch['img'][0] = torch.nn.Upsample(scale_factor=(1, 1, self.hparams.downz), mode='trilinear')(batch['img'][0])

        if self.hparams.cropz > 0:
            z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            batch['img'][0] = batch['img'][0][:, :, :, :, z_init:z_init + self.hparams.cropz]
            # batch['img'][1] = batch['img'][1][:, :, :, :, z_init:z_init + self.hparams.cropz]

        # extra downsample
        if self.hparams.dsp > 1:
            batch['img'][0] = nn.Upsample(scale_factor=(1, 1, 1 / self.hparams.dsp), mode='area')(batch['img'][0])

        self.oriX = batch['img'][0]  # (B, C, X, Y, Z) # original
        #self.oriY = batch['img'][1]  # (B, C, X, Y, Z) # original

        # X-Y permute
        if np.random.randint(2) == 1:
            self.oriX = self.oriX.permute(0, 1, 3, 2, 4)

        if self.hparams.cubic:
            self.oriX = self.oriX.permute(2, 1, 3, 4, 0).squeeze(0)  # (X, C, Y, Z)
            self.Xup = self.upsample2d(self.oriX)
            self.Xup = self.Xup.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            self.Xup = self.upsample(self.oriX)  # (B, C, X, Y, Z)

        #self.Yup = self.upsample(self.oriY)  # (B, C, X, Y, Z)

        self.goutz = self.net_g(self.Xup, method='encode')
        self.XupX = self.net_g(self.goutz, method='decode')['out0']

        if not self.hparams.nocyc:
            self.XupXback = self.net_gback(self.XupX)['out0']

    def get_xy_plane(self, x):  # (B, C, X, Y, Z)
        return x.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]  # (Z, C, X, Y, B)

    def adv_loss_six_way(self, x: torch.Tensor, net_d: nn.Module, truth: bool) -> torch.Tensor:
        """Calculate adversarial loss from randomly selected view angles.
        
        Args:
            x: Input tensor (B, C, X, Y, Z)
            net_d: Discriminator network
            truth: Whether input should be classified as real
        
        Returns:
            Average adversarial loss across selected views
        """
        # Define permutation groups for each view plane
        view_groups = {
            0: [  # YZ-plane views
                (x.permute(2, 1, 4, 3, 0), 2),  # (X, C, Z, Y)
                (x.permute(2, 1, 3, 4, 0), 3),  # (X, C, Y, Z)
            ],
            1: [  # XZ-plane views
                (x.permute(3, 1, 4, 2, 0), 2),  # (Y, C, Z, X)
                (x.permute(3, 1, 2, 4, 0), 3),  # (Y, C, X, Z)
            ],
            2: [  # XY-plane views (weighted more heavily)
                (x.permute(4, 1, 2, 3, 0), None),  # (Z, C, X, Y)
                (x.permute(4, 1, 3, 2, 0), None),  # (Z, C, Y, X)
            ]
        }
        
        # Randomly select a view plane
        view_idx = np.random.randint(3)
        views = view_groups[view_idx]
        
        # Calculate losses for selected views
        loss = 0
        for permuted_x, flip_dim in views:
            # Extract the base view
            view = permuted_x[:, :, :, :, 0]
            weight = 2 if view_idx == 2 else 1
            
            # Add loss for base view
            loss += weight * self.add_loss_adv(a=view, net_d=net_d, truth=truth)
            
            # Add loss for flipped view if needed
            if flip_dim is not None:
                flipped_view = torch.flip(view, [flip_dim])
                loss += weight * self.add_loss_adv(a=flipped_view, net_d=net_d, truth=truth)
        
        return loss / 4

    def adv_loss_six_way_y(self, x: torch.Tensor, truth: bool) -> torch.Tensor:
        """Calculate adversarial loss using specialized discriminators for different views.
        
        Args:
            x: Input tensor (B, C, X, Y, Z)
            truth: Whether input should be classified as real
        
        Returns:
            Average adversarial loss across all views
        """
        loss = 0
        loss += self.add_loss_adv(a=x.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],  # (X, C, Z, Y)
                                        net_d=self.net_dzy, truth=truth)
        loss += self.add_loss_adv(a=x.permute(2, 1, 3, 4, 0)[:, :, :, :, 0],  # (X, C, Y, Z)
                                        net_d=self.net_dzy, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 4, 2, 0)[:, :, :, :, 0],  # (Y, C, Z, X)
                                        net_d=self.net_dzx, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 2, 4, 0)[:, :, :, :, 0],  # (Y, C, X, Z)
                                        net_d=self.net_dzx, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0],  # (Z, C, X, Y)
                                        net_d=self.net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 3, 2, 0)[:, :, :, :, 0],  # (Z, C, Y, X)
                                        net_d=self.net_d, truth=truth)
        loss = loss / 6
        return loss

    def backward_g(self) -> Dict[str, torch.Tensor]:
        """Calculate generator losses and return loss dictionary.
        
        Returns:
            Dictionary containing individual losses and their sum
        """
        loss_dict = {}
        
        # Adversarial loss
        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)
        loss_dict['axx'] = axx
        
        # L1 loss with projection
        if self.hparams.l1how != 'maxmin':
            loss_l1 = self.add_loss_l1(a=self.get_projection(self.XupX, depth=self.hparams.uprate
                                                                              * self.hparams.skipl1, how='max'),
                                        b=self.get_projection(self.Xup, depth=self.hparams.uprate
                                                                                * self.hparams.skipl1, how='max'))
            loss_l1 += self.add_loss_l1(a=self.get_projection(self.XupX, depth=self.hparams.uprate
                                                                                * self.hparams.skipl1, how='min'),
                                        b=self.get_projection(self.Xup, depth=self.hparams.uprate
                                                                                * self.hparams.skipl1, how='min'))
            loss_l1 = loss_l1 / 2

        elif self.hparams.l1how != 'downup':
            self.XupXDU = self.downsample(self.upsample(self.XupX))  # (B, C, X, Y, Z)
            loss_l1 = self.add_loss_l1(a=self.XupXDU, b=self.Xup)

        else:
            loss_l1 = self.add_loss_l1(a=self.get_projection(self.XupX, depth=self.hparams.uprate
                                                                              * self.hparams.skipl1,
                                                             how=self.hparams.l1how),
                                       b=self.oriX[:, :, :, :, ::self.hparams.skipl1])

        loss_dict['l1'] = loss_l1
        
        # Initialize total loss
        loss_g = axx * self.hparams.adv + loss_l1 * self.hparams.lamb
        
        # Cycle consistency losses
        if not self.hparams.nocyc:
            cycle_losses = self._compute_cycle_losses()
            loss_dict.update(cycle_losses)
            loss_g += cycle_losses['gback'] * self.hparams.adv + cycle_losses['l1b'] * self.hparams.lambB
        
        # Contrastive losses
        if not self.hparams.nocut:
            nce_loss = self._compute_nce_loss()
            loss_dict['nce'] = nce_loss
            loss_g += nce_loss * self.hparams.lbNCE
        
        loss_dict['sum'] = loss_g
        return loss_dict
    
    def _compute_cycle_losses(self) -> Dict[str, torch.Tensor]:
        """Compute cycle consistency losses."""
        gback = self.adv_loss_six_way_y(self.XupXback, truth=True)
        loss_l1_back = self.add_loss_l1(a=self.XupXback, b=self.Xup)
        return {'gback': gback, 'l1b': loss_l1_back}
    
    def _compute_nce_loss(self) -> torch.Tensor:
        """Compute contrastive learning loss."""
        feat_q = self.goutz
        feat_k = self.net_g(self.XupX, method='encode')
        
        # Get patches
        feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)
        
        # Calculate NCE loss
        nce_losses = [crit(f_q, f_k) * f_w for f_q, f_k, crit, f_w in 
                     zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.hparams.fWhich)]
        return sum(loss.mean() for loss in nce_losses) / len(self.criterionNCE)

    def backward_d(self) -> Dict[str, torch.Tensor]:
        """Calculate discriminator losses and return loss dictionary.
        
        Returns:
            Dictionary containing individual losses and their sum
        """
        loss_dict = {}
        
        # Main discriminator losses
        dxx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=False)
        dx = self.add_loss_adv(a=self.get_xy_plane(self.oriX), net_d=self.net_d, truth=True)
        loss_dict['dxx_x'] = dxx + dx
        
        # Initialize total loss
        loss_d = (dxx + dx) * self.hparams.adv

        # Cycle discriminator losses
        if not self.hparams.nocyc:
            dyy = self.adv_loss_six_way_y(self.XupXback, truth=False)
            dy = self.adv_loss_six_way_y(self.oriX, truth=True)
            loss_dict['dyy'] = dyy + dy
            loss_d += (dyy + dy) * self.hparams.adv
        
        loss_dict['sum'] = loss_d
        return loss_dict

    def get_projection(self, x: torch.Tensor, depth: int, how: str = 'mean') -> torch.Tensor:
        """Project 3D tensor to 2D using various methods.
        
        Args:
            x: Input tensor (B, C, H, W, D)
            depth: Depth for unfolding or stride for downsampling
            how: Projection method ('dsp', 'mean', or 'max')
        
        Returns:
            Projected tensor
        """
        if how == 'dsp':
            stride = self.hparams.uprate * self.hparams.skipl1
            offset = self.hparams.uprate // 2
            return x[:, :, :, :, offset::stride]
        
        x = x.unfold(-1, depth, depth)
        return x.mean(dim=-1) if how == 'mean' else x.max(dim=-1)[0]


# USAGE
# train.py --jsn womac4X --prj IsoREF/3dLb10mlpNCE10 --models IsoREF --env runpod --n_epochs 801 --gdim 3d_3d --nocyc --use_mlp --epoch_save 100 --lbNCE 10 --lamb 10