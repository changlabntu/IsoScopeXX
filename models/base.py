from typing import Dict, List, Optional, Union, Any

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import yaml
import torchvision.models as models
from pytorch_lightning.loggers import MLFlowLogger
from mlflow.exceptions import MlflowException
from taming.modules.losses.lpips import LPIPS
from torchmetrics.image.kid import KernelInceptionDistance
from networks.networks import get_scheduler
from networks.loss import GANLoss
from networks.registry import network_registry
from utils.data_utils import *
import tifffile as tiff


def _weights_init(m: nn.Module) -> None:
    """Initialize network weights using normal distribution."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class BaseModel(pl.LightningModule):
    """Base model for GAN training using PyTorch Lightning."""
    
    def __init__(self, hparams: Any, train_loader: Any, eval_loader: Any, checkpoints: str) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.epoch = 0
        self.dir_checkpoints = checkpoints

        # Model and loss names
        self.netg_names = {'net_g': 'netG'}
        self.netd_names = {'net_d': 'netD'}
        self.loss_g_names = ['loss_g']
        self.loss_d_names = ['loss_d']

        # Process hyperparameters
        hparams_dict = vars(hparams).copy()
        for k in hparams.not_tracking_hparams:
            hparams_dict.pop(k, None)
        hparams_dict.pop("not_tracking_hparams", None)
        self.hparams.update(hparams_dict)
        self.save_hyperparameters(self.hparams)

        # Initialize loss functions
        self._init_loss_functions()

        # Final
        self.hparams.update(vars(self.hparams))   # updated hparams to be logged in tensorboard
        #self.train_loader.dataset.shuffle_images()  # !!! shuffle again just to make sure
        #self.train_loader.dataset.shuffle_images()  # !!! shuffle again just to make sure

        self.all_label = []
        self.all_out = []
        self.all_loss = []

        self.buffer = {}
        self._epoch_time_sum = 0.0
        self._epoch_count = 0
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        os.makedirs('out', exist_ok=True)

    def _init_loss_functions(self) -> None:
        """Initialize loss functions."""
        self.criterionL1 = nn.L1Loss()
        self.criterionL2 = nn.MSELoss()
        if self.hparams.gan_mode == 'vanilla':
            self.criterionGAN = nn.BCEWithLogitsLoss()
        else:
            self.criterionGAN = GANLoss(self.hparams.gan_mode)
        self.lpips_fn = LPIPS().eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False
        self.kid = KernelInceptionDistance(feature=64, subset_size=16)

    def _get_mlflow_client_and_run_id(self):
        """Get the MLflow client and run ID.

        Only returns valid references on rank 0 to avoid duplicate logging in DDP.

        Returns:
            Tuple of (MlflowClient, run_id), or (None, None) if unavailable.
        """
        if not hasattr(self, 'trainer') or self.trainer is None:
            return None, None
        if not self.trainer.is_global_zero:
            return None, None
        for logger in self.loggers:
            if isinstance(logger, MLFlowLogger):
                client = logger.experiment
                run_id = logger.run_id
                if client is not None and run_id is not None:
                    return client, run_id
        return None, None

    def on_fit_start(self):
        """Log config.json and YAML config to MLflow artifacts at the start of training."""
        client, run_id = self._get_mlflow_client_and_run_id()
        if client is None:
            return
        try:
            config_path = os.path.join(self.dir_checkpoints, 'config.json')
            client.log_artifact(run_id, config_path, artifact_path="config")
            yaml_path = os.path.join(self.dir_checkpoints, self.hparams.yaml + '.yaml')
            client.log_artifact(run_id, yaml_path, artifact_path="config")
        except (MlflowException, OSError) as e:
            print(f"WARNING: Failed to log config artifacts to MLflow: {e}")

    @staticmethod
    def _save_3d_as_gif(arr, path, duration=100):
        """Create an animated GIF from Z slices of a 3D array.

        Args:
            arr: 3D numpy array with shape (Z, H, W).
            path: Output file path for the GIF.
            duration: Duration per frame in milliseconds.
        """
        if arr.ndim != 3:
            print(f"WARNING: _save_3d_as_gif expected 3D array, got {arr.ndim}D — skipping")
            return
        arr = arr.astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx - mn > 1e-8:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)
        arr_uint8 = (arr * 255).astype(np.uint8)

        frames = [Image.fromarray(arr_uint8[z]) for z in range(arr_uint8.shape[0])]

        if frames:
            frames[0].save(
                path, save_all=True, append_images=frames[1:],
                duration=duration, loop=0
            )

    def _log_gif_artifact(self, arr, prefix):
        """Create a GIF from a 3D array and log to MLflow artifacts.

        Args:
            arr: 3D numpy array with shape (Z, H, W).
            prefix: Filename prefix (e.g. 'train'), used as ``{prefix}_epoch_{N}.gif``.
        """
        client, run_id = self._get_mlflow_client_and_run_id()
        if client is None:
            return
        gif_path = os.path.join(self.dir_checkpoints, f'{prefix}_epoch_{self.epoch}.gif')
        try:
            self._save_3d_as_gif(arr, gif_path)
            if os.path.exists(gif_path):
                client.log_artifact(run_id, gif_path, artifact_path="images")
        except (MlflowException, OSError) as e:
            print(f"WARNING: Failed to log {prefix} GIF artifact to MLflow: {e}")
        finally:
            if os.path.exists(gif_path):
                os.remove(gif_path)

    def configure_optimizers(self):  ## THIS IS REQUIRED
        print('configuring optimizer being called....')
        print(self.netg_names.keys())
        print(self.netd_names.keys())
        print('configuring optimizer done')

        netg_parameters = []
        for g in self.netg_names.keys():
            netg_parameters = netg_parameters + list(getattr(self, g).parameters())
        print('Number of parameters in generator: ', sum(p.numel() for p in netg_parameters if p.requires_grad))

        netd_parameters = []
        for d in self.netd_names.keys():
            netd_parameters = netd_parameters + list(getattr(self, d).parameters())
        print('Number of parameters in discriminator: ', sum(p.numel() for p in netd_parameters if p.requires_grad))

        self.optimizer_g = optim.Adam(netg_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.optimizer_d = optim.Adam(netd_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.net_g_scheduler = get_scheduler(self.optimizer_g, self.hparams)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, self.hparams)
        # not using pl scheduler for now....
        return [self.optimizer_g, self.optimizer_d], []

    def add_loss_adv(self, a, net_d, truth):
        disc_logits = net_d(a)[0]
        if truth:
            # Create a tensor of ones with the same shape as disc_logits
            target = torch.ones_like(disc_logits)
        else:
            # Create a tensor of zeros with the same shape as disc_logits
            target = torch.zeros_like(disc_logits)
        
        adv = self.criterionGAN(disc_logits, target)
        return adv

    def add_loss_l1(self, a, b):
        l1 = self.criterionL1(a, b)
        return l1

    def add_loss_l2(self, a, b):
        l1 = self.criterionL2(a, b)
        return l1

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 1:
            self.generation(batch)
            loss_d = self.backward_d()
            if loss_d is not None:
                for k in list(loss_d.keys()):
                    if k != 'sum':
                        self.log(k, loss_d[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                return loss_d['sum']
            else:
                return None

        if optimizer_idx == 0:
            self.generation(batch)
            loss_g = self.backward_g()
            for k in list(loss_g.keys()):
                if k != 'sum':
                    self.log(k, loss_g[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
            return loss_g['sum']

    def on_train_epoch_start(self):
        """Records epoch start time for duration tracking."""
        self._epoch_start_time = time.time()

    def training_epoch_end(self, outputs):
        if hasattr(self, '_epoch_start_time'):
            epoch_duration = time.time() - self._epoch_start_time
            self._epoch_time_sum += epoch_duration
            self._epoch_count += 1
            avg = self._epoch_time_sum / self._epoch_count
            if self.trainer.is_global_zero:
                print(f'Epoch {self.epoch} | {epoch_duration//60:.0f}m {epoch_duration%60:.1f}s (avg: {avg//60:.0f}m {avg%60:.1f}s)')
            client, run_id = self._get_mlflow_client_and_run_id()
            if client is not None:
                client.log_metric(run_id, 'epoch_time', epoch_duration, step=self.epoch)

        self.log('lr_g', self.trainer.optimizers[0].param_groups[0]['lr'],
                 on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # checkpoint — rank 0 only to avoid duplicate writes
        if self.epoch % self.hparams.epoch_save == 0 and self.trainer.is_global_zero:
            for name in self.netg_names.keys():
                path_g = os.path.join(self.dir_checkpoints, '{}_model_epoch_{}.pth'.format(self.netg_names[name], self.epoch))
                torch.save(getattr(self, name), path_g)
                print("Checkpoint saved to {}".format(path_g))

            if self.hparams.save_d:
                for name in self.netd_names.keys():
                    path_d = os.path.join(self.dir_checkpoints, '{}_model_epoch_{}.pth'.format(self.netd_names[name], self.epoch))
                    torch.save(getattr(self, name), path_d)
                    print("Checkpoint saved to {}".format(path_d))

        self.net_g_scheduler.step()
        self.net_d_scheduler.step()

        self.reset_metrics()

        # GIF artifact is logged from validation's first batch in validation_epoch_end

        self.epoch += 1

    def get_metrics(self):
        pass

    def reset_metrics(self):
        pass

    def validation_step(self, batch, batch_idx):
        if not hasattr(self, 'generation'):
            return None
        self.generation(batch)
        if not (hasattr(self, 'Xup') and hasattr(self, 'XupX')):
            return None

        # Store first batch for artifact logging (rank 0 only).
        # Order for the GIF: data0 (input) -> enhanced -> data1 -> data2 -> ...
        # Extra views (xcube/ycube/...) are interpolated to the output shape so the
        # panels concatenate cleanly; single-view runs just give [input, enhanced].
        if batch_idx == 0 and self.trainer.is_global_zero:
            views = [self.Xup.detach().clone(), self.XupX.detach().clone()]
            for v in getattr(self, 'aux_views', []):
                v = nn.functional.interpolate(v, size=self.XupX.shape[2:], mode='trilinear')
                views.append(v.detach().clone())
            # Coarse multi-scale outputs (e.g. vqcleanM0aMS's 1/2 and 1/4 heads). Upsample
            # to the full-res grid with 'trilinear' so the panels concatenate cleanly and
            # render as smooth lower-res copies of the full-res output. ('nearest' was tried
            # to make the resolution difference obvious, but it blows each native voxel into
            # a hard block that reads as a checkerboard artifact.) Guarded — models without
            # gif_scales are unaffected and keep the original two-panel layout.
            for v in getattr(self, 'gif_scales', []):
                v = nn.functional.interpolate(v, size=self.XupX.shape[2:], mode='trilinear')
                views.append(v.detach().clone())
            self.val_views = views

        def to_rgb(t):
            return t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t

        with torch.no_grad():
            B, C, X, Y, Z = self.XupX.shape
            # XY view: all Z-slices → (B*Z, C, X, Y)
            xy_tgt  = self.Xup.permute(0, 4, 1, 2, 3).reshape(B * Z, C, X, Y)[::16]
            # YZ view: all X-slices → (B*X, C, Y, Z)  [requires X==Z for LPIPS batch match]
            yz_tgt  = self.Xup.permute(0, 2, 1, 3, 4).reshape(B * X, C, Y, Z)[::16]
            yz_pred = self.XupX.permute(0, 2, 1, 3, 4).reshape(B * X, C, Y, Z)[::16]
            # lpips_tgt: XY vs YZ of ground truth (baseline anisotropy)
            # lpips_pred: XY of target vs YZ of prediction (isotropy of output)
            lpips_tgt  = self.lpips_fn(to_rgb(xy_tgt), to_rgb(yz_tgt)).mean()
            lpips_pred = self.lpips_fn(to_rgb(xy_tgt), to_rgb(yz_pred)).mean()

        # on_epoch=True: PL averages across all batches and ranks automatically
        self.log('val_lpips_tgt', lpips_tgt, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_lpips_pred', lpips_pred, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # Held-out real X/Y projection error (vqcleanM0aSup0-style). Mirrors backward_g's
        # l1_x/l1_y: pool XupX along dim3 (->xcube) and dim2 (->ycube) by --aniso (op =
        # --l1how_xy) and L1-match the real views. cropz is train-only, so at val the
        # views' Z can differ from XupX by subsample rounding -> interpolate to XupX shape
        # first (as the GIF does). Guarded so models without lamb_xy/get_projection are
        # unaffected.
        if (hasattr(self, 'get_projection') and getattr(self.hparams, 'lamb_xy', 0) > 0
                and hasattr(self.hparams, 'aniso')
                and len(getattr(self, 'aux_views', [])) >= 2):
            r = self.hparams.aniso
            how_xy = getattr(self.hparams, 'l1how_xy', 'max')
            with torch.no_grad():
                xcube = nn.functional.interpolate(self.aux_views[0], size=self.XupX.shape[2:],
                                                  mode='trilinear')
                ycube = nn.functional.interpolate(self.aux_views[1], size=self.XupX.shape[2:],
                                                  mode='trilinear')
                val_l1_x = self.add_loss_l1(
                    a=self.get_projection(self.XupX, depth=r, how=how_xy, axis=3),
                    b=self.get_projection(xcube, depth=r, how=how_xy, axis=3))
                val_l1_y = self.add_loss_l1(
                    a=self.get_projection(self.XupX, depth=r, how=how_xy, axis=2),
                    b=self.get_projection(ycube, depth=r, how=how_xy, axis=2))
            self.log('val_l1_x', val_l1_x, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_l1_y', val_l1_y, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # KID: accumulate XY(tgt) vs YZ(pred) across all val batches
        # KernelInceptionDistance expects uint8 [0, 255]
        def to_uint8(t):
            t = to_rgb(t).clamp(-1, 1)
            return ((t + 1) / 2 * 255).to(torch.uint8)

        self.kid.update(to_uint8(xy_tgt), real=True)
        self.kid.update(to_uint8(yz_pred), real=False)
        return None

    def testing_step(self, batch, batch_idx):
        self.generation(batch)

    def validation_epoch_end(self, x):
        kid_mean, kid_std = self.kid.compute()
        self.log('val_kid', kid_mean, logger=True, sync_dist=True)
        self.kid.reset()

        # Log a GIF from validation's first batch (stored in validation_step), every 5 epochs.
        # Panels are laid out left-to-right as data0 (input) | enhanced | data1 | data2 | ...
        if (self.epoch % 5 == 0 and self.trainer.is_global_zero
                and hasattr(self, 'val_views') and len(self.val_views) >= 2):
            def to_panel(t):
                # (B, C, X, Y, Z) -> (X, C*Y, Z): channels stacked vertically, Z is the frame width
                return np.concatenate([t[:, c, ::].squeeze().detach().cpu().numpy()
                                       for c in range(t.shape[1])], 1)
            panels = [to_panel(v) for v in self.val_views]
            concat_arr = np.concatenate(panels, 2)  # side by side along Z (frame width)
            self._log_gif_artifact(concat_arr, 'val')

        return None

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        if 'loss' in tqdm_dict:
            del tqdm_dict['loss']
        return tqdm_dict

    def generation(self, batch):
        pass

    def backward_g(self):
        pass

    def backward_d(self):
        pass

    def set_networks(self, net='all'):
        """Create network instances using the network registry.
        
        Args:
            net: Which networks to return ('all', 'g', or 'd')
            
        Returns:
            Generator and/or discriminator networks based on the net parameter
        """
        # Common parameters for both networks
        common_params = {
            'input_nc': self.hparams.input_nc,
            'output_nc': self.hparams.output_nc,
            'norm': self.hparams.norm,
            'norm_layer': nn.BatchNorm2d
        }
        
        # Generator-specific parameters
        generator_params = {
            **common_params,
            'ngf': self.hparams.ngf,
            'use_dropout': self.hparams.mc,
            'final': self.hparams.final,
            'mc': self.hparams.mc
        }
        
        # Discriminator-specific parameters
        discriminator_params = {
            **common_params,
            'ndf': self.hparams.ndf
        }
        
        # Create networks
        net_g = network_registry.get_generator(self.hparams.netG, **generator_params)
        net_d = network_registry.get_discriminator(self.hparams.netD, **discriminator_params)
        
        # Return requested networks
        if net == 'all':
            return net_g, net_d
        elif net == 'g':
            return net_g
        elif net == 'd':
            return net_d

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


