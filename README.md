# Compression-Aware Isotropic Super-Resolution for Expansion Microscopy

A single-stage, self-supervised framework for whole-organ nanoscale imaging that simultaneously enhances resolution and compresses data.

## Overview

This framework addresses critical challenges in expansion microscopy (ExM):
- **Resolution anisotropy**: Achieves up to 8× axial resolution enhancement without requiring high-resolution ground truth
- **Storage burden**: Provides ~128× compression, reducing storage by ~1000× compared to fully isotropic volumes  
- **Scalability**: Processes raw slices directly through a 2D encoder + lightweight 3D decoder architecture
- **Robustness**: Handles depth-varying aberrations and preserves cross-slice continuity

## Key Features

- Self-supervised training (no ground truth needed)  
- On-demand isotropic reconstruction from compact latents  
- Scales to organ-level datasets  
- Compatible with clinical biomarker discovery workflows  

## Configuration

Edit `env/jsn/aisr.json` to configure training:

### Core Settings
- `dataset` - Dataset for training
- `model` - Architecture (`ae0iso0tccutvqq` for VQ-VAE, `ae0iso0tccut` for KL-VAE)

### Training Parameters
- `batch_size`, `n_epochs`, `epoch_save` - Batch size, total epochs, checkpoint frequency
- `lr`, `beta1`, `lr_policy`, `n_epochs_decay` - Learning rate and optimizer settings
- `cropsize`, `cropz` - Spatial (X,Y) and Z dimensions of training crops
- `flip`, `rotate`, `resize` - Data augmentation toggles

### Loss Weights
- `lamb`, `adv` - Reconstruction/regularization and adversarial losses
- `resizebranch` - Match latent features to physical Z,X,Y dimensions

### PatchNCE Settings (Optional)
- `num_patches` - Patches per layer (default: 256)
- `nce_T` - Temperature (default: 0.07, lower = more selective)
- `use_mlp` - Enable MLP (default: False)
- `c_mlp` - MLP channels (default: 256)
- `fWhich` - Target layers (default: all)
- `lbNCE` - Loss weight (default: 1.0)
- `nce_includes_all_negatives_from_minibatch` - Cross-batch negatives (default: False)

### VQ-VAE/VQGAN Settings
- `embed_dim` - Latent embedding channels (default: 4)
- `n_embed` - Codebook size, 2^n entries (default: 256 for 8-bit)
- `kl_weight` - KL divergence weight (default: 0.000001)
- `disc_weight` - 2D discriminator weight (default: 0.5)

## Samples of Results (visualized with avivator by HIDIVELAB (https://hidivelab.org))

### Human surgical brain, Golgi stain
https://avivator.gehlenborglab.org/?image_url=https://storage.googleapis.com/brc_data/Golgi.ome.tiff

### Stitching of large NA acquisition of Drosophila neurons (50X ExM)
https://avivator.gehlenborglab.org/?image_url=https://storage.googleapis.com/brc_data/X2527T102MM_stitching.ome.tiff

### Total protein staining of Drosophila brain
https://avivator.gehlenborglab.org/?image_url=https://storage.googleapis.com/brc_data/totalprotein092625.ome.tiff

### Confocal TH neurons (10X ExM)
https://avivator.gehlenborglab.org/?image_url=https://storage.googleapis.com/brc_data/thx10.ome.tiff