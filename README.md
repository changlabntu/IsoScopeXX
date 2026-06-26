# Compression-Aware Isotropic Super-Resolution for Expansion Microscopy

A single-stage, self-supervised framework for whole-organ nanoscale imaging that simultaneously enhances resolution and compresses data.

## Overview

This framework addresses critical challenges in expansion microscopy (ExM):
- **Resolution anisotropy**: Achieves up to 8x axial resolution enhancement without requiring high-resolution ground truth
- **Storage burden**: Provides ~128x compression, reducing storage by ~1000x compared to fully isotropic volumes
- **Scalability**: Processes raw slices directly through a 2D encoder + lightweight 3D decoder architecture
- **Robustness**: Handles depth-varying aberrations and preserves cross-slice continuity

## Key Features

- Self-supervised training (no ground truth needed)  
- On-demand isotropic reconstruction from compact latents  
- Scales to organ-level datasets  
- Compatible with clinical biomarker discovery workflows  

## Samples of Results

| Sample (Visualized with [Avivator](https://avivator.gehlenborglab.org) by [HIDIVE Lab](https://hidivelab.org)) | Link |
|--------|------|
| Human surgical brain, Golgi stain | [View](https://avivator.gehlenborglab.org/?image_url=https://storage.googleapis.com/brc_data/Golgi.ome.tiff) |
| Stitching of large NA acquisition of Drosophila neurons (50X ExM) | [View](https://avivator.gehlenborglab.org/?image_url=https://storage.googleapis.com/brc_data/X2527T102MM_stitching.ome.tiff) |
| Total protein staining of Drosophila brain | [View](https://avivator.gehlenborglab.org/?image_url=https://storage.googleapis.com/brc_data/totalprotein092625.ome.tiff) |
| Confocal TH neurons (10X ExM) | [View](https://avivator.gehlenborglab.org/?image_url=https://storage.googleapis.com/brc_data/thx10.ome.tiff) |

## Getting Started

```bash
# Create conda environment
conda create -n isoscope python=3.10
conda activate isoscope

# Install PyTorch (adjust for your CUDA version)
# CUDA 11.8:
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.8:
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

## Training

### Configuration

Training is configured through two files in `cfg/`:

- **`cfg/env.json`**: per-machine paths `DATASET`, `LOGS`, `TRACKING_URI`, keyed by environment name
- **`cfg/{experiment}.yaml`**: training hyperparameters such as model, dataset, direction, learning rate

The `--yaml` flag selects which YAML config to use. See YAML files in `cfg/` for available options and parameters.

| Category | Parameter | Description |
|----------|-----------|-------------|
| Core | `dataset` | Dataset for training |
| Core | `models` | Architecture (`ae0iso0tccutvqq` for VQ-VAE, `ae0iso0tccut` for KL-VAE) |
| Training | `batch_size`, `n_epochs`, `epoch_save` | Batch size, total epochs, checkpoint frequency |
| Training | `lr`, `beta1`, `lr_policy`, `n_epochs_decay` | Learning rate and optimizer settings |
| Training | `cropsize`, `cropz` | Spatial (X,Y) and Z dimensions of training crops |
| Training | `flip`, `rotate`, `resize` | Data augmentation toggles |
| Loss | `lamb`, `adv` | Reconstruction/regularization and adversarial losses |
| Loss | `resizebranch` | Match latent features to physical Z,X,Y dimensions |
| PatchNCE | `num_patches` | Patches per layer (default: 256) |
| PatchNCE | `nce_T` | Temperature (default: 0.07, lower = more selective) |
| PatchNCE | `use_mlp` | Enable MLP (default: False) |
| PatchNCE | `c_mlp` | MLP channels (default: 256) |
| PatchNCE | `fWhich` | Target layers (default: all) |
| PatchNCE | `lbNCE` | Loss weight (default: 1.0) |
| PatchNCE | `nce_includes_all_negatives_from_minibatch` | Cross-batch negatives (default: False) |
| VQ-VAE | `embed_dim` | Latent embedding channels (default: 4) |
| VQ-VAE | `n_embed` | Codebook size, 2^n entries (default: 256 for 8-bit) |
| VQ-VAE | `kl_weight` | KL divergence weight (default: 0.000001) |
| VQ-VAE | `disc_weight` | 2D discriminator weight (default: 0.5) |

### Experiment Tracking

Each run is tracked by **TensorBoard** and **MLflow** with a shared timestamp.

MLflow operates in two modes:

- **Server mode**: when `TRACKING_URI` is set in `cfg/env.json`, training connects to an MLflow Tracking Server. The server must be reachable or training aborts.
- **Local tracking**: when no `TRACKING_URI` is configured, MLflow logs to a local SQLite database at `${LOGS}/mlflow/mlflow.db`.

```bash
# TensorBoard
tensorboard --logdir /path/to/logs/{dataset}/{prj}/logs/TensorBoardLogger

# MLflow Tracking Server
LOGS=/path/to/logs

mlflow server \
  --backend-store-uri sqlite:///${LOGS}/mlflow/mlflow.db \
  --artifacts-destination ${LOGS}/mlflow/mlartifacts \
  --host 0.0.0.0 --port 5002
```

Tracking URI priority: CLI `--tracking_uri` > `TRACKING_URI` in `cfg/env.json` > local SQLite.

For detailed path assembly logic, see [docs/paths.md](docs/paths.md).
