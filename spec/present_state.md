# IsoScopeXX Project - Present State

**Last Updated:** October 30, 2025  
**Status:** Active Development

---

## Project Overview

IsoScopeXX is a deep learning framework for 3D medical image enhancement and translation using VQGAN (Vector Quantized GAN) and related architectures. The project focuses on isotropic super-resolution and volumetric image-to-image translation for microscopy and medical imaging data.

### Core Capabilities
- 3D volumetric image translation
- Isotropic super-resolution (enhancing z-axis resolution)
- VQGAN-based autoencoder with vector quantization
- Contrastive learning (CUT - Contrastive Unpaired Translation)
- Multi-view adversarial training (6-way discriminator)
- Support for multiple model architectures

---

## Architecture

### Technology Stack
- **Framework:** PyTorch + PyTorch Lightning
- **Training:** Multi-GPU (DDP) with TensorBoard logging
- **Data Processing:** Albumentations, TIFF/PIL image loading
- **Models:** Custom VQGAN, Autoencoder-KL, CUT-based architectures

### Project Structure

```
IsoScopeXX/
├── train.py                    # Main training entry point
├── run.sh                      # Simplified training launcher
├── models/                     # Model architectures
│   ├── base.py                # BaseModel (PyTorch Lightning)
│   ├── ae0iso0tccutvqq.py     # VQGAN with VQ quantization (primary)
│   ├── ae0iso0tccut.py        # VQGAN with KL divergence
│   ├── ae0iso0tccutvqqssim.py # VQGAN + SSIM loss
│   ├── IsoREF.py              # Reference isotropic GAN
│   └── CUT.py                 # Contrastive learning helpers
├── networks/                   # Network building blocks
│   ├── networks.py            # Generator/discriminator definitions
│   ├── networks_cut.py        # CUT-specific networks
│   ├── registry.py            # Network factory/registry
│   ├── loss.py                # Loss functions
│   └── EncoderDecoder/        # Encoder-decoder architectures
├── dataloader/                 # Data loading
│   └── data_multi.py          # PairedImageDataset (3D slices)
├── ldm/                        # Latent Diffusion Model components
│   ├── vqgan.yaml             # VQGAN architecture config (active)
│   ├── ldmaex2.yaml           # Alternative LDM config
│   ├── modules/               # Model modules (encoder, decoder, quantizers)
│   └── util.py                # LDM utilities
├── taming/                     # Taming Transformers integration
│   └── modules/               # VQ modules, losses
├── utils/                      # Utilities
│   ├── get_args.py            # Argument parser
│   ├── data_utils.py          # Data utilities
│   └── make_config.py         # Config management
└── env/                        # Environment configuration
    ├── env                    # Environment paths (LOGS, DATASET)
    └── jsn/                   # Experiment configs (JSON)
        └── aisr.json          # Active AISR experiment config
```

---

## Current Configuration

### Active Model: `ae0iso0tccutvqq`

**Architecture:** VQGAN with Vector Quantization
- **Encoder-Decoder:** Based on `ldm/vqgan.yaml` configuration
  - Channels: 64, multipliers: [1, 2, 2, 4]
  - Z-channels: 4
  - Latent interpolator: `ed023e`
- **Quantizer:** VectorQuantizer2 (256 codes, 4-dim embeddings)
- **Generator:** 3D U-Net-style with skip connections
- **Discriminator:** PatchGAN (16x16 patches)
- **Losses:**
  - VQGAN perceptual + adversarial (LPIPS)
  - L1 reconstruction (with max pooling projection)
  - Optional: Contrastive NCE loss (CUT)
  - VQ commitment loss (codebook learning)

### Training Configuration (`env/jsn/aisr.json`)

**Dataset & Preprocessing:**
- Dataset: `X2527T102MM`
- Direction: `x3d0` (single-channel input)
- Crop size: 128x128x16 (X, Y, Z)
- Normalization: `00` (no normalization)
- Rotation augmentation: enabled
- No resize/precrop

**Model Hyperparameters:**
- L1 loss weight (`lamb`): 10
- L1 projection method (`l1how`): `max` (max pooling)
- Skip layers for L1 (`skipl1`): 4
- CUT disabled (`nocut`: true, `lbNCE`: 0)
- Two-channel mode (`tc`): false
- YAML config: `vqgan`

**Training Settings:**
- Batch size: 1
- Epochs: 10,001
- Learning rate: 0.0002 (Adam, beta1=0.5)
- LR schedule: cosine decay after 100 epochs
- Checkpoint save: every 100 epochs
- GAN mode: vanilla (BCEWithLogitsLoss)
- Adversarial weight (`adv`): 1.0

**Environment:**
- Compute env: `brcb`
- Logs: `/home/gary/workspace/logs/`
- Data: `/home/gary/workspace/Data/`

---

## Training Workflow

### Argument Parsing Flow (Recently Updated)
1. Load base arguments from `utils/get_args.py`
2. **Parse `--jsn` flag** to determine JSON config file
3. **Load JSON defaults** from `env/jsn/<jsn>.json`
4. **Resolve model name** from JSON or CLI (`--models`)
5. **Import model** and add model-specific arguments
6. **Final parse** with JSON defaults + CLI overrides
7. Load environment paths from `env/env`

**Benefits:** Minimal CLI commands; most settings in JSON.

### Running Training

**Current command:**
```bash
NO_ALBUMENTATIONS_UPDATE=1 python train.py --jsn aisr --prj dist2/max10skip4
```

**Command breakdown:**
- `NO_ALBUMENTATIONS_UPDATE=1`: Prevent albumentations package updates
- `--jsn aisr`: Load config from `env/jsn/aisr.json`
- `--prj dist2/max10skip4`: Project name (output directory)

All other parameters loaded from JSON.

### Data Loading Pipeline

1. **PairedImageDataset** (`dataloader/data_multi.py`):
   - Loads paired 3D TIFF images from `{DATASET}/X2527T102MM/train/`
   - Expects directory structure: `<root>/<direction>/`
   - Finds common filenames across paired directories
   - Applies Albumentations transforms:
     - Center crop (if precrop > 0)
     - Rotation (if enabled)
     - Random crop to `cropsize`
     - Converts to tensor

2. **Batch structure:**
   - `img`: List of tensors (B, C, X, Y, Z)
   - `labels`: Per-sample labels
   - `filenames`: Source file paths

### Model Training Loop (PyTorch Lightning)

1. **Optimizer 1 (Discriminator):**
   - `generation()`: Forward pass through encoder → quantizer → decoder → generator
   - `backward_d()`: Compute discriminator losses
     - Real vs. fake classification
     - VQGAN discriminator loss
     - Multi-view adversarial (6 orientations)

2. **Optimizer 0 (Generator):**
   - `generation()`: Same forward pass
   - `backward_g()`: Compute generator losses
     - Adversarial loss (fool discriminator)
     - L1 reconstruction loss (projected via max pooling)
     - VQGAN perceptual + adversarial
     - VQ commitment loss
     - Optional: NCE contrastive loss

3. **Checkpointing:**
   - Save every 100 epochs
   - Components: encoder, decoder, quantizer, quant_conv, post_quant_conv, net_g, net_d
   - Location: `{LOGS}/X2527T102MM/dist2/max10skip4/checkpoints/`

---

## Available Models

### 1. `ae0iso0tccutvqq` (Active)
- VQGAN with vector quantization
- 3D encoder-decoder + 3D generator
- VQ codebook learning

### 2. `ae0iso0tccut`
- VQGAN with KL-divergence (VAE-like)
- DiagonalGaussian posterior sampling
- Optional CUT contrastive loss

### 3. `ae0iso0tccutvqqssim`
- Same as `ae0iso0tccutvqq`
- Additional SSIM loss term

### 4. `IsoREF`
- Reference isotropic GAN
- Encoder-decoder architecture
- Cycle consistency option
- CUT contrastive learning

---

## Key Features & Design Decisions

### 1. Argument Parsing Strategy
- **JSON-driven configuration:** Most parameters in `env/jsn/*.json`
- **CLI overrides:** Command-line flags override JSON defaults
- **Model-specific args:** Added dynamically based on selected model
- **Abbreviation support:** Argparse allows `--l1` as shorthand for `--l1how`

### 2. Multi-View Discriminator
- 6 orthogonal views: (X,Y,Z) × (normal, flipped)
- Prevents mode collapse in 3D
- Used in all VQGAN models

### 3. L1 Projection Methods (`l1how`)
- `max`: Max pooling over Z-depth
- `min`: Min pooling over Z-depth
- `mean`: Average pooling
- `dsp`: Downsampling at fixed stride
- Applied to match generator output to ground truth slices

### 4. Contrastive Learning (CUT)
- Optional feature matching between encoder layers
- PatchNCE loss on 3D feature patches
- Can be disabled via `nocut: true`

### 5. Two-Channel Mode (`tc`)
- When enabled: concatenates two inputs
- Encoder processes 2-channel input → 1-channel output
- Currently disabled in active config

---

## Environment Setup

### Supported Environments (`env/env`)
- **brcb** (active): `/home/gary/workspace/`
- **a6k**: `/home/ubuntu/Data/`
- **runpod**: `/workspace/`
- **t09**: `/media/ExtHDD01/`
- **aisr**: `/media/ghc/Ghc_data3/BRC/aisr/`

Each environment defines:
- `LOGS`: Training logs and checkpoints
- `DATASET`: Root directory for datasets

---

## Recent Changes

### Oct 30, 2025 - Training Flow Refactor
1. **Argument parsing robustness:**
   - JSON loaded before model import
   - Model name resolved from JSON if not on CLI
   - Clearer error messages for missing configurations

2. **Configuration cleanup:**
   - Removed unused keys: `load3d`, `dataset_mode`, `flip`, `seed`, `mode`, `port`
   - Set `prj` to null (always override via CLI)

3. **Simplified run command:**
   - From: 200+ char command with all flags
   - To: `python train.py --jsn aisr --prj <name>`

4. **Documentation:**
   - Identified `--l1` as abbreviation for `--l1how`
   - Documented unused/overridden config keys
   - Created this present state summary

---

## Known Issues & Limitations

### Configuration
- `netG` and `final` in JSON are overridden by model-specific code
- Argparse abbreviations can cause confusion (e.g., `--l1` → `--l1how`)
- No explicit seeding for reproducibility

### Data
- Assumes TIFF or PIL-compatible images
- Requires exact filename matches across paired directories
- No built-in data validation

### Training
- No automatic mixed precision (AMP)
- No gradient accumulation exposed in config
- Limited to DDP strategy (no FSDP or DeepSpeed)

---

## Future Directions

### Suggested Improvements
1. **Config format:** Switch to YAML for inline comments
2. **Reproducibility:** Add global seeding (torch, numpy, random, pl)
3. **Validation:** Add dataset path validation before training
4. **Strict parsing:** Disable argparse abbreviations (`allow_abbrev=False`)
5. **Mixed precision:** Enable AMP for faster training
6. **Experiment tracking:** Integrate Weights & Biases or MLflow
7. **Documentation:** Add README with setup instructions and examples

### Experimental Variations
- Test different `l1how` methods (mean, min, dsp)
- Enable CUT loss (`nocut: false`, `lbNCE: 1.0`)
- Try two-channel mode (`tc: true`)
- Experiment with different YAML configs (ldmaex2, ldmaex2B)

---

## Dependencies

### Core Libraries (inferred)
- PyTorch (with CUDA)
- PyTorch Lightning
- Albumentations
- tifffile
- PIL/Pillow
- numpy
- tqdm
- tensorboard
- python-dotenv
- pyyaml

### Submodules/External
- `taming`: Taming Transformers (VQ-GAN components)
- `ldm`: Latent Diffusion Model modules

---

## Contact & Maintenance

**Project Path:** `/Users/garychang/Dropbox/TheSource/scripts/IsoScopeXX`  
**Active Config:** `env/jsn/aisr.json`  
**Active Model:** `ae0iso0tccutvqq`

---

*This document reflects the current state of the project as of the last update. For experiment-specific details, refer to the JSON configs in `env/jsn/` and model implementations in `models/`.*
