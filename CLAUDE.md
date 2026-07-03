# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Self-supervised **isotropic super-resolution + compression** for expansion microscopy. Input volumes are anisotropic (good X/Y resolution, poor Z). The model learns to synthesize a Z-isotropic, super-resolved volume **without any high-resolution ground truth**, while encoding the data into a compact VQ latent. Supervision comes from two self-consistency signals, not paired HR data:
- A **six-way (3-axis) PatchGAN discriminator** forces every orthogonal slice of the output to look like a real X/Y slice (enforcing isotropy).
- An **L1 projection loss** projects the high-Z output back down (max/mean/dsp pooling) and matches it to the observed low-Z input.

The anisotropy the model corrects is **created in-model at train time** by subsampling Z (`--dsp`, `--cropz`), so a single isotropic dataset is enough.

**Multi-view fused data** (`E2507218fuse`, `--direction zcube_xcube_ycube`) is a newer regime where the anisotropy is *real*, not faked: three registered optical-sectioning views of the same sample, each 8× low-res along a different axis (zcube→Z, xcube→X, ycube→Y). The HR-Z structure missing from zcube is present in xcube/ycube, enabling **real projection supervision** of the Z output. This is implemented in **`models/vqcleanM0aSup0.py`** (run with `--models vqcleanM0aSup0 --direction zcube_xcube_ycube --nm 11p --lamb_xy <w> --aniso 8`): `generation()` captures xcube/ycube at full Z (before the `dsp` subsample), and `backward_g` mean-pools the isotropic output `XupX` along X (dim3)→xcube and along Y (dim2)→ycube by `--aniso`, L1-matching the real views to supervise `XupX`'s Z with genuine HR structure (weighted by `--lamb_xy`; `--nm 11p` puts all views on a shared intensity scale). See `docs/fuse_data.md` for the axis mapping and the registration/photometry caveats.

## Running training

There is no build step and no test suite. The entry point is `train.py`, usually launched via copy-pasting a line from `run.sh` (most lines there are commented-out experiment history; the **last uncommented line is the current focused experiment**).

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NO_ALBUMENTATIONS_UPDATE=1 python train.py \
  --yaml aisr --models vqcleanM0a --prj <output/subdir> --env brcb \
  --dataset <dir-under-DATASET> --direction <subdir-under-train/>
```

- `NO_ALBUMENTATIONS_UPDATE=1` suppresses the albumentations version check.
- `CUDA_VISIBLE_DEVICES` selects GPUs; the trainer always uses **all visible GPUs with DDP** (`pl.Trainer(gpus=-1, strategy='ddp')`).
- `--prj` is **required** and names the output subtree; `--env` selects a machine profile from `cfg/env.json`.
- Any `train:` key in the YAML can be overridden by a matching CLI flag.

Monitoring (see `docs/paths.md` for exact path assembly):
```bash
tensorboard --logdir $LOGS/{dataset}/{prj}/logs/TensorBoardLogger
mlflow server --backend-store-uri sqlite:///$LOGS/mlflow.db \
  --artifacts-destination $LOGS/mlartifacts --host 0.0.0.0 --port 5002
```

## Configuration layering (read before changing any default)

A run's effective config is assembled in `train.py`'s `__main__` in a specific order — changing a value may require editing the right layer:

1. `cfg/{--yaml}.yaml` → its `train:` block is loaded as the **base namespace**.
2. `cfg/env.json[{--env}]` → supplies `DATASET`, `LOGS`, and optional `TRACKING_URI` (per machine).
3. The model named by `--models` (or `models:` in the YAML) is imported **dynamically**, and its `add_model_specific_args()` injects model-specific flags into the parser.
4. CLI flags override everything from steps 1–3 (`parser.parse_args(namespace=json_args)`).
5. `ldm/{--ldmyaml}.yaml` (e.g. `vqgan.yaml`) defines the **encoder/decoder/quantizer/loss network architecture** (`ddconfig`, `embed_dim`, `n_embed`, `lossconfig`), loaded inside the model's `__init__`.

So model **hyperparameters** live in `cfg/*.yaml`, but the **autoencoder network shape** lives in `ldm/*.yaml`. MLflow tracking URI priority: CLI `--tracking_uri` > `env.json` `TRACKING_URI` > local SQLite at `$LOGS/mlflow.db`. An `http://` URI must pass a `/health` check or training aborts.

Each run snapshots `config.json`, the active `cfg/*.yaml`, and the model's `.py` source into the timestamped checkpoint dir.

## Code architecture

**`models/base.py` → `BaseModel(pl.LightningModule)`** is the shared training scaffold. It owns the PL hooks: `training_step` (two optimizers — `optimizer_idx==1` discriminator, `==0` generator), checkpointing every `epoch_save`, the validation loop (LPIPS isotropy metric: XY-vs-YZ of prediction, plus KID), and MLflow GIF artifact logging. It defines the loss helpers (`add_loss_adv`, `add_loss_l1`) and `set_networks()`.

**Each file in `models/` is a concrete `GAN(BaseModel)`** that the training loop selects by `--models`. They differ in the autoencoder/quantization scheme but share the same self-supervised recipe. Each must implement `generation()`, `backward_g()`, `backward_d()`, and `add_model_specific_args()`. Lineage (oldest → current focus): `ae0iso0tccutvqq` (single VQ) → `vqclean` → **`vqcleanM0a`** (multi-scale residual VQ) → `vqcleanM0aSup0` (vqcleanM0a + real X/Y projection supervision for multi-view fused data) → experimental `flowV0`/`flowV1`/`flow1` (flow matching).

**The central method to understand is `generation()`** (e.g. `models/vqcleanM0a.py`). Per batch it:
1. Crops Z to `cropz`, then **subsamples Z by `dsp`** to synthesize the anisotropic input (and optionally upsamples by `usp`).
2. Reshapes the volume into a stack of 2D slices `(Z, C, X, Y)` and runs the **2D** VQ-encoder → quantizer → decoder.
3. Feeds the latent into the **3D** generator `net_g` to produce the isotropic volume `self.XupX`.
4. `backward_g` computes the six-way adversarial loss + the L1 self-consistency loss (`get_projection` pools `XupX` back down per `--l1how`, weighted by `--lamb`) + VQ codebook loss + LPIPS/adversarial from `lossconfig`.

`vqcleanM0a` specifically implements **VAR-style multi-scale residual quantization** in `encode()`: `--num_scales` coarse-to-fine passes, each quantizing the residual and subtracting its upsampled contribution; `--shared_codebook` toggles one codebook vs. per-scale. Key geometry flags: `dsp` (Z downsample), `usp` (Z upsample), `cropz`/`cropsize` (Z vs X/Y crop — equal = isotropic-cube training), `skipl1`/`l1how` (L1 projection), `downbranch`/`resizebranch` (latent Z reshaping into `net_g`). `uprate` is derived as `(cropsize//cropz)*dsp/usp`.

**`networks/`** builds the generators/discriminators. `networks/registry.py` is a factory keyed by the `--netG`/`--netD` strings (e.g. `ed023e`, `patch_16`); `BaseModel.set_networks()` resolves them. `networks/EncoderDecoder/` holds the 3D U-Net generators.

**`ldm/` and `taming/`** are vendored from Latent Diffusion / Taming Transformers, providing the 2D `Encoder`/`Decoder` (`ldm/modules/diffusionmodules/modelcut.py`), `VectorQuantizer2`, and `VQLPIPSWithDiscriminator`. Treat them as upstream libraries.

**`dataloader/data_multi.py` → `PairedImageDataset`** loads 3D TIFFs. `--direction` is split on `_` to address multiple paired sub-directories under `train/`/`val/` (e.g. `x3d0_x3d1`); files are paired by **shared filename** across those dirs. Albumentations applies precrop → resize → rotate → random crop. A missing `val/` directory is tolerated (training proceeds without validation).

## Conventions & gotchas

- **Volume tensor layout is `(B, C, Y, X, Z)`** (dim2=Y, dim3=X, dim4=Z). TIFFs are read as `(Z, Y, X)` and the loader moves the page/Z axis to **last** (`np.transpose(v,(1,2,0))` + `ToTensorV2` + `.permute(1,2,0)` in `dataloader/data_multi.py`), which is why `cropz`/`dsp` subsample dim4. Verified against `E2507218fuse` by per-axis gradient energy: zcube is low-res along dim4 (Z), xcube along dim3 (X), ycube along dim2 (Y) — matching `vqcleanM0aSup0`'s xcube→axis3 / ycube→axis2 projection. Code frequently `.permute(...)`s to pull a given axis to the front as a 2D batch; when editing model math, track which axis is the slice/batch dimension.
- Argparse **abbreviations are enabled**: `--l1` resolves to `--l1how`. Be explicit to avoid silent collisions.
- `accumulate_grad_batches=2` is hardcoded in `train.py`, so effective batch size is `2 × batch_size × n_gpus`.
- No global seeding — runs are not bit-reproducible.
- Targets PyTorch Lightning's **legacy API** (`pl.Trainer(gpus=-1)`, `training_step(..., optimizer_idx)`); do not "modernize" these without checking the installed PL version in `requirements.txt`.
- `cfg/env.json` keys are machine profiles (`brcb` is the primary dev box at `/home/gary/workspace/`). Add a new profile rather than editing paths when moving machines.
