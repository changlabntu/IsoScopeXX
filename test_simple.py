"""
test_simple.py: VQQ encode / decode / full pipeline for 3D .tif files.

--method encode : encode .tif files → root/latent/*.npz
--method decode : decode root/latent/*.npz → root/enhanced/*.tif
--method full   : encode + decode in one pass

Hardcoded sub-directories (under --root):
  latent/    — quantized latent .npz files
  enhanced/  — decoded output .tif files

Usage:
    python test_simple.py --method encode \\
        --root /home/gary/workspace/Data/THX10SDM20xw \\
        --checkpoint_dir /path/to/checkpoints \\
        --epoch 1000 --data_dir roiAdsp4/ --nm 01

    python test_simple.py --method decode \\
        --root /home/gary/workspace/Data/THX10SDM20xw \\
        --checkpoint_dir /path/to/checkpoints \\
        --epoch 1000 --downbranch 2

    python test_simple.py --method full \\
        --root /home/gary/workspace/Data/THX10SDM20xw \\
        --checkpoint_dir /path/to/checkpoints \\
        --epoch 1000 --data_dir roiAdsp4/ --nm 01 --downbranch 2
"""

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tifffile
from torch.utils.data import Dataset, DataLoader

LATENT_DIR   = 'latent'
ENHANCED_DIR = 'enhanced'


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SimpleTifDataset(Dataset):
    """Load 3D .tif volumes from a directory.

    Returns tensors of shape (Z, 1, H, W) after center-crop and normalization.
    Handles common axis orderings: (Z, H, W), (H, W, Z), (Z, C, H, W).
    """

    def __init__(self, data_dir: str, cropsize: int, nm: str):
        self.files = sorted(
            glob(os.path.join(data_dir, '**', '*.tif'),  recursive=True) +
            glob(os.path.join(data_dir, '**', '*.tiff'), recursive=True)
        )
        if not self.files:
            raise FileNotFoundError(f"No .tif/.tiff files found under {data_dir}")
        self.cropsize = cropsize
        self.nm = nm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        vol = tifffile.imread(path).astype(np.float32)

        if vol.ndim == 2:
            vol = vol[np.newaxis]
        elif vol.ndim == 3:
            if vol.shape[2] < vol.shape[0] and vol.shape[2] < vol.shape[1]:
                vol = vol.transpose(2, 0, 1)    # (H, W, Z) → (Z, H, W)
        elif vol.ndim == 4:
            if vol.shape[1] <= 4:
                vol = vol[:, 0, :, :]           # (Z, C, H, W) → (Z, H, W)
            else:
                vol = vol[0]                    # (C, Z, H, W) → (Z, H, W)

        vol = self._normalize(vol, self.nm)
        vol = self._center_crop(vol, self.cropsize)
        tensor = torch.from_numpy(vol[:, np.newaxis, :, :])  # (Z, 1, H, W)
        return tensor, Path(path).stem

    @staticmethod
    def _normalize(vol: np.ndarray, nm: str) -> np.ndarray:
        vmin, vmax = vol.min(), vol.max()
        if vmax == vmin:
            return vol
        if nm == '01':
            return (vol - vmin) / (vmax - vmin)
        if nm == '11':
            return 2.0 * (vol - vmin) / (vmax - vmin) - 1.0
        return vol

    @staticmethod
    def _center_crop(vol: np.ndarray, size: int) -> np.ndarray:
        _, h, w = vol.shape
        if h < size or w < size:
            raise ValueError(f"Volume spatial size ({h}x{w}) is smaller than cropsize={size}")
        top  = (h - size) // 2
        left = (w - size) // 2
        return vol[:, top:top + size, left:left + size]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_module(checkpoint_dir: str, name: str, epoch: int,
                 device: torch.device) -> nn.Module:
    path = os.path.join(checkpoint_dir, f'{name}_model_epoch_{epoch}.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(path, map_location=device)
    if isinstance(obj, nn.Module):
        obj = obj.to(device).eval()
    return obj


def load_encode_components(checkpoint_dir: str, epoch: int, device: torch.device):
    """Load encoder, quant_conv, quantize."""
    encoder    = _load_module(checkpoint_dir, 'encoder',    epoch, device)
    quant_conv = _load_module(checkpoint_dir, 'quant_conv', epoch, device)
    quantize   = _load_module(checkpoint_dir, 'quantize',   epoch, device)
    return encoder, quant_conv, quantize


def load_decode_components(checkpoint_dir: str, epoch: int, device: torch.device):
    """Load decoder and net_g."""
    decoder = _load_module(checkpoint_dir, 'decoder', epoch, device)
    net_g   = _load_module(checkpoint_dir, 'net_g',   epoch, device)
    return decoder, net_g


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_volume(vol: torch.Tensor, encoder, quant_conv, quantize,
                  device: torch.device, batch_size: int):
    """Encode (Z, 1, H, W) → indices (Z, h, w), quant (Z, 4, h, w)."""
    Z = vol.shape[0]
    all_indices, all_quant = [], []

    for start in range(0, Z, batch_size):
        slices = vol[start:start + batch_size].to(device)  # (B, 1, H, W)

        h, _hbranch, _hs = encoder(slices)   # (B, z_channels, H/16, W/16)
        h = quant_conv(h)                    # (B, embed_dim,  H/16, W/16)
        quant, _emb_loss, info = quantize(h)

        B, C_emb, H_lat, W_lat = quant.shape

        # info[2] is flat (B*H*W,) when sane_index_shape=False
        indices = info[2].reshape(B, H_lat, W_lat)

        # Re-derive quant from indices — guaranteed discrete codebook vectors
        quant = (
            quantize.embedding(indices.reshape(-1))  # (B*H*W, embed_dim)
            .view(B, H_lat, W_lat, C_emb)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        all_indices.append(indices.cpu().numpy())
        all_quant.append(quant.cpu().numpy())

    return (
        np.concatenate(all_indices, axis=0).astype(np.int64),  # (Z, h, w)
        np.concatenate(all_quant,   axis=0),                   # (Z, 4, h, w)
    )


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_volume(quant: np.ndarray, decoder, net_g,
                  downbranch: int, resizebranch: float,
                  device: torch.device) -> torch.Tensor:
    """Decode quantized latents (Z, 4, h, w) → enhanced volume (Z, C, H, W).

    Matches the generation() path in models/ae0iso0tccutvqq.py lines 256–270:
      quant → [downbranch] → [resizebranch] → decoder.conv_in
            → permute → net_g → output
    Note: post_quant_conv is intentionally skipped (not used in the net_g path).
    """
    h = torch.from_numpy(quant).to(device)  # (Z, 4, h, w)

    if downbranch > 1:
        h = h.permute(1, 2, 3, 0).unsqueeze(0)                      # (1, 4, h, w, Z)
        h = torch.nn.MaxPool3d((1, 1, downbranch))(h)               # (1, 4, h, w, Z//db)
        h = h.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]                # (Z//db, 4, h, w)

    if resizebranch != 1:
        h = h.permute(1, 2, 3, 0).unsqueeze(0)                      # (1, 4, h, w, Z)
        h = torch.nn.Upsample(scale_factor=(1, 1, resizebranch),
                               mode='trilinear')(h)                  # (1, 4, h, w, Z*rb)
        h = h.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]                # (Z*rb, 4, h, w)

    h = decoder.conv_in(h)                   # (Z, 256, h, w)
    h = h.permute(1, 2, 3, 0).unsqueeze(0)  # (1, 256, h, w, Z)

    out = net_g(h, method='decode')['out0']  # (1, C, X, Y, Z)
    out = out[0].permute(3, 0, 1, 2)        # (Z, C, X, Y)
    return out.cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='VQQ encode / decode / full pipeline')
        parser.add_argument('--method', required=True, choices=['encode', 'decode', 'full'],
                            help='Pipeline stage to run')
        parser.add_argument('--root', default='',
                            help='Root directory prepended to --checkpoint_dir and --data_dir')
        parser.add_argument('--checkpoint_dir', required=True,
                            help='Directory containing per-component .pth files (relative to --root)')
        parser.add_argument('--epoch', type=int, required=True,
                            help='Epoch number N (files: encoder_model_epoch_N.pth, ...)')
        parser.add_argument('--data_dir', default='',
                            help='Input .tif directory (relative to --root). Required for encode/full.')
        parser.add_argument('--nm', default='01', choices=['00', '01', '11'],
                            help='Normalization: 00=none, 01=[0,1], 11=[-1,1] (default: 01)')
        parser.add_argument('--cropsize', type=int, default=256,
                            help='Center-crop H×W to this size (default: 256)')
        parser.add_argument('--batch_size', type=int, default=8,
                            help='Z-slices per encoder forward pass (default: 8)')
        parser.add_argument('--downbranch', type=int, default=1,
                            help='Z MaxPool3d factor before net_g (default: 1 = off)')
        parser.add_argument('--resizebranch', type=float, default=1.0,
                            help='Z trilinear upsample factor before net_g (default: 1 = off)')
        parser.add_argument('--device', default=None,
                            help='Device: cuda / cpu (default: cuda if available)')
        parser.add_argument('--every_n', type=int, default=1,
                            help='Process every N-th file (default: 1 = all). '
                                'For encode/full: noise is injected every 4th selected file.')
        args = parser.parse_args()

    # --- apply root ---
    if args.root:
        args.checkpoint_dir = os.path.join(args.root, args.checkpoint_dir)
        if args.data_dir:
            args.data_dir = os.path.join(args.root, args.data_dir)

    latent_dir   = os.path.join(args.root, LATENT_DIR)   if args.root else LATENT_DIR
    enhanced_dir = os.path.join(args.root, ENHANCED_DIR) if args.root else ENHANCED_DIR

    # --- device ---
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    print(f"Device: {device}  |  method: {args.method}")

    # --- validate data_dir for encode/full ---
    if args.method in ('encode', 'full') and not args.data_dir:
        parser.error("--data_dir is required for --method encode/full")

    # --- load components ---
    encoder = quant_conv = quantize = decoder = net_g = None

    if args.method in ('encode', 'full'):
        print(f"Loading encode components (epoch {args.epoch}) ...")
        encoder, quant_conv, quantize = load_encode_components(
            args.checkpoint_dir, args.epoch, device)
        print("  encoder / quant_conv / quantize  OK")

    if args.method in ('decode', 'full'):
        print(f"Loading decode components (epoch {args.epoch}) ...")
        decoder, net_g = load_decode_components(args.checkpoint_dir, args.epoch, device)
        print("  decoder / net_g  OK")

    # -----------------------------------------------------------------------
    # ENCODE
    # -----------------------------------------------------------------------
    if args.method in ('encode', 'full'):
        os.makedirs(latent_dir, exist_ok=True)

        dataset = SimpleTifDataset(args.data_dir, cropsize=args.cropsize, nm=args.nm)
        loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=lambda x: x[0])
        print(f"Found {len(dataset)} .tif file(s) in {args.data_dir}")

        all_unique_codes = []
        selected = 0

        for i, (vol, stem) in enumerate(loader):
            if i % args.every_n != 0:
                continue

            Z, _, H, W = vol.shape
            altered = (selected % 4 == 3)
            if altered:
                noise = torch.randn_like(vol) * 0.2
                vol   = (vol + noise).clamp(-1.0, 1.0)

                #init = np.random.randint(0, 48)
                #vol[init:init+8, ::] = vol.min()

                print(f"[{i+1}/{len(dataset)}] #{selected+1} {stem}  (Z={Z} H={H} W={W})  [NOISY]")
            else:
                print(f"[{i+1}/{len(dataset)}] #{selected+1} {stem}  (Z={Z} H={H} W={W})")
            selected += 1

            indices, quant = encode_volume(vol, encoder, quant_conv, quantize,
                                           device, args.batch_size)

            npz_path = os.path.join(latent_dir, f'{stem}.npz')
            np.savez(npz_path, indices=indices, quant=quant, altered=altered)

            unique = np.unique(indices)
            all_unique_codes.extend(unique.tolist())
            print(f"  → latent  {npz_path}  indices {indices.shape}  "
                  f"unique codes {len(unique)}/256")

            # if full, immediately decode and save
            if args.method == 'full':
                os.makedirs(enhanced_dir, exist_ok=True)
                out = decode_volume(quant, decoder, net_g,
                                    args.downbranch, args.resizebranch, device)
                tif_path = os.path.join(enhanced_dir, f'{stem}.tif')
                _save_tif(out, tif_path)
                print(f"  → enhanced {tif_path}  shape {tuple(out.shape)}")

        total_unique = len(set(all_unique_codes))
        print(f"\nEncode done. {selected}/{len(dataset)} files  "
              f"(every_n={args.every_n})  "
              f"codebook utilization {total_unique}/256")

    # -----------------------------------------------------------------------
    # DECODE (standalone)
    # -----------------------------------------------------------------------
    if args.method == 'decode':
        os.makedirs(enhanced_dir, exist_ok=True)

        npz_files = sorted(glob(os.path.join(latent_dir, '*.npz')))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {latent_dir}")
        print(f"Found {len(npz_files)} .npz file(s) in {latent_dir}")

        selected = 0
        for i, npz_path in enumerate(npz_files):
            if i % args.every_n != 0:
                continue

            stem  = Path(npz_path).stem
            quant = np.load(npz_path)['quant']   # (Z, 4, h, w)
            print(f"[{i+1}/{len(npz_files)}] #{selected+1} {stem}  quant {quant.shape}")
            selected += 1

            out = decode_volume(quant, decoder, net_g,
                                args.downbranch, args.resizebranch, device)

            tif_path = os.path.join(enhanced_dir, f'{stem}.tif')
            _save_tif(out, tif_path)
            print(f"  → enhanced {tif_path}  shape {tuple(out.shape)}")

        print(f"\nDecode done. {selected}/{len(npz_files)} files decoded "
              f"(every_n={args.every_n})")


def _save_tif(vol: torch.Tensor, path: str):
    """Save (Z, C, H, W) or (Z, 1, H, W) tensor as a 3D .tif (Z, H, W)."""
    arr = vol.float().numpy()
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]   # (Z, H, W)
    tifffile.imwrite(path, arr)


# ---------------------------------------------------------------------------
# Analysis utilities (unchanged from test_encode.py)
# ---------------------------------------------------------------------------

def plot_tsne(latent_dir: str, output_path: str = 'tsne.png',
              pca_components: int = 50, perplexity: float = None,
              random_state: int = 42):
    """Load all .npz latent files; each file is ONE data point for t-SNE."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    npz_files = sorted(glob(os.path.join(latent_dir, '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {latent_dir}")
    n_files = len(npz_files)
    print(f"Loading {n_files} .npz file(s) from {latent_dir} ...")

    features     = []
    stems        = []
    altered_mask = []

    for path in npz_files:
        d     = np.load(path)
        quant = d['quant']
        features.append(quant.flatten().astype(np.float32))
        stems.append(Path(path).stem)
        altered_mask.append(bool(d['altered']) if 'altered' in d else False)
        tag = '  [NOISY]' if altered_mask[-1] else ''
        print(f"  {Path(path).name}: {quant.shape} → {quant.size} dims{tag}")

    X = np.stack(features)
    n_pca = min(pca_components, n_files - 1) if pca_components > 0 else 0
    if n_pca > 0:
        print(f"PCA: {X.shape[1]} → {n_pca} dims ...")
        X = PCA(n_components=n_pca, random_state=random_state).fit_transform(X)

    perp = perplexity if perplexity is not None else max(5.0, n_files / 3.0)
    perp = min(perp, n_files - 1)
    print(f"t-SNE: {X.shape} → 2D  (perplexity={perp:.1f}) ...")
    emb = TSNE(n_components=2, perplexity=perp, random_state=random_state,
               n_jobs=-1).fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap    = cm.get_cmap('tab20', n_files)
    colors  = [cmap(i) for i in range(n_files)]

    for i, (x, y) in enumerate(emb):
        noisy  = altered_mask[i]
        marker = 'x' if noisy else 'o'
        ax.scatter(x, y, color=colors[i], marker=marker,
                   s=120 if noisy else 80, linewidths=2.5 if noisy else 1, zorder=3)
        label = f"{stems[i]} [noisy]" if noisy else stems[i]
        ax.annotate(label, (x, y), textcoords='offset points',
                    xytext=(6, 4), fontsize=7, color=colors[i])

    ax.set_title(f't-SNE of {n_files} latent volumes  (each point = one .npz file)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(args.root + output_path, dpi=150, bbox_inches='tight')
    print(f"t-SNE plot saved to {args.root + output_path}")
    plt.close()

    return emb, stems, altered_mask


def sort_by_tsne_dist(emb: np.ndarray, stems: list, data_dir: str, output_dir: str):
    """Copy .tif files sorted by distance from the t-SNE centroid (closest first)."""
    import shutil

    all_tifs = (glob(os.path.join(data_dir, '**', '*.tif'),  recursive=True) +
                glob(os.path.join(data_dir, '**', '*.tiff'), recursive=True))
    stem_to_path = {Path(p).stem: p for p in all_tifs}

    missing = [s for s in stems if s not in stem_to_path]
    if missing:
        raise FileNotFoundError(f"Could not find .tif for stems: {missing}")

    center = emb.mean(axis=0)
    dists  = np.linalg.norm(emb - center, axis=1)
    order  = np.argsort(dists)
    os.makedirs(output_dir, exist_ok=True)
    n_pad  = len(str(len(stems)))

    print(f"\nSorting {len(stems)} files by t-SNE distance → {output_dir}")
    for rank, idx in enumerate(order):
        stem = stems[idx]
        src  = stem_to_path[stem]
        ext  = Path(src).suffix
        dst  = os.path.join(output_dir, f"{rank:0{n_pad}d}_{stem}{ext}")
        shutil.copy2(src, dst)
        print(f"  {rank:>{n_pad}d}  {dists[idx]:8.3f}  {stem}{ext}")

    print(f"\nDone. {len(stems)} file(s) copied to {output_dir}")


if __name__ == '__main__':
    case = 'mouse'
    if case == 'thx':
        import sys
        sys.argv = ['test_simple.py']  # clear argv so argparse doesn't complain
        args = argparse.Namespace(
            method='full',
            root='/home/gary/workspace/Data/THX10SDM20xw',
            checkpoint_dir='/home/gary/workspace/logs/THX10SDM20xw/roiAdsp4/max5skip4/checkpoints',
            data_dir='roiAdsp4/',
            nm='11',
            epoch=1000,
            downbranch=1,
            resizebranch=1.0,
            cropsize=256,
            batch_size=8,
            every_n=4,
            device=None,
        )
        main(args)
        emb, stems, altered_mask = plot_tsne('/home/gary/workspace/Data/THX10SDM20xw/latent')

    elif case == 'mouse':
        args = argparse.Namespace(
            method='full',
            root='/home/gary/workspace/Data/MouseGolgi/train/',
            checkpoint_dir='/home/gary/workspace/logs/MouseGolgi/roiAdsp4/max2skip8/checkpoints',
            data_dir='ME2509302/',
            nm='11',
            epoch=400,
            downbranch=1,
            resizebranch=1.0,
            cropsize=256,
            batch_size=8,
            every_n=10,
            device=None,
        )
        main(args)
        emb, stems, altered_mask = plot_tsne('/home/gary/workspace/Data/MouseGolgi/train/latent')
    # python test_simple.py --method encode --root /home/gary/workspace/Data/THX10SDM20xw --checkpoint_dir /home/gary/workspace/logs/THX10SDM20xw/roiAdsp4/max5skip4/checkpoints --data_dir roiAdsp4/ --nm 11 --epoch 1000
    # python test_simple.py --method decode --root /home/gary/workspace/Data/THX10SDM20xw --checkpoint_dir /home/gary/workspace/logs/THX10SDM20xw/roiAdsp4/max5skip4/checkpoints --epoch 1000 --downbranch 1
    # python test_simple.py --method full   --root /home/gary/workspace/Data/THX10SDM20xw --checkpoint_dir /home/gary/workspace/logs/THX10SDM20xw/roiAdsp4/max5skip4/checkpoints --data_dir roiAdsp4/ --nm 11 --epoch 1000 --downbranch 1 --every_n 4
