"""
test_encode.py: Load VQGAN encoder components and encode 3D .tif files into quantized latents.

Each 3D volume is processed Z-slice by Z-slice (the encoder is 2D).
Outputs one .npz per input file with:
  - indices: (Z, H/16, W/16)  int64  codebook indices  [0–255]
  - quant:   (Z, 4, H/16, W/16) float32 quantized embeddings

Usage:
    python test_encode.py \
        --checkpoint_dir /path/to/checkpoints \
        --epoch 100 \
        --data_dir /path/to/tif_folder \
        --output_dir ./latents \
        --nm 01 \
        --cropsize 256 \
        --batch_size 8
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
        vol = tifffile.imread(path).astype(np.float32)  # load raw

        # --- resolve axis order to (Z, H, W) ---
        if vol.ndim == 2:
            # single 2D image: treat as Z=1
            vol = vol[np.newaxis]               # (1, H, W)
        elif vol.ndim == 3:
            # heuristic: smallest dim is Z unless it's clearly HW-dominant
            # common: (Z, H, W) where Z << H, W
            # also common: (H, W, Z) from ImageJ saves
            if vol.shape[2] < vol.shape[0] and vol.shape[2] < vol.shape[1]:
                vol = vol.transpose(2, 0, 1)    # (H, W, Z) → (Z, H, W)
            # else already (Z, H, W)
        elif vol.ndim == 4:
            # (Z, C, H, W) or (C, Z, H, W) — take first channel
            if vol.shape[1] <= 4:
                vol = vol[:, 0, :, :]           # (Z, C, H, W) → (Z, H, W)
            else:
                vol = vol[0]                    # (C, Z, H, W) → (Z, H, W) using C=0

        # --- normalize ---
        vol = self._normalize(vol, self.nm)

        # --- center-crop H and W ---
        vol = self._center_crop(vol, self.cropsize)  # (Z, H, W)

        # add channel dim → (Z, 1, H, W)
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
        return vol  # '00': no normalization

    @staticmethod
    def _center_crop(vol: np.ndarray, size: int) -> np.ndarray:
        """Center-crop the H and W dimensions of (Z, H, W)."""
        _, h, w = vol.shape
        if h < size or w < size:
            raise ValueError(
                f"Volume spatial size ({h}x{w}) is smaller than cropsize={size}"
            )
        top  = (h - size) // 2
        left = (w - size) // 2
        return vol[:, top:top + size, left:left + size]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_components(checkpoint_dir: str, epoch: int, device: torch.device):
    """Load encoder, quant_conv, and quantize from per-component .pth files."""
    def load(name):
        path = os.path.join(checkpoint_dir, f'{name}_model_epoch_{epoch}.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        obj = torch.load(path, map_location=device)
        # Saved as full nn.Module; move to device and set eval
        if isinstance(obj, nn.Module):
            obj = obj.to(device).eval()
        return obj

    encoder    = load('encoder')
    quant_conv = load('quant_conv')
    quantize   = load('quantize')
    return encoder, quant_conv, quantize


@torch.no_grad()
def encode_volume(vol: torch.Tensor, encoder, quant_conv, quantize,
                  device: torch.device, batch_size: int):
    """Encode a single 3D volume (Z, 1, H, W) → indices (Z, h, w), quant (Z, 4, h, w)."""
    Z = vol.shape[0]
    all_indices = []
    all_quant   = []

    for start in range(0, Z, batch_size):
        slices = vol[start:start + batch_size].to(device)  # (B, 1, H, W)

        h, _hbranch, _hs = encoder(slices)  # (B, z_channels, H/16, W/16)
        h = quant_conv(h)                   # (B, embed_dim,  H/16, W/16)
        quant, _emb_loss, info = quantize(h)

        B, C_emb, H_lat, W_lat = quant.shape

        # VectorQuantizer with sane_index_shape=False returns info[2] as flat (B*H*W,)
        indices = info[2].reshape(B, H_lat, W_lat)  # (B, H/16, W/16)

        # Re-derive quant directly from indices via the embedding table so the
        # saved values are guaranteed discrete codebook vectors, not a
        # straight-through mixture of z and the codebook.
        quant = (
            quantize.embedding(indices.reshape(-1))  # (B*H*W, embed_dim)
            .view(B, H_lat, W_lat, C_emb)            # (B, H, W, C)
            .permute(0, 3, 1, 2)                     # (B, C, H, W)
            .contiguous()
        )

        all_indices.append(indices.cpu().numpy())
        all_quant.append(quant.cpu().numpy())

    return (
        np.concatenate(all_indices, axis=0).astype(np.int64),   # (Z, h, w)
        np.concatenate(all_quant,   axis=0),                    # (Z, 4, h, w)
    )


def main():
    parser = argparse.ArgumentParser(description='Encode 3D .tif files with VQGAN')
    parser.add_argument('--root', default='',
                        help='Root directory prepended to --checkpoint_dir, --data_dir, --output_dir')
    parser.add_argument('--checkpoint_dir', required=True,
                        help='Directory containing per-component .pth files (relative to --root if set)')
    parser.add_argument('--epoch', type=int, required=True,
                        help='Epoch number N (files: encoder_model_epoch_N.pth, ...)')
    parser.add_argument('--data_dir', required=True,
                        help='Directory to search for .tif/.tiff files (relative to --root if set)')
    parser.add_argument('--output_dir', default='latents',
                        help='Output directory for .npz files (relative to --root if set, default: latents)')
    parser.add_argument('--nm', default='01', choices=['00', '01', '11'],
                        help='Normalization: 00=none, 01=[0,1], 11=[-1,1] (default: 01)')
    parser.add_argument('--cropsize', type=int, default=256,
                        help='Center-crop H and W to this size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of Z-slices per encoder forward pass (default: 8)')
    parser.add_argument('--device', default=None,
                        help='Device: cuda / cpu (default: cuda if available)')
    parser.add_argument('--every_n', type=int, default=1,
                        help='Only encode every N-th .tif file (default: 1 = all). '
                             'Noise is still injected every 4th of the selected files.')
    args = parser.parse_args()

    # --- apply root ---
    if args.root:
        args.checkpoint_dir = os.path.join(args.root, args.checkpoint_dir)
        args.data_dir       = os.path.join(args.root, args.data_dir)
        args.output_dir     = os.path.join(args.root, args.output_dir)

    # --- device ---
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- load model components ---
    print(f"Loading encoder components from {args.checkpoint_dir} (epoch {args.epoch}) ...")
    encoder, quant_conv, quantize = load_components(args.checkpoint_dir, args.epoch, device)
    print("  encoder    OK")
    print("  quant_conv OK")
    print("  quantize   OK")

    # --- dataset / dataloader ---
    dataset = SimpleTifDataset(args.data_dir, cropsize=args.cropsize, nm=args.nm)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                         collate_fn=lambda x: x[0])  # returns (tensor, stem) directly
    print(f"Found {len(dataset)} .tif file(s) in {args.data_dir}")

    # --- output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- encode loop ---
    all_unique_codes = []
    selected = 0  # count of files actually encoded (drives the noise schedule)
    for i, (vol, stem) in enumerate(loader):
        if i % args.every_n != 0:
            continue

        Z, _, H, W = vol.shape

        # every 4th *selected* file gets heavy Gaussian noise
        altered = (selected % 4 == 3)
        if altered:
            noise = torch.randn_like(vol) * 0.3
            vol   = (vol + noise).clamp(0.0, 1.0)
            print(f"[{i+1}/{len(dataset)}] #{selected+1} {stem}.tif  →  (Z={Z}, H={H}, W={W})  [NOISY]")
        else:
            print(f"[{i+1}/{len(dataset)}] #{selected+1} {stem}.tif  →  (Z={Z}, H={H}, W={W})")
        selected += 1

        indices, quant = encode_volume(vol, encoder, quant_conv, quantize,
                                       device, args.batch_size)

        out_path = os.path.join(args.output_dir, f'{stem}.npz')
        np.savez(out_path, indices=indices, quant=quant, altered=altered)

        unique = np.unique(indices)
        all_unique_codes.extend(unique.tolist())
        print(f"         saved → {out_path}")
        print(f"         indices {indices.shape}, quant {quant.shape}, "
              f"unique codes: {len(unique)}/256")

    # --- summary ---
    total_unique = len(set(all_unique_codes))
    print(f"\nDone. {selected}/{len(dataset)} file(s) encoded (every_n={args.every_n}).")
    print(f"Codebook utilization across all files: {total_unique}/256 codes used.")


def plot_tsne(latent_dir: str, output_path: str = 'tsne.png',
              pca_components: int = 50, perplexity: float = None,
              random_state: int = 42):
    """Load all .npz latent files; each file is ONE data point for t-SNE.

    Feature vector per file: quant (Z, 4, h, w) flattened → (Z*4*h*w,)
    PCA reduces to pca_components dims first, then t-SNE → 2D.
    Each point is labeled with the filename stem.

    Args:
        latent_dir:     Directory containing .npz files.
        output_path:    Where to save the t-SNE scatter plot (.png).
        pca_components: PCA dims before t-SNE (set 0 to skip PCA).
        perplexity:     t-SNE perplexity (defaults to max(5, n_files//3)).
        random_state:   Random seed for reproducibility.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    npz_files = sorted(glob(os.path.join(latent_dir, '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {latent_dir}")
    n_files = len(npz_files)
    print(f"Loading {n_files} .npz file(s) from {latent_dir} ...")

    features  = []   # one flattened vector per file
    stems     = []   # filename labels
    altered_mask = []  # True if this file had noise injected

    for path in npz_files:
        d     = np.load(path)
        quant = d['quant']          # (Z, 4, h, w)
        features.append(quant.flatten().astype(np.float32))
        stems.append(Path(path).stem)
        altered_mask.append(bool(d['altered']) if 'altered' in d else False)
        Z, C, h, w = quant.shape
        tag = '  [NOISY]' if altered_mask[-1] else ''
        print(f"  {Path(path).name}: quant {quant.shape} → {quant.size} dims{tag}")

    X = np.stack(features)  # (n_files, Z*4*h*w)
    print(f"Feature matrix: {X.shape}")

    # --- PCA ---
    n_pca = min(pca_components, n_files - 1) if pca_components > 0 else 0
    if n_pca > 0:
        print(f"PCA: {X.shape[1]} → {n_pca} dims ...")
        X = PCA(n_components=n_pca, random_state=random_state).fit_transform(X)

    # --- t-SNE ---
    perp = perplexity if perplexity is not None else max(5.0, n_files / 3.0)
    perp = min(perp, n_files - 1)  # perplexity must be < n_samples
    print(f"t-SNE: {X.shape} → 2D  (perplexity={perp:.1f}) ...")
    emb = TSNE(n_components=2, perplexity=perp, random_state=random_state,
               n_jobs=-1).fit_transform(X)  # (n_files, 2)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap   = cm.get_cmap('tab20', n_files)
    colors = [cmap(i) for i in range(n_files)]

    for i, (x, y) in enumerate(emb):
        noisy  = altered_mask[i]
        marker = 'x' if noisy else 'o'
        ms     = 120  if noisy else 80
        lw     = 2.5  if noisy else 1
        ax.scatter(x, y, color=colors[i], marker=marker, s=ms,
                   linewidths=lw, zorder=3)
        label = f"{stems[i]} [noisy]" if noisy else stems[i]
        ax.annotate(label, (x, y), textcoords='offset points',
                    xytext=(6, 4), fontsize=7, color=colors[i])

    ax.set_title(f't-SNE of {n_files} latent volumes  (each point = one .npz file)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"t-SNE plot saved to {output_path}")
    plt.close()

    return emb, stems, altered_mask   # (n_files, 2), list[str], list[bool]


def sort_by_tsne_dist(emb: np.ndarray, stems: list, data_dir: str,
                      output_dir: str):
    """Copy .tif files sorted by distance from the t-SNE centroid.

    Files closest to the centroid get the lowest index prefix (most
    representative first).  Each file is copied as:
        {output_dir}/{rank:04d}_{original_stem}.tif

    Args:
        emb:        (n_files, 2) t-SNE coordinates returned by plot_tsne.
        stems:      List of filename stems (same order as emb rows).
        data_dir:   Directory where the original .tif files live (searched
                    recursively by stem name).
        output_dir: Destination directory for sorted copies.
    """
    import shutil

    # --- build stem → source path map (recursive search) ---
    all_tifs = (
        glob(os.path.join(data_dir, '**', '*.tif'),  recursive=True) +
        glob(os.path.join(data_dir, '**', '*.tiff'), recursive=True)
    )
    stem_to_path = {Path(p).stem: p for p in all_tifs}

    missing = [s for s in stems if s not in stem_to_path]
    if missing:
        raise FileNotFoundError(
            f"Could not find .tif source for stems: {missing}\n"
            f"Searched under: {data_dir}"
        )

    # --- compute distances to centroid ---
    center = emb.mean(axis=0)                           # (2,)
    dists  = np.linalg.norm(emb - center, axis=1)      # (n_files,)
    order  = np.argsort(dists)                          # closest first

    os.makedirs(output_dir, exist_ok=True)
    n_pad = len(str(len(stems)))  # zero-pad width

    print(f"\nSorting {len(stems)} files by t-SNE distance → {output_dir}")
    print(f"{'Rank':>6}  {'Dist':>8}  File")
    for rank, idx in enumerate(order):
        stem = stems[idx]
        src  = stem_to_path[stem]
        ext  = Path(src).suffix
        dst  = os.path.join(output_dir, f"{rank:0{n_pad}d}_{stem}{ext}")
        shutil.copy2(src, dst)
        print(f"  {rank:>{n_pad}d}  {dists[idx]:8.3f}  {stem}{ext}")

    print(f"\nDone. {len(stems)} file(s) copied to {output_dir}")


if __name__ == '__main__':
    main()
    # python test_encode.py --root /home/gary/workspace/Data/THX10SDM20xw --checkpoint_dir /home/gary/workspace/logs/THX10SDM20xw/roiAdsp4/max5skip4/checkpoints  --data_dir roiAdsp4/  --output_dir latent/  --nm 01 --epoch 1000 --every_n 1

    emb, stems, altered_mask = plot_tsne('/home/gary/workspace/Data/THX10SDM20xw/latent')

    sort_by_tsne_dist(                                                                                                                                                                                  
      emb, stems,                                           
      data_dir   = '/home/gary/workspace/Data/THX10SDM20xw/train',                                                                                                                                    
      output_dir = '/home/gary/workspace/Data/THX10SDM20xw/sorted',) 

      
