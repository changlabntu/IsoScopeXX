"""Per-slice feature planes for registration: latent / codec / pixel.

All modes take a (Z, Y, X) float32 volume in [-1, 1] and return
(features (Z, C, Y/8, X/8) float32 numpy, stride=8):

- 'latent': continuous pre-VQ encoder output. The model's 2D encoder runs on
  each XY slice independently (models/MSclean.py vol_to_slices), so
  h = encoder(slice) is a per-slice (4, Y/8, X/8) plane. Applies the model's
  trained '11g' normalization first (Engine.normalize).
- 'codec': the quantized latent as rebuilt from stored codebook indices only
  (Engine.encode -> indices -> Engine.latents_from_indices, scales summed) —
  registration from the compressed representation alone. Optionally saves the
  indices npz (encode_stack scale_k key format) via codec_out.
- 'pixel': the raw [-1, 1] slices average-pooled 8x — the model-free baseline
  at the same spatial resolution (no engine needed).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

STRIDE = 8   # encoder downsampling (ldm/vqgan.yaml ch_mult [1,2,2,4])


def get_engine(model='skipU', epoch=None, device=None, half=False):
    from inference import Engine
    eng = Engine.from_registry(model, epoch=epoch, device=device)
    if half:
        eng.gan.half()
    return eng


def extract(vol, feat, eng=None, batch=8, half=False, codec_out=None):
    """(Z, Y, X) in [-1, 1] -> (features (Z, C, Y/8, X/8), stride)."""
    Z, Y, X = vol.shape
    if feat == 'pixel':
        t = torch.from_numpy(vol)[:, None]
        return F.avg_pool2d(t, STRIDE).numpy(), STRIDE

    assert eng is not None, f"feat '{feat}' needs a loaded Engine"
    dtype = torch.float16 if half else torch.float32
    normed = eng.normalize(vol).astype(np.float32)
    feats = []

    if feat == 'latent':
        with torch.inference_mode():
            for a in range(0, Z, batch):
                s = torch.from_numpy(normed[a:a + batch])[:, None]
                h = eng.gan.encoder(s.to(eng.device, dtype))[0]     # (b, 4, Y/8, X/8)
                feats.append(h.float().cpu().numpy())
        return np.concatenate(feats), STRIDE

    if feat == 'codec':
        all_idx = None
        with torch.inference_mode():
            for a in range(0, Z, batch):
                # (1, 1, Y, X, Zc) volume chunk — the encoder is per-slice, so
                # chunking over Z is exact
                x = torch.from_numpy(normed[a:a + batch]).permute(1, 2, 0)[None, None]
                _, indices = eng.encode(x.to(eng.device, dtype))    # (1, Zc, hk, wk) each
                lat = sum(eng.latents_from_indices(indices))        # (1, 4, Y/8, X/8, Zc)
                feats.append(lat[0].permute(3, 0, 1, 2).float().cpu().numpy())
                if all_idx is None:
                    all_idx = [[] for _ in indices]
                for k, idx in enumerate(indices):
                    all_idx[k].append(idx[0].cpu().numpy().astype(np.int32))
        if codec_out:
            np.savez_compressed(codec_out, **{f'scale_{k}': np.concatenate(v)
                                              for k, v in enumerate(all_idx)})
            print(f'codec indices -> {codec_out}')
        return np.concatenate(feats), STRIDE

    raise ValueError(f"unknown feat '{feat}'")
