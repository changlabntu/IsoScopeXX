"""2x3 affine utilities shared by the registration experiment.

Conventions (used consistently across registration/):
- A transform is a 2x3 matrix M = [A | t] acting on CENTERED pixel coordinates
  p = (x, y) (x along width / numpy axis -1, y along height / axis -2):
  p' = A @ p + t. "Applying M to a slice" moves the CONTENT by M (push-forward):
  out(p) = in(M^-1 p).
- Because coordinates are centered and normalized proportionally, the same
  rotation/scale applies at any resolution; only the translation scales with
  resolution (t_latent = t_pixel / 8 for this model's 8x-downsampled latent).
- Similarity parameterization: (rot_deg, tx, ty, scale) with
  A = scale * R(rot), R = [[cos, -sin], [sin, cos]].
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- numpy side

def params_to_mat(rot_deg, tx, ty, scale=1.0):
    """(rot_deg, tx, ty, scale) -> 2x3 forward matrix."""
    r = np.deg2rad(rot_deg)
    c, s = np.cos(r) * scale, np.sin(r) * scale
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float64)


def mat_to_params(m):
    """2x3 similarity matrix -> (rot_deg, tx, ty, scale). For a pure similarity
    this is exact; for a general affine it is the best similarity read-out."""
    a, b = m[0, 0], m[0, 1]
    c, d = m[1, 0], m[1, 1]
    rot = np.rad2deg(np.arctan2(c - b, a + d))
    scale = np.sqrt(max(a * d - b * c, 0.0))
    return float(rot), float(m[0, 2]), float(m[1, 2]), float(scale)


def identity():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def compose(m2, m1):
    """Apply m1 first, then m2: (m2 o m1)."""
    a = m2[:, :2] @ m1[:, :2]
    t = m2[:, :2] @ m1[:, 2] + m2[:, 2]
    return np.concatenate([a, t[:, None]], axis=1)


def invert(m):
    ai = np.linalg.inv(m[:, :2])
    return np.concatenate([ai, (-ai @ m[:, 2])[:, None]], axis=1)


# ---------------------------------------------------------------- torch warp

def _theta_from_pullback(a00, a01, a10, a11, tx, ty, H, W):
    """Pull-back matrix (pixel, centered) -> F.affine_grid theta (normalized,
    align_corners=False). Works on torch scalars/1D tensors (differentiable)."""
    sx, sy = W / 2.0, H / 2.0
    row0 = torch.stack([a00, a01 * (sy / sx), tx / sx], dim=-1)
    row1 = torch.stack([a10 * (sx / sy), a11, ty / sy], dim=-1)
    return torch.stack([row0, row1], dim=-2)   # (..., 2, 3)


def mats_to_theta(mats, H, W):
    """(N, 2, 3) FORWARD transforms (torch or numpy, pixel units) -> theta for
    affine_grid (theta encodes the pull-back = inverse)."""
    if not torch.is_tensor(mats):
        mats = torch.as_tensor(np.asarray(mats), dtype=torch.float32)
    A = mats[:, :, :2]
    t = mats[:, :, 2]
    Ai = torch.inverse(A)
    ti = -(Ai @ t[:, :, None])[:, :, 0]
    return _theta_from_pullback(Ai[:, 0, 0], Ai[:, 0, 1], Ai[:, 1, 0], Ai[:, 1, 1],
                                ti[:, 0], ti[:, 1], H, W)


def theta_from_params(rot_rad, tx, ty, log_s, H, W):
    """Differentiable theta for FORWARD similarity params (1D tensors, pixel
    units). The pull-back is built analytically: A^-1 = (1/s) R(-rot),
    t^-1 = -A^-1 t — no matrix inverses, safe for autograd."""
    inv_s = torch.exp(-log_s)
    c, s = torch.cos(rot_rad) * inv_s, torch.sin(rot_rad) * inv_s
    # pull-back A = [[c, s], [-s, c]]
    tix = -(c * tx + s * ty)
    tiy = -(-s * tx + c * ty)
    return _theta_from_pullback(c, s, -s, c, tix, tiy, H, W)


def warp_slices(imgs, mats_or_theta, mode='bilinear', fill=0.0, is_theta=False):
    """Warp (N, C, H, W) slices by per-slice FORWARD transforms.

    mats_or_theta: (N, 2, 3) forward matrices (pixel units) or, with
    is_theta=True, a ready theta from theta_from_params (keeps gradients).
    Areas mapped from outside the input get `fill`.
    """
    N, C, H, W = imgs.shape
    theta = mats_or_theta if is_theta else mats_to_theta(mats_or_theta, H, W)
    grid = F.affine_grid(theta.to(imgs.device, imgs.dtype), (N, C, H, W),
                         align_corners=False)
    out = F.grid_sample(imgs - fill, grid, mode=mode, padding_mode='zeros',
                        align_corners=False)
    return out + fill


# ------------------------------------------------------------- global solves

def solve_graph(measurements, n, anchor=0.0, px_scale=100.0):
    """Least-squares absolute transforms from pairwise measurements.

    measurements: list of (i, j, D) with D (2x3) the measured forward
    transform aligning slice i's content onto slice j (D ~ M_j o M_i^-1, so
    M_j = D o M_i). Unknowns are M_1..M_{n-1} (6 entries each); the gauge is
    fixed at M_0 = I. Linear because M_j - A_D M_i = t_D is linear in the
    entries of M_i, M_j. Returns (n, 2, 3) absolute matrices.

    anchor > 0 adds a weak minimum-norm prior pulling every M_z toward
    identity. Pairwise measurements only weakly constrain the smooth
    low-frequency modes of the chain, so real structural drift through Z leaks
    into them as spurious coherent motion (mostly translation); the anchor
    picks the smallest trajectory consistent with the measurements. `anchor`
    weights the translation entries directly (measurement rows are ~unit
    weight in px); linear (rot/scale) entries get anchor/px_scale, i.e.
    px_scale px of shift is treated like 100% of stretch.
    """
    def cols(z):     # column indices of M_z's 6 unknowns [a00 a01 tx a10 a11 ty]
        return None if z == 0 else slice(6 * (z - 1), 6 * z)

    ident = identity().reshape(-1)                  # M_0 entries
    rows, rhs = [], []
    for i, j, D in measurements:
        A_D, t_D = np.asarray(D)[:, :2], np.asarray(D)[:, 2]
        # 6 equations: entry (r, c) of M_j equals sum_k A_D[r,k] M_i[k,c] (+ t_D[r] for c=2)
        for r in range(2):
            for c in range(3):
                row = np.zeros(6 * (n - 1))
                b = t_D[r] if c == 2 else 0.0
                if j == 0:
                    b -= ident[3 * r + c]
                else:
                    row[6 * (j - 1) + 3 * r + c] = 1.0
                for k in range(2):
                    if i == 0:
                        b += A_D[r, k] * ident[3 * k + c]
                    else:
                        row[6 * (i - 1) + 3 * k + c] -= A_D[r, k]
                rows.append(row)
                rhs.append(b)
    if anchor > 0:
        for z in range(1, n):
            for e in range(6):
                w = anchor * (1.0 if e in (2, 5) else 1.0 / px_scale)
                row = np.zeros(6 * (n - 1))
                row[6 * (z - 1) + e] = w
                rows.append(row)
                rhs.append(w * ident[e])
    x = np.linalg.lstsq(np.stack(rows), np.array(rhs), rcond=None)[0]
    mats = np.concatenate([ident[None], x.reshape(n - 1, 6)]).reshape(n, 2, 3)
    return mats


def fit_gauge(pred, gt):
    """Fit the single global transform G minimizing sum_z ||gt_z - pred_z o G||^2
    (the unobservable gauge of a pairwise-only registration). Returns
    (G (2x3), gauged pred (n, 2, 3))."""
    pred, gt = np.asarray(pred), np.asarray(gt)
    rows, rhs = [], []
    for Mz, Gz in zip(pred, gt):
        A, t = Mz[:, :2], Mz[:, 2]
        # (pred o G)[r, c] = sum_k A[r,k] G[k,c]  (+ t[r] for c=2)
        for r in range(2):
            for c in range(3):
                row = np.zeros(6)
                for k in range(2):
                    row[3 * k + c] = A[r, k]
                rows.append(row)
                rhs.append(Gz[r, c] - (t[r] if c == 2 else 0.0))
    G = np.linalg.lstsq(np.stack(rows), np.array(rhs), rcond=None)[0].reshape(2, 3)
    return G, np.stack([compose(M, G) for M in pred])
