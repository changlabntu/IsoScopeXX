"""Fused-lasso (total-variation) graph solve — one prior for every corruption.

Drop-in alternative to affine.solve_graph for turning pairwise measurements
into an absolute per-slice chain. solve_graph's identity anchor assumes the
smallest overall trajectory; that is right for smooth drift but has no notion
of sparse events, and per-slice freedom lets estimator bias leak into slices
that never moved. In practice we do NOT know whether a stack suffers smooth
drift, a chunk-boundary tear, a random walk, or a mixture — so instead of a
per-model solver, this one puts an L1 (group fused-lasso) penalty on the
slice-to-slice INCREMENTS of the chain:

- increments smaller than the evidence supports collapse to ZERO (bias and
  jitter are not propagated into untouched slices; small smooth drift yields
  a near-identity chain instead of hallucinated motion);
- genuinely large increments (chunk tears, walk steps) are kept sharp — L1
  charges a single big jump far less than L2 would.

Implemented as iteratively-reweighted least squares (IRLS): the same
linear data rows as affine.solve_graph (affine._data_rows, with the linear
entries rescaled by px_scale into px-equivalents — 1% stretch or 0.57 deg
~ px_scale/100 px — so one lambda governs all components), plus per-increment
difference rows whose weight is 1/sqrt(||increment|| + eps), re-solved a few
times. A small identity anchor keeps the still-unobservable global modes
damped.
"""

import numpy as np

from registration import affine


def solve_graph_tv(measurements, n, tv=4.0, tv_lin=1.0, anchor=0.01, iters=12,
                   px_scale=100.0, eps=0.05):
    """Pairwise measurements -> (n, 2, 3) absolute chain, gauge M_0 = I,
    group-fused-lasso prior on increments (see module docstring).

    The translation and linear (rot/scale) parts of each increment form
    SEPARATE penalty groups with separate weights: their L1 threshold must
    sit just above the respective measurement-noise floor (~1 px for
    translation, ~0.3 px-equivalent for rotation at px_scale=100), else real
    sub-threshold rotation steps get soft-thresholded away with the noise.

    tv      translation-increment penalty (px units; data rows ~unit weight)
    tv_lin  linear-increment penalty (px-equivalents via px_scale)
    anchor  small identity prior for the global unobservable modes
    iters   IRLS iterations
    eps     px floor inside the IRLS weight
    """
    ident = affine.identity().reshape(-1)
    T_ENT, L_ENT = (2, 5), (0, 1, 3, 4)

    data_rows, data_rhs = affine._data_rows(measurements, n, px_scale)
    anchor_rows, anchor_rhs = [], []
    if anchor > 0:
        for z in range(1, n):
            for e in range(6):
                # px-equivalent units throughout (matches the scaled data rows)
                w = anchor * (1.0 if e in T_ENT else px_scale)
                row = np.zeros(6 * (n - 1))
                row[6 * (z - 1) + e] = w
                anchor_rows.append(row)
                anchor_rhs.append(w * ident[e])

    # Warm start from the plain anchored solve: IRLS weights are computed
    # from the CURRENT increments, so an identity start pins small-magnitude
    # (linear) increments at zero — the anchor solution's increments already
    # carry the real steps, letting the L1 reweighting judge them fairly.
    x = affine.solve_graph(measurements, n, anchor=max(anchor, 0.01))[1:].reshape(-1)
    for _ in range(iters):
        chain = np.concatenate([ident[None], x.reshape(n - 1, 6)])
        inc = np.diff(chain, axis=0)                # (n-1, 6)
        w_t = 1.0 / np.sqrt(np.linalg.norm(inc[:, T_ENT], axis=1) + eps)
        w_l = 1.0 / np.sqrt(np.linalg.norm(inc[:, L_ENT] * px_scale, axis=1) + eps)

        tv_rows, tv_rhs = [], []
        for z in range(n - 1):                      # increment between z, z+1
            for e in range(6):
                w = (tv * w_t[z] if e in T_ENT
                     else tv_lin * w_l[z] * px_scale)
                row = np.zeros(6 * (n - 1))
                row[6 * z + e] = w                  # x_{z+1}
                b = 0.0
                if z == 0:
                    b = w * ident[e]                # minus fixed M_0
                else:
                    row[6 * (z - 1) + e] = -w       # minus x_z
                tv_rows.append(row)
                tv_rhs.append(b)

        A = np.stack(data_rows + anchor_rows + tv_rows)
        b = np.array(data_rhs + anchor_rhs + tv_rhs)
        x = np.linalg.lstsq(A, b, rcond=None)[0]

    return np.concatenate([ident[None], x.reshape(n - 1, 6)]).reshape(n, 2, 3)
