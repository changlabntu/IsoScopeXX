"""M4a — shared overlay rendering (extracted from latent_overlay.py so both
the legacy patch overlays and the new per-cell mosaics draw through one path).

`draw_boxes` takes explicit pixel origins + per-box RGB, so any producer
(patch-level PCs, cell-level UMAP embeddings, anchor-similarity colors) can
render without re-implementing the matplotlib plumbing.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402


def chunk_figure(sl, ds_target=3000, fig_width=18):
    """Downsample a raw (y, x) slice and show it in gray.

    Returns (fig, ax, ds); callers pass pixel coordinates divided by ds.
    """
    ds = max(1, max(sl.shape) // ds_target)
    sld = sl[::ds, ::ds]
    vlo, vhi = np.percentile(sld, 1), np.percentile(sld, 99.5)
    fig, ax = plt.subplots(
        figsize=(fig_width, fig_width * sld.shape[0] / sld.shape[1]))
    ax.imshow(sld, cmap='gray', vmin=vlo, vmax=vhi, interpolation='nearest')
    ax.set_xlim(0, sld.shape[1]), ax.set_ylim(sld.shape[0], 0)
    ax.set_xticks([]), ax.set_yticks([])
    return fig, ax, ds


def draw_boxes(ax, x_px, y_px, size_px, rgbs, ds=1, alpha=0.55):
    """Filled squares at (x_px, y_px) top-left corners (raw-pixel frame),
    colored by rgbs (N, 3); ds = the figure's downsample factor."""
    s = size_px / ds
    for x, y, rgb in zip(np.asarray(x_px) / ds, np.asarray(y_px) / ds,
                         np.asarray(rgbs)):
        ax.add_patch(Rectangle((x, y), s, s, facecolor=rgb, alpha=alpha,
                               edgecolor='none', zorder=2))


def close(fig):
    plt.close(fig)
