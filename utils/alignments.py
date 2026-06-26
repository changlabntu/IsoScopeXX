import numpy as np
from scipy.ndimage import shift
from scipy.signal import correlate
from tqdm.auto import tqdm


def align_stack_xy(x: np.ndarray, reference='first', subpixel=False, fill_value=0.0):
    """
    Align a (Z, X, Y) stack in-plane (X,Y) by cross-correlating each slice to a reference.

    Parameters
    ----------
    x : np.ndarray
        Input volume of shape (Z, X, Y).
    reference : {'first', 'mean', np.ndarray}
        Alignment reference. 'first' = x[0], 'mean' = mean of all slices.
        You may also pass an explicit 2D array of shape (X, Y).
    subpixel : bool
        If True, do a simple quadratic peak refinement for ~subpixel shift.
        (Lightweight parabola fit around the integer peak.)
    fill_value : float
        Value used to fill edges introduced by shifting.

    Returns
    -------
    x_aligned : np.ndarray
        Aligned volume, same shape as x.
    shifts : np.ndarray
        Array of per-slice shifts with shape (Z, 2): (dy, dx) applied to each slice.
        Note: shift() uses order (shift_along_axis0, shift_along_axis1) = (dy, dx).
    """
    assert x.ndim == 3, "Expected (Z, X, Y) array."
    Z, X, Y = x.shape

    # choose reference
    if isinstance(reference, np.ndarray):
        ref = reference
        assert ref.shape == (X, Y), "Custom reference must be (X, Y)."
    elif reference == 'first':
        ref = x[0].astype(np.float32, copy=False)
    elif reference == 'mean':
        ref = x.astype(np.float32).mean(axis=0)
    else:
        raise ValueError("reference must be 'first', 'mean', or a 2D array")

    ref = np.nan_to_num(ref, copy=False)

    x_aligned = np.empty_like(x, dtype=x.dtype)
    shifts = np.zeros((Z, 2), dtype=np.float32)  # (dy, dx)

    # helper for local quadratic refinement (optional)
    def _quadratic_subpixel_offset(corr, peak_y, peak_x):
        # Fit a 1D parabola around the peak in y and x independently.
        dy = 0.0
        dx = 0.0
        if 1 <= peak_y < corr.shape[0] - 1:
            a = corr[peak_y - 1, peak_x]
            b = corr[peak_y, peak_x]
            c = corr[peak_y + 1, peak_x]
            denom = (a - 2 * b + c)
            if denom != 0:
                dy = 0.5 * (a - c) / denom
        if 1 <= peak_x < corr.shape[1] - 1:
            a = corr[peak_y, peak_x - 1]
            b = corr[peak_y, peak_x]
            c = corr[peak_y, peak_x + 1]
            denom = (a - 2 * b + c)
            if denom != 0:
                dx = 0.5 * (a - c) / denom
        return dy, dx

    # precompute center for translating correlation peak to shift
    center_y = X - 1
    center_x = Y - 1

    for z in tqdm(range(Z), desc="Aligning slices"):
        img = np.nan_to_num(x[z].astype(np.float32, copy=False), copy=False)

        # full cross-correlation (2D). correlate flips the second argument internally for correlation.
        # mode='full' gives a (2X-1, 2Y-1) map where the peak gives the best alignment.
        corr = correlate(img, ref, mode='full', method='auto')

        # integer-precision peak
        peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)

        # convert peak location to shift (dy, dx) to apply to img so it matches ref
        # For arrays of size (X,Y), the zero-shift peak would be at (X-1, Y-1).
        dy = peak_y - center_y
        dx = peak_x - center_x

        # optional subpixel refinement with a simple parabola around the peak
        if subpixel:
            sub_dy, sub_dx = _quadratic_subpixel_offset(corr, peak_y, peak_x)
            dy = dy + sub_dy
            dx = dx + sub_dx

        # Apply shift: scipy.ndimage.shift expects (shift_y, shift_x)
        aligned = shift(
            img,
            shift=(dy, dx),
            order=1,  # bilinear; change to 0 for nearest if desired
            mode='constant',
            cval=fill_value,
            prefilter=True
        )

        x_aligned[z] = aligned.astype(x.dtype, copy=False)
        shifts[z] = (dy, dx)

    return x_aligned, shifts


# Example usage:
if __name__ == "__main__":
    import tifffile as tiff

    # Load your actual TIFF data
    x = tiff.imread('/media/ghc/GHc_data2/BRC/iUExM/roiA.tif')[:, ::]

    # x is your (Z, X, Y) numpy array
    x_aligned, per_slice_shifts = align_stack_xy(x, reference='first', subpixel=True)
    print(per_slice_shifts[:5])  # (dy, dx) for the first 5 slices

    # Optional: Save the aligned result
    tiff.imwrite('temp.tif', x_aligned)