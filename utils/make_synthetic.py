import tifffile as tiff
import numpy as np


def add_noise(image, gaussian_weight=0.1, poisson_weight=0.1):
    """Add controllable Gaussian and Poisson noise to float32 image"""

    img_float = image.astype(np.float32)

    # Gaussian noise
    if gaussian_weight > 0:
        gaussian_noise = np.random.normal(0, gaussian_weight * img_float.std(), img_float.shape)
        img_float += gaussian_noise

    # Poisson noise (shot noise) - direct on float32 values
    if poisson_weight > 0:
        # Ensure non-negative for Poisson
        img_positive = np.clip(img_float, 0, None)
        # Apply Poisson noise directly to float32 values
        poisson_noise = np.random.poisson(img_positive * poisson_weight) / poisson_weight - img_positive
        img_float += poisson_noise

    return img_float.astype(image.dtype)


# Load and process your data
#x = tiff.imread('/media/ExtHDD01/Dataset/paired_images/bloodvessel/fortrain.tif')
#xx = x[:100, :, :]

# Add noise with controllable weights
xx_noisy = add_noise(xx, gaussian_weight=0.1, poisson_weight=0.00)

# Compare
print(f"Original shape: {xx.shape}")
print(f"Original range: {xx.min():.3f} - {xx.max():.3f}")
print(f"Noisy range: {xx_noisy.min():.3f} - {xx_noisy.max():.3f}")

# Save
xx = 1 * x
xx[xx <= -0.85] = -0.85
xx = (xx - xx.min()) / (xx.max() - xx.min())
xx = (2 * xx - 1).astype(np.float32)  # Scale to [-1, 1]
tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/bloodvessel/sample.tif', xx)