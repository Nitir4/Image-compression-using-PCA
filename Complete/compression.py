"""Numerical PCA/SVD helpers, independent of the desktop interface."""
import math
import numbers

import numpy as np


def validate_k(k):
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or k < 1:
        raise ValueError('k must be a positive integer.')
    return int(k)


def _channel(channel, k):
    validate_k(k)
    values = np.asarray(channel, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError('Image channels must be nonempty, finite 2D arrays.')
    return values


def custom_pca(channel, k):
    values = _channel(channel, k)
    mean = values.mean(axis=0)
    centered = values - mean
    # Economy SVD yields the covariance eigenvectors without forming a width²
    # matrix. It also handles single-column and single-row images directly.
    _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
    retained = vectors[:min(k, len(vectors))]
    reconstruction = (centered @ retained.T) @ retained + mean
    eigenvalues = singular_values ** 2 / max(values.shape[0] - 1, 1)
    return np.rint(np.clip(reconstruction, 0, 255)).astype(np.uint8), eigenvalues


def custom_svd(channel, k):
    values = _channel(channel, k)
    u, spectrum, vt = np.linalg.svd(values, full_matrices=False)
    rank = min(k, len(spectrum))
    reconstruction = (u[:, :rank] * spectrum[:rank]) @ vt[:rank]
    return np.rint(np.clip(reconstruction, 0, 255)).astype(np.uint8), spectrum


def compress_image_full(image, k, method):
    validate_k(k)
    image = np.asarray(image)
    if image.ndim == 2:
        channels = [image]
    elif image.ndim == 3 and image.shape[2] == 3:
        channels = [image[..., i] for i in range(3)]
    else:
        raise ValueError('Choose a grayscale or three-channel image.')
    if method not in ('PCA', 'SVD'):
        raise ValueError('Method must be PCA or SVD.')
    reconstructed, ratios = [], []
    for channel in channels:
        result, spectrum = (custom_pca if method == 'PCA' else custom_svd)(channel, k)
        energy = spectrum if method == 'PCA' else spectrum ** 2
        total = energy.sum()
        ratios.append(float(energy[:k].sum() / total) if total > 0 else 1.0)
        reconstructed.append(result)
    output = reconstructed[0] if image.ndim == 2 else np.stack(reconstructed, axis=2)
    return output, float(np.mean(ratios))


def mse_metric(a, b):
    return float(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2))


def psnr_metric(a, b):
    mse = mse_metric(a, b)
    return float('inf') if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))
