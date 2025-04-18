"""Float-valued PCA for the legacy normalized RGB interface."""
import numbers
import numpy as np

def PCA(matrix, k):
    values = np.asarray(matrix, dtype=float)
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or k < 1:
        raise ValueError('k must be a positive integer.')
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty finite image channel.')
    mean = values.mean(axis=0)
    centered = values - mean
    _, spectrum, vt = np.linalg.svd(centered, full_matrices=False)
    retained = vt[:min(k, len(vt))]
    eigenvalues = spectrum ** 2 / max(len(values) - 1, 1)
    return (centered @ retained.T) @ retained + mean, eigenvalues, np.arange(len(eigenvalues))
