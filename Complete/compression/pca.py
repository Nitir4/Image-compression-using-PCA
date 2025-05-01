import numpy as np

def custom_pca(channel: np.ndarray, k: int):
    mean_cols = channel.mean(axis=0)
    centered = channel.astype(np.float64) - mean_cols
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    k_eff = min(k, eigvecs.shape[1])
    V_k = eigvecs[:, :k_eff]
    proj = centered @ V_k
    recon = proj @ V_k.T
    recon += mean_cols
    return np.clip(recon, 0, 255).astype(np.uint8), eigvals
