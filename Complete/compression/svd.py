import numpy as np

def custom_svd(channel: np.ndarray, k: int):
    A = channel.astype(np.float64)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k_eff = min(k, len(S))
    S_trunc = np.zeros_like(S)
    S_trunc[:k_eff] = S[:k_eff]
    recon = (U * S_trunc) @ Vt
    return np.clip(recon, 0, 255).astype(np.uint8), S
