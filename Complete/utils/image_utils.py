import cv2
import numpy as np
from compression.pca import custom_pca
from compression.svd import custom_svd

def compress_image_full(image: np.ndarray, k: int, method: str):
    chans = [image] if image.ndim == 2 else cv2.split(image)
    recon_chs, ratios = [], []
    for ch in chans:
        if method == 'PCA':
            recon, eigvals = custom_pca(ch, k)
            total = eigvals.sum() if eigvals.sum() > 0 else 1.0
            ratio = float(eigvals[:k].sum() / total)
        else:
            recon, S = custom_svd(ch, k)
            total = (S**2).sum() if (S**2).sum() > 0 else 1.0
            ratio = float((S[:k]**2).sum() / total)
        recon_chs.append(recon)
        ratios.append(ratio)
    merged = recon_chs[0] if len(recon_chs) == 1 else cv2.merge(recon_chs)
    return merged, float(np.mean(ratios))
