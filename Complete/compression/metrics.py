import math
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def mse_metric(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))

def psnr_metric(a: np.ndarray, b: np.ndarray) -> float:
    mse = mse_metric(a, b)
    return float('inf') if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))

def ssim_metric(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(ssim(a, b, data_range=255))
