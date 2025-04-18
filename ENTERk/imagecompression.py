import cv2
import numpy as np
from tkinter import messagebox
try:
    from .PCA import PCA
except ImportError:
    from PCA import PCA

def on_k_input_change(self, event):
    # Update the compressed image when the k value is changed
    try:
        self.update_image(int(self.k_input.get()))
    except ValueError as exc:
        messagebox.showerror('Invalid k', str(exc))

def update_image(self, k):
    if self.r is None:
        return

    # Apply PCA on each color channel (R, G, B)
    r_rec, eig_r, _ = PCA(self.r, k)
    g_rec, eig_g, _ = PCA(self.g, k)
    b_rec, eig_b, _ = PCA(self.b, k)

    # Reconstruct the image from the compressed channels
    img_rec = cv2.merge((np.rint(r_rec * 255).clip(0, 255).astype(np.uint8),
                         np.rint(g_rec * 255).clip(0, 255).astype(np.uint8),
                         np.rint(b_rec * 255).clip(0, 255).astype(np.uint8)))

    # Display the compressed image
    self.display_image(img_rec, self.compressed_label)

    # Update the k value label
    self.k_label.config(text=f"k = {k}")

    # Calculate the explained variance for the compression
    variance = 100 * np.mean([eig[:k].sum() / eig.sum() if eig.sum() > 0 else 1.0 for eig in (eig_r, eig_g, eig_b)])
    self.compression_info.config(text=f"Explained Variance: {variance:.2f}%")

    # Update size reduction info
    self.display_size_reduction(k)
