import math
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.metrics import structural_similarity as ssim

# --- Custom PCA Implementation ---
def custom_pca(channel: np.ndarray, k: int):
    """Custom PCA on a single channel."""
    mean_cols = channel.mean(axis=0)
    centered = channel.astype(np.float64) - mean_cols
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # take top-k
    k_eff = min(k, eigvecs.shape[1])
    V_k = eigvecs[:, :k_eff]
    proj = centered @ V_k
    recon = proj @ V_k.T
    recon += mean_cols
    return np.clip(recon, 0, 255).astype(np.uint8), eigvals

# --- Custom SVD Implementation ---
def custom_svd(channel: np.ndarray, k: int):
    """Custom SVD on a single channel."""
    A = channel.astype(np.float64)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k_eff = min(k, len(S))
    S_trunc = np.zeros_like(S)
    S_trunc[:k_eff] = S[:k_eff]
    recon = (U * S_trunc) @ Vt
    return np.clip(recon, 0, 255).astype(np.uint8), S

# --- Compression Utilities ---
def compress_image_full(image: np.ndarray, k: int, method: str):
    """
    Compress image (2D gray or 3D BGR) channel-wise using custom PCA/SVD.
    Returns:
      - compressed uint8 image
      - mean explained-variance ratio across channels
    """
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

# --- Quality Metrics ---
def mse_metric(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))

def psnr_metric(a: np.ndarray, b: np.ndarray) -> float:
    mse = mse_metric(a, b)
    return float('inf') if mse == 0 else 20*math.log10(255.0/math.sqrt(mse))

def ssim_metric(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(ssim(a, b, data_range=255))

# --- GUI Application ---
class ImageCompressionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compression Explorer: PCA vs SVD")
        self.geometry("920x720")
        ttk.Style(self).theme_use('clam')
        self.font = ('Arial', 11)

        self.orig_pil = None
        self.orig_color = None
        self.orig_gray = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=5); top.pack(fill=tk.X)
        ttk.Button(top, text="Load Image", command=self.open_image).pack(side=tk.LEFT)
        self.file_lbl = ttk.Label(top, text="No file selected", font=self.font)
        self.file_lbl.pack(side=tk.LEFT, padx=10)
        self.rgb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="RGB Mode", variable=self.rgb_var).pack(side=tk.LEFT, padx=5)

        self.notebook = ttk.Notebook(self); self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._build_compress_tab()
        self._build_compare_tab()
        self._build_analyze_tab()

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.png *.jpg *.jpeg *.bmp")])
        if not path: return
        self.file_lbl.config(text=os.path.basename(path))
        pil = Image.open(path).convert('RGB')
        arr = np.array(pil)
        self.orig_pil   = pil
        self.orig_color = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        self.orig_gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        thumb = pil.copy(); thumb.thumbnail((200,200))
        tk_thumb = ImageTk.PhotoImage(thumb)
        for lbl in (self.orig_lbl1, self.orig_lbl2, self.orig_lbl3):
            lbl.config(image=tk_thumb); lbl.image = tk_thumb

    def _build_compress_tab(self):
        frm = ttk.Frame(self.notebook); self.notebook.add(frm, text="Compress Image")
        mf = ttk.Frame(frm, padding=5); mf.pack(anchor='w')
        ttk.Label(mf, text="Method:", font=self.font).pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value='PCA')
        ttk.Combobox(mf, textvariable=self.method_var, values=['PCA','SVD'], width=6).pack(side=tk.LEFT, padx=5)
        kf = ttk.Frame(frm, padding=5); kf.pack(anchor='w')
        ttk.Label(kf, text="k:", font=self.font).pack(side=tk.LEFT)
        self.k_cmp = tk.IntVar(value=10)
        ttk.Entry(kf, textvariable=self.k_cmp, width=6).pack(side=tk.LEFT, padx=5)
        bf = ttk.Frame(frm, padding=5); bf.pack(anchor='w')
        self.btn_cmp  = ttk.Button(bf, text="Compress",        command=self.compress_image)
        self.btn_save = ttk.Button(bf, text="Save Compressed", command=self.save_image, state=tk.DISABLED)
        self.btn_cmp.pack(side=tk.LEFT); self.btn_save.pack(side=tk.LEFT, padx=5)
        self.prog1 = ttk.Label(frm, font=self.font); self.prog1.pack(pady=5)
        imgf = ttk.Frame(frm); imgf.pack()
        self.orig_lbl1 = ttk.Label(imgf); self.orig_lbl1.pack(side=tk.LEFT, padx=5)
        self.comp_lbl1 = ttk.Label(imgf); self.comp_lbl1.pack(side=tk.LEFT, padx=5)
        self.stats1    = ttk.Label(frm, font=self.font); self.stats1.pack(pady=5)

    def _build_compare_tab(self):
        frm = ttk.Frame(self.notebook); self.notebook.add(frm, text="Compare PCA vs SVD")
        kf = ttk.Frame(frm, padding=5); kf.pack(anchor='w')
        ttk.Label(kf, text="k:", font=self.font).pack(side=tk.LEFT)
        self.k_cmp2 = tk.IntVar(value=10)
        ttk.Entry(kf, textvariable=self.k_cmp2, width=6).pack(side=tk.LEFT, padx=5)
        bf = ttk.Frame(frm, padding=5); bf.pack(anchor='w')
        self.btn_cmp2 = ttk.Button(bf, text="Compare", command=self.compare_methods)
        self.btn_cmp2.pack(side=tk.LEFT)
        self.prog2 = ttk.Label(frm, font=self.font); self.prog2.pack(pady=5)
        imgf = ttk.Frame(frm); imgf.pack()
        self.orig_lbl2 = ttk.Label(imgf); self.orig_lbl2.pack(side=tk.LEFT, padx=5)
        self.pca_lbl2  = ttk.Label(imgf); self.pca_lbl2.pack(side=tk.LEFT, padx=5)
        self.svd_lbl2  = ttk.Label(imgf); self.svd_lbl2.pack(side=tk.LEFT, padx=5)
        self.stats2    = ttk.Label(frm, font=self.font); self.stats2.pack(pady=5)

    def _build_analyze_tab(self):
        frm = ttk.Frame(self.notebook); self.notebook.add(frm, text="Analyze Metrics")
        kf = ttk.Frame(frm, padding=5); kf.pack(anchor='w')
        ttk.Label(kf, text="Max k:", font=self.font).pack(side=tk.LEFT)
        self.k_an = tk.IntVar(value=50)
        ttk.Entry(kf, textvariable=self.k_an, width=6).pack(side=tk.LEFT, padx=5)
        self.btn_an = ttk.Button(kf, text="Analyze", command=self.analyze_metrics)
        self.btn_an.pack(side=tk.LEFT, padx=5)
        self.prog3 = ttk.Label(frm, font=self.font); self.prog3.pack(pady=5)
        imgf = ttk.Frame(frm); imgf.pack()
        self.orig_lbl3 = ttk.Label(imgf); self.orig_lbl3.pack(side=tk.LEFT, padx=5)
        fig = plt.Figure(figsize=(6,4))
        self.ax_mse = fig.add_subplot(1,2,1); self.ax_var = fig.add_subplot(1,2,2)
        self.ax_mse.set_title("MSE vs k"); self.ax_var.set_title("Variance vs k")
        self.canvas = FigureCanvasTkAgg(fig, master=frm)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def compress_image(self):
        if not self.orig_pil:
            messagebox.showwarning("No Image","Load an image first."); return
        k, method, rgb = self.k_cmp.get(), self.method_var.get(), self.rgb_var.get()
        self.btn_cmp.config(state=tk.DISABLED)
        threading.Thread(target=self._thread_compress, args=(k,method,rgb), daemon=True).start()

    def _thread_compress(self, k, method, rgb):
        self.after(0, lambda: self.prog1.config(text=f"Compressing k={k}..."))
        data = self.orig_color if rgb else self.orig_gray
        comp, var = compress_image_full(data, k, method)
        mse  = mse_metric(data, comp)
        psnr = psnr_metric(data, comp)
        sm   = ssim_metric(data, comp)
        disp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB) if rgb else comp
        pil  = Image.fromarray(disp); pil.thumbnail((200,200)); tk_img = ImageTk.PhotoImage(pil)
        stats = f"{method} k={k}\nVar:{var:.2%} MSE:{mse:.2f} PSNR:{psnr:.2f} SSIM:{sm:.4f}"
        self.after(0, lambda: (
            self.comp_lbl1.config(image=tk_img),
            setattr(self,'comp_pil',pil),
            self.stats1.config(text=stats),
            self.prog1.config(text="Done."),
            self.btn_cmp.config(state=tk.NORMAL),
            self.btn_save.config(state=tk.NORMAL)
        ))

    def save_image(self):
        if not hasattr(self,'comp_pil'): return
        ext = os.path.splitext(self.file_lbl.cget('text'))[-1]
        path = filedialog.asksaveasfilename(defaultextension=ext)
        if path:
            self.comp_pil.save(path)
            messagebox.showinfo("Saved", f"Saved to {path}")

    def compare_methods(self):
        if not self.orig_pil:
            messagebox.showwarning("No Image","Load an image first."); return
        k, rgb = self.k_cmp2.get(), self.rgb_var.get()
        self.btn_cmp2.config(state=tk.DISABLED)
        threading.Thread(target=self._thread_compare, args=(k,rgb), daemon=True).start()

    def _thread_compare(self, k, rgb):
        self.after(0, lambda: self.prog2.config(text=f"Comparing k={k}..."))
        data = self.orig_color if rgb else self.orig_gray
        p_img, var_p = compress_image_full(data, k, 'PCA')
        s_img, var_s = compress_image_full(data, k, 'SVD')
        mse_p, psnr_p, sm_p = mse_metric(data, p_img), psnr_metric(data, p_img), ssim_metric(data, p_img)
        mse_s, psnr_s, sm_s = mse_metric(data, s_img), psnr_metric(data, s_img), ssim_metric(data, s_img)
        def to_pil(im): return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) if rgb else Image.fromarray(im)
        pil_p, pil_s = to_pil(p_img), to_pil(s_img)
        for lbl, pil in ((self.pca_lbl2,pil_p),(self.svd_lbl2,pil_s)):
            tmp=pil.copy(); tmp.thumbnail((200,200)); tl=ImageTk.PhotoImage(tmp)
            lbl.config(image=tl); lbl.image=tl
        stats = (
            f"PCA MSE:{mse_p:.2f} PSNR:{psnr_p:.2f} SSIM:{sm_p:.4f} Var:{var_p:.2%}\n"
            f"SVD MSE:{mse_s:.2f} PSNR:{psnr_s:.2f} SSIM:{sm_s:.4f} Var:{var_s:.2%}"
        )
        self.after(0, lambda: (
            self.stats2.config(text=stats),
            self.prog2.config(text="Done."),
            self.btn_cmp2.config(state=tk.NORMAL)
        ))

    def analyze_metrics(self):
        if not self.orig_pil:
            messagebox.showwarning("No Image","Load an image first."); return
        max_k, rgb = self.k_an.get(), self.rgb_var.get()
        self.btn_an.config(state=tk.DISABLED)
        threading.Thread(target=self._thread_analyze, args=(max_k,rgb), daemon=True).start()

    def _thread_analyze(self, max_k, rgb):
        data = self.orig_color if rgb else self.orig_gray
        mses_p, vars_p, mses_s, vars_s = [], [], [], []
        for k in range(1, max_k+1):
            self.after(0, lambda i=k: self.prog3.config(text=f"Analyzing k={i}/{max_k}"))
            p_img, var_p = compress_image_full(data, k, 'PCA')
            s_img, var_s = compress_image_full(data, k, 'SVD')
            mses_p.append(mse_metric(data, p_img))
            mses_s.append(mse_metric(data, s_img))
            vars_p.append(var_p)
            vars_s.append(var_s)
        self.after(0, lambda: self._update_plots(mses_p, vars_p, mses_s, vars_s, max_k))

    def _update_plots(self, mses_p, vars_p, mses_s, vars_s, max_k):
        ks = list(range(1, max_k+1))
        self.ax_mse.clear()
        self.ax_mse.plot(ks, mses_p, 'r-', label='PCA MSE')
        self.ax_mse.plot(ks, mses_s, 'b-', label='SVD MSE')
        self.ax_mse.set_title('MSE vs k'); self.ax_mse.legend()
        self.ax_var.clear()
        self.ax_var.plot(ks, vars_p, 'r--', label='PCA Var')
        self.ax_var.plot(ks, vars_s, 'b--', label='SVD Var')
        self.ax_var.set_title('Variance vs k'); self.ax_var.legend()
        self.canvas.draw()
        self.after(0, lambda: self.prog3.config(text='Analysis Done.'))

if __name__ == "__main__":
    app = ImageCompressionApp()
    app.mainloop()
