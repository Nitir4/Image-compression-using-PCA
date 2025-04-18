import queue
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

try:
    from .compression import (custom_pca, custom_svd, compress_image_full,
                              mse_metric, psnr_metric, validate_k)
except ImportError:  # python Complete/main.py
    from compression import (custom_pca, custom_svd, compress_image_full,
                             mse_metric, psnr_metric, validate_k)

def ssim_metric(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    window = min(7, *a.shape[:2])
    if window < 3:
        return float('nan')  # SSIM is undefined for fewer than 3 pixels per side.
    if window % 2 == 0:
        window -= 1
    return float(ssim(a, b, data_range=255, win_size=window))

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

        self.comp_pil = None
        self._updates = queue.Queue()
        self._busy = False
        self._build_ui()
        self.after(50, self._drain_updates)

    def _build_ui(self):
        top = ttk.Frame(self, padding=5); top.pack(fill=tk.X)
        self.btn_load = ttk.Button(top, text="Load Image", command=self.open_image)
        self.btn_load.pack(side=tk.LEFT)
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
        try:
            with Image.open(path) as image:
                pil = image.convert('RGB')
        except (OSError, ValueError) as exc:
            messagebox.showerror('Cannot load image', str(exc)); return
        self.file_lbl.config(text=os.path.basename(path))
        self.comp_pil = None
        self.btn_save.config(state=tk.DISABLED)
        self.comp_lbl1.config(image='')
        self.comp_lbl1.image = None
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

    def _ui(self, callback):
        self._updates.put(callback)

    def _drain_updates(self):
        while True:
            try:
                callback = self._updates.get_nowait()
            except queue.Empty:
                break
            callback()
        self.after(50, self._drain_updates)

    def _finish_job(self):
        self._busy = False
        for button in (self.btn_load, self.btn_cmp, self.btn_cmp2, self.btn_an):
            button.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL if self.comp_pil is not None else tk.DISABLED)

    def _start_job(self, worker, args, status):
        if self._busy:
            return
        if self.orig_pil is None:
            messagebox.showwarning('No Image', 'Load an image first.'); return
        try:
            validate_k(args[0])
        except (ValueError, TypeError) as exc:
            messagebox.showerror('Invalid k', str(exc)); return
        self._busy = True
        for button in (self.btn_load, self.btn_cmp, self.btn_cmp2, self.btn_an, self.btn_save):
            button.config(state=tk.DISABLED)
        data = (self.orig_color if args[-1] else self.orig_gray).copy()
        def run():
            try:
                worker(data, *args)
            except Exception as exc:
                error = str(exc)
                self._ui(lambda: (status.config(text='Failed.'), messagebox.showerror('Processing failed', error)))
            finally:
                self._ui(self._finish_job)
        threading.Thread(target=run, daemon=True).start()

    def compress_image(self):
        try:
            k = self.k_cmp.get()
        except tk.TclError:
            messagebox.showerror('Invalid k', 'k must be a positive integer.'); return
        self._start_job(self._thread_compress, (k, self.method_var.get(), self.rgb_var.get()), self.prog1)

    def _thread_compress(self, data, k, method, rgb):
        self._ui(lambda: self.prog1.config(text=f'Compressing k={k}...'))
        comp, var = compress_image_full(data, k, method)
        mse, psnr, sm = mse_metric(data, comp), psnr_metric(data, comp), ssim_metric(data, comp)
        disp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB) if rgb else comp
        full_image = Image.fromarray(disp)
        stats = f'{method} k={k}\nVar:{var:.2%} MSE:{mse:.2f} PSNR:{psnr:.2f} SSIM:{sm:.4f}'
        self._ui(lambda: self._show_compression(full_image, stats))

    def _show_compression(self, full_image, stats):
        self.comp_pil = full_image
        thumbnail = full_image.copy()
        thumbnail.thumbnail((200, 200))
        preview = ImageTk.PhotoImage(thumbnail)
        self.comp_lbl1.config(image=preview)
        self.comp_lbl1.image = preview
        self.stats1.config(text=stats)
        self.prog1.config(text='Done.')

    def save_image(self):
        if self.comp_pil is None: return
        ext = os.path.splitext(self.file_lbl.cget('text'))[-1]
        path = filedialog.asksaveasfilename(defaultextension=ext)
        if path:
            try:
                self.comp_pil.save(path)
            except (OSError, ValueError) as exc:
                messagebox.showerror('Cannot save image', str(exc)); return
            messagebox.showinfo("Saved", f"Saved to {path}")

    def compare_methods(self):
        try:
            k = self.k_cmp2.get()
        except tk.TclError:
            messagebox.showerror('Invalid k', 'k must be a positive integer.'); return
        self._start_job(self._thread_compare, (k, self.rgb_var.get()), self.prog2)

    def _thread_compare(self, data, k, rgb):
        self._ui(lambda: self.prog2.config(text=f'Comparing k={k}...'))
        p_img, var_p = compress_image_full(data, k, 'PCA')
        s_img, var_s = compress_image_full(data, k, 'SVD')
        mse_p, psnr_p, sm_p = mse_metric(data, p_img), psnr_metric(data, p_img), ssim_metric(data, p_img)
        mse_s, psnr_s, sm_s = mse_metric(data, s_img), psnr_metric(data, s_img), ssim_metric(data, s_img)
        images = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if rgb else im) for im in (p_img, s_img)]
        stats = (
            f'PCA MSE:{mse_p:.2f} PSNR:{psnr_p:.2f} SSIM:{sm_p:.4f} Var:{var_p:.2%}\n'
            f'SVD MSE:{mse_s:.2f} PSNR:{psnr_s:.2f} SSIM:{sm_s:.4f} Var:{var_s:.2%}'
        )
        self._ui(lambda: self._show_comparison(images, stats))

    def _show_comparison(self, images, stats):
        for label, image in zip((self.pca_lbl2, self.svd_lbl2), images):
            thumbnail = image.copy()
            thumbnail.thumbnail((200, 200))
            preview = ImageTk.PhotoImage(thumbnail)
            label.config(image=preview)
            label.image = preview
        self.stats2.config(text=stats)
        self.prog2.config(text='Done.')

    def analyze_metrics(self):
        try:
            k = self.k_an.get()
        except tk.TclError:
            messagebox.showerror('Invalid k', 'k must be a positive integer.'); return
        self._start_job(self._thread_analyze, (k, self.rgb_var.get()), self.prog3)

    def _thread_analyze(self, data, max_k, rgb):
        max_k = min(max_k, min(data.shape[:2]))
        mses_p, vars_p, mses_s, vars_s = [], [], [], []
        for k in range(1, max_k + 1):
            self._ui(lambda i=k: self.prog3.config(text=f'Analyzing k={i}/{max_k}'))
            p_img, var_p = compress_image_full(data, k, 'PCA')
            s_img, var_s = compress_image_full(data, k, 'SVD')
            mses_p.append(mse_metric(data, p_img))
            mses_s.append(mse_metric(data, s_img))
            vars_p.append(var_p)
            vars_s.append(var_s)
        self._ui(lambda: self._update_plots(mses_p, vars_p, mses_s, vars_s, max_k))

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
        self.prog3.config(text='Analysis Done.')

if __name__ == "__main__":
    app = ImageCompressionApp()
    app.mainloop()
