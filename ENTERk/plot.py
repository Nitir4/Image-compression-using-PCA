import numpy as np
from tkinter import Toplevel, BOTH, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    from .PCA import PCA
except ImportError:
    from PCA import PCA

def plot_size_vs_k(self):
    if self.r is None:
        messagebox.showwarning("Warning", "Please load an image first!")
        return

    height, width = self.r.shape
    ks, reductions = [], []

    for k in range(1, min(height, width) + 1):
        original_size = height * width * 3
        compressed_size = 3 * (height * k + width * k + width) * 8
        reduction = 100 * (1 - (compressed_size / original_size))
        if reduction <= 0:
            break
        ks.append(k)
        reductions.append(reduction)

    self._show_plot_window("Estimated Factor Storage vs k", ks, reductions, "k", "Raw RGB Storage Reduction (%)")

def plot_variance_vs_k(self):
    if self.r is None:
        messagebox.showwarning("Warning", "Please load an image first!")
        return

    height, width = self.r.shape
    _, eig_r, _ = PCA(self.r, width)
    total_variance = np.sum(eig_r)

    ks, variances = [], []
    for k in range(1, min(height, width) + 1):
        explained_var = 100 * eig_r[:k].sum() / total_variance if total_variance > 0 else 100.0
        ks.append(k)
        variances.append(explained_var)

    self._show_plot_window("Explained Variance vs k", ks, variances, "k", "Explained Variance (%)")

def _show_plot_window(self, title, x, y, xlabel, ylabel):
    # Create a new window to display the plot
    plot_window = Toplevel(self.root)
    plot_window.title(title)
    plot_window.geometry("600x400")

    # Generate the plot using matplotlib
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker='o', linestyle='-', color='blue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    # Embed the plot into the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)
