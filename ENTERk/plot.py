    def plot_size_vs_k(self):
        if self.r is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        height, width = self.r.shape
        ks, reductions = [], []

        for k in range(1, width + 1):
            original_size = height * width * 3
            compressed_size = 3 * (height * k + width * k)
            reduction = 100 * (1 - (compressed_size / original_size))
            if reduction <= 0:
                break
            ks.append(k)
            reductions.append(reduction)

        self._show_plot_window("Size Reduction vs k", ks, reductions, "k", "Size Reduction (%)")

    def plot_variance_vs_k(self):
        if self.r is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        height, width = self.r.shape
        _, eig_r, _ = PCA(self.r, width)
        total_variance = np.sum(eig_r)

        ks, variances = [], []
        for k in range(1, width + 1):
            explained_var = 100 * np.sum(np.sort(eig_r)[-k:]) / total_variance
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
