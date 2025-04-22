    def on_k_input_change(self, event):
        # Update the compressed image when the k value is changed
        self.update_image(int(self.k_input.get()))

    def update_image(self, k):
        if self.r is None:
            return

        # Apply PCA on each color channel (R, G, B)
        r_rec, eig_r, _ = PCA(self.r, k)
        g_rec, eig_g, _ = PCA(self.g, k)
        b_rec, eig_b, _ = PCA(self.b, k)

        # Reconstruct the image from the compressed channels
        img_rec = cv2.merge(((r_rec * 255).clip(0, 255).astype(np.uint8),
                             (g_rec * 255).clip(0, 255).astype(np.uint8),
                             (b_rec * 255).clip(0, 255).astype(np.uint8)))

        # Display the compressed image
        self.display_image(img_rec, self.compressed_label)
        
        # Update the k value label
        self.k_label.config(text=f"k = {k}")

        # Calculate the explained variance for the compression
        variance = 100 * np.sum(np.sort(eig_r)[-k:]) / np.sum(eig_r)
        self.compression_info.config(text=f"Explained Variance: {variance:.2f}%")
        
        # Update size reduction info
        self.display_size_reduction(k)
