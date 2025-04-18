from PIL import Image, ImageTk

def display_image(self, img_array, label):
    # Convert image array to a PIL image and display it in the GUI
    img_pil = Image.fromarray(img_array)
    img_pil.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img_pil)
    label.config(image=img_tk)
    label.image = img_tk

def display_size_reduction(self, k):
    # Calculate and display the size reduction of the compressed image
    height, width = self.r.shape
    k = min(k, height, width)
    original_size = height * width * 3
    compressed_size = 3 * (height * k + width * k + width) * 8
    size_reduction = 100 * (1 - (compressed_size / original_size))
    self.compression_info.config(
        text=f"Explained Variance: {self.compression_info.cget('text').split('%')[0]}% | Factor Storage Estimate: {size_reduction:.2f}%"
    )
