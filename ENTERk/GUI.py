class PCACompressorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PCA Image Compression Tool")
        self.root.geometry("1000x600")
        self.root.configure(bg="#f0f0f0")

        # Initialize placeholders for images and RGB channels
        self.img_original = None
        self.img_path = None
        self.r = self.g = self.b = None

        # Build the graphical user interface (GUI)
        self.build_gui()

    def build_gui(self):
        # Create the title label for the GUI window
        title = Label(self.root, text="PCA Image Compression Tool", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
        title.pack(pady=10)

        # Create frame for buttons and inputs
        button_frame = Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=5)

        # Load image button
        load_btn = Button(button_frame, text="Load Image", command=self.load_image, font=("Helvetica", 12), bg="#4285f4", fg="white")
        load_btn.pack(side=LEFT, padx=10)

        # Label and input for principal components (k)
        self.k_label = Label(button_frame, text="k = 50", font=("Helvetica", 12), bg="#f0f0f0")
        self.k_label.pack(side=LEFT, padx=10)

        self.k_input = Entry(button_frame, font=("Helvetica", 12), width=5)
        self.k_input.insert(0, "50")
        self.k_input.pack(side=LEFT, padx=10)
        self.k_input.bind("<Return>", self.on_k_input_change)

        # Buttons for plotting variance and size vs k
        Button(button_frame, text="Plot Size vs k", command=self.plot_size_vs_k, font=("Helvetica", 10)).pack(side=LEFT, padx=10)
        Button(button_frame, text="Plot Variance vs k", command=self.plot_variance_vs_k, font=("Helvetica", 10)).pack(side=LEFT, padx=10)

        # Information label for compression stats
        self.compression_info = Label(self.root, text="", font=("Helvetica", 10), bg="#f0f0f0")
        self.compression_info.pack(pady=5)

        # Frame for original and compressed images
        img_frame = Frame(self.root, bg="#f0f0f0")
        img_frame.pack(pady=10)

        # Labels for displaying images
        self.original_label = Label(img_frame, text="Original Image", bg="#f0f0f0")
        self.original_label.pack(side=LEFT, padx=20)

        self.compressed_label = Label(img_frame, text="Compressed Image", bg="#f0f0f0")
        self.compressed_label.pack(side=RIGHT, padx=20)

        # Allow drag-and-drop functionality for loading images
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

