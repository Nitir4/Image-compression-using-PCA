    def load_image(self):
        # Open file dialog to select an image
        self.img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not self.img_path:
            return
        self._process_image(cv2.imread(self.img_path))

    def on_drop(self, event):
        # Handle the case when an image is dropped into the window
        self.img_path = event.data
        if not os.path.isfile(self.img_path):
            return
        self._process_image(cv2.imread(self.img_path))

    def _process_image(self, img):
        # Convert image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Split the image into RGB channels
        self.r, self.g, self.b = cv2.split(img)
        
        # Normalize the channels to the range [0, 1]
        self.r, self.g, self.b = self.r / 255.0, self.g / 255.0, self.b / 255.0
        
        # Display the original image
        self.display_image(img, self.original_label)
        
        # Update the compressed image based on the selected k value
        self.update_image(int(self.k_input.get()))
