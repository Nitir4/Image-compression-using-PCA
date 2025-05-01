import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from compression.pca import custom_pca
from compression.svd import custom_svd
from compression.metrics import mse_metric, psnr_metric, ssim_metric
from utils.image_utils import compress_image_full
