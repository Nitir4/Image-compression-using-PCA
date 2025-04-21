"""Legacy GUI entrypoint; drag and drop is optional."""
import tkinter as tk
try:
    from .Libraries import TkinterDnD
    from .GUI import PCACompressorGUI
except ImportError:
    from Libraries import TkinterDnD
    from GUI import PCACompressorGUI

if __name__ == '__main__':
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    app = PCACompressorGUI(root)
    root.mainloop()
