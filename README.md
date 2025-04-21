# Image-compression-using-PCA
This was done as a freshmen for my final project of linear algebra. It is basically 1 out of the 3 mini projects built for linear algebra.
The Enterk one is basically just a prototype I made just for the demonstration of PCA

Install `requirements.txt` into a virtual environment, with Tk support supplied
by your operating system. From the repository root, run `python -m Complete.main`
for PCA/SVD compression, comparison and metrics, or `python -m ENTERk.main` for
the older PCA demonstration. Optional `tkinterdnd2` enables drag and drop in the
prototype; selecting files works without it.

Saved images retain the original dimensions. Display previews are separate.
Choose a positive integer rank; ranks larger than the available dimension are
clamped. Single-row/column and constant images are supported. SSIM is reported
as undefined (`nan`) when either dimension is smaller than three pixels.

Saving exports the reconstructed pixels as a normal image, not a serialized PCA
factorization. The prototype's factor-storage estimate includes float64 scores,
basis vectors and means, relative to raw RGB bytes; it does not predict PNG/JPEG
file size.

Regression checks: `python -m pytest tests -q` (install `pytest` separately).
