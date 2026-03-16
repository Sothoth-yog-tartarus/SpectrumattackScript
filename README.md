# SpectrumattackScript

This script is used to perform frequency-domain analysis on CT volume data from the BTCV dataset. The program first reads medical images in the .nii.gz format, then converts the images into the frequency domain through a three-dimensional Fourier transform. It subsequently extracts the low-frequency, mid-frequency, and high-frequency components separately. Finally, the data are transformed back to the spatial domain using the inverse Fourier transform, and the results are saved as .mat files for subsequent experiments.
