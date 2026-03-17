import os
import numpy as np
import SimpleITK as sitk
import scipy.io as sio
import matplotlib.pyplot as plt

img_dir = "/data/Aaron/BTCV-rawdata/RawData/RawData/Training/img"
low_dir = "/data/Aaron/BTCV-RawData_freq/low_freq"
mid_dir = "/data/Aaron/BTCV-RawData_freq/mid_freq"
high_dir = "/data/Aaron/BTCV-RawData_freq/high_freq"

save_dir = "/output/compare_vis"
os.makedirs(save_dir, exist_ok=True)

def norm(x):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x

files = sorted([f for f in os.listdir(img_dir) if f.endswith(".nii.gz")])
print("Total cases:", len(files))


for fname in files:

    case = fname.replace(".nii.gz", "")

    print("Processing:", case)

    img_path = os.path.join(img_dir, fname)
    img = sitk.ReadImage(img_path)
    volume = sitk.GetArrayFromImage(img).astype(np.float32)

    low_path = os.path.join(low_dir, case + ".mat")
    mid_path = os.path.join(mid_dir, case + ".mat")
    high_path = os.path.join(high_dir, case + ".mat")

    if not (os.path.exists(low_path) and os.path.exists(mid_path) and os.path.exists(high_path)):
        print("Missing:", case)
        continue

    low = sio.loadmat(low_path)["volume"]
    mid = sio.loadmat(mid_path)["volume"]
    high = sio.loadmat(high_path)["volume"]

    z = volume.shape[0] // 2

    img_slice = volume[z]
    low_slice = low[z]
    mid_slice = mid[z]
    high_slice = high[z]

    plt.figure(figsize=(10, 8))

    plt.subplot(2,2,1)
    plt.imshow(norm(img_slice), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(norm(low_slice), cmap='gray')
    plt.title("Low")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(norm(mid_slice), cmap='gray')
    plt.title("Mid")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(norm(high_slice), cmap='gray')
    plt.title("High")
    plt.axis('off')

    save_path = os.path.join(save_dir, case + ".png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print("Saved:", case)

print("ALL DONE")
