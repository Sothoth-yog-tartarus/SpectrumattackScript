import os
import numpy as np
import SimpleITK as sitk
import scipy.io as sio
import matplotlib.pyplot as plt

img_dir = "/data/Aaron/BTCV-rawdata/RawData/RawData/Training/img"
low_dir = "/data/Aaron/BTCVrawdata_remove/low_freq"
mid_dir = "/data/Aaron/BTCVrawdata_remove/mid_freq"
high_dir = "/data/Aaron/BTCVrawdata_remove/high_freq"

save_dir = "/output/compare_vis"
os.makedirs(save_dir, exist_ok=True)


def norm(x):
    p1, p99 = np.percentile(x, (1, 99))
    x = np.clip(x, p1, p99)
    x = (x - p1) / (p99 - p1 + 1e-8)
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

    print("Original range:", volume.min(), volume.max())

    z = volume.shape[0] // 2

    img_slice = volume[z]
    low_slice = low[z]
    mid_slice = mid[z]
    high_slice = high[z]

    # ---------------------------
    # 处理函数
    # ---------------------------
    # 原图可视化：百分位裁剪 + gamma
    p2, p98 = np.percentile(img_slice, (2, 98))
    img_vis = np.clip((img_slice - p2) / (p98 - p2 + 1e-8), 0, 1)
    img_vis = np.power(img_vis, 0.8)  # gamma 0.8，略暗，增强对比度但不过白

    # low/mid/high 可视化：均值/标准差对齐原图
    def match_vis(component, ref):
        comp_vis = (component - component.mean()) / (component.std() + 1e-8) * ref.std() + ref.mean()
        return np.clip(comp_vis, 0, 1)

    low_vis  = match_vis(low_slice, img_slice)
    mid_vis  = match_vis(mid_slice, img_slice)
    high_vis = match_vis(high_slice, img_slice)

    # ---------------------------
    # 绘图
    # ---------------------------
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img_vis, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(low_vis, cmap='gray')
    plt.title("Low")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(mid_vis, cmap='gray')
    plt.title("Mid")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(high_vis, cmap='gray')
    plt.title("High")
    plt.axis('off')

    save_path = os.path.join(save_dir, case + ".png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print("Saved:", case)

print("ALL DONE")
