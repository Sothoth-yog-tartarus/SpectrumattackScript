import os
import numpy as np
import SimpleITK as sitk
import scipy.io as sio

nii_dir = "/data/Aaron/BTCV-rawdata/RawData/RawData/Training/img"
out_dir = "/output"

os.makedirs(out_dir + "/low_freq", exist_ok=True)
os.makedirs(out_dir + "/mid_freq", exist_ok=True)
os.makedirs(out_dir + "/high_freq", exist_ok=True)

files = sorted(os.listdir(nii_dir))


def fft_attack(volume):

    D,H,W = volume.shape

    F = np.fft.fftn(volume)
    F = np.fft.fftshift(F)

    center = np.array([D//2, H//2, W//2])

    r_low = int(min(D,H,W) * 0.1)
    r_mid = int(min(D,H,W) * 0.25)

    Z,Y,X = np.ogrid[:D,:H,:W]

    dist = np.sqrt(
        (Z-center[0])**2 +
        (Y-center[1])**2 +
        (X-center[2])**2
    )

    mask_low = dist < r_low

    mask_mid = (dist >= r_low) & (dist < r_mid)

    mask_high = dist >= r_mid

    F_low = F * mask_low
    F_mid = F * mask_mid
    F_high = F * mask_high

    low_img = np.fft.ifftn(np.fft.ifftshift(F_low)).real
    mid_img = np.fft.ifftn(np.fft.ifftshift(F_mid)).real
    high_img = np.fft.ifftn(np.fft.ifftshift(F_high)).real

    return low_img, mid_img, high_img


for fname in files:

    if not fname.endswith(".nii.gz"):
        continue

    case = fname.replace(".nii.gz","")

    nii_path = os.path.join(nii_dir,fname)

    img = sitk.ReadImage(nii_path)

    volume = sitk.GetArrayFromImage(img).astype(np.float32)

    low_ct, mid_ct, high_ct = fft_attack(volume)

    sio.savemat(
        os.path.join(out_dir,"low_freq",case+".mat"),
        {"volume":low_ct}
    )

    sio.savemat(
        os.path.join(out_dir,"mid_freq",case+".mat"),
        {"volume":mid_ct}
    )

    sio.savemat(
        os.path.join(out_dir,"high_freq",case+".mat"),
        {"volume":high_ct}
    )

    print("processed:",case)

print("DONE")
