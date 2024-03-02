import os
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi
from skimage.measure import block_reduce

from interpolation import interpolate


ROOT_DIR = Path(os.path.dirname(os.path.dirname(__file__)))


def main():
    img_load_path = ROOT_DIR.joinpath("lab2/data/tif/rice.tif")
    img = np.array(cv2.imread(str(img_load_path), 0), dtype=np.uint8)
    img_downsample = block_reduce(img, block_size=(2, 2), func=np.max)
    
    ssim_scores = []
    psnr_scores = []
    nmi_scores = []
    time_cost = []
    methods = ["nearest", "linear", "cubic", "quintic"]
    for method in methods:
        img_interp, duration = interpolate(img_downsample, img.shape, method)
        ssim_score = round(ssim(img, img_interp), 3)
        psnr_score = round(psnr(img, img_interp), 2)
        nmi_score = round(nmi(img, img_interp), 2)
        
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
        nmi_scores.append(nmi_score)
        time_cost.append(duration)
        
    print(f"psnr: {psnr_scores}")
    print(f"ssim: {ssim_scores}")
    print(f"nmi: {nmi_scores}")
    print(f"time cost: {time_cost}")
    
    
if __name__ == "__main__":
    main()
