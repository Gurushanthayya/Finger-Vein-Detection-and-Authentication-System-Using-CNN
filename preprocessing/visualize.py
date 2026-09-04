"""
Visualization script for Finger Vein Preprocessing Pipeline (Week 4).
Applies grayscale conversion, contrast enhancement, vein masking, and Gabor filtering.
Saves the comparative visualization plot to docs/preprocessing_stages.png.
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend to prevent Win32 capsulation errors
import matplotlib.pyplot as plt

# Resolve paths when executing from script context
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocess import (
    to_grayscale, resize_image, apply_clahe, 
    extract_vein_mask, GABOR_KERNELS
)

def run_visualization():
    sample_path = os.path.join("dataset", "train", "001", "left", "index_1.bmp")
    if not os.path.exists(sample_path):
        print(f"[ERROR] Sample image not found at {sample_path}. Please complete Week 3 first.")
        return

    # Read original image (BGR)
    img_bgr = cv2.imread(sample_path)
    
    # 1. Grayscale
    img_gray = to_grayscale(img_bgr)
    
    # 2. Resized
    img_resized = resize_image(img_gray, (128, 64))
    
    # 3. CLAHE Contrast Enhanced
    img_enhanced = apply_clahe(img_resized)
    
    # 4. Vein Mask (Adaptive thresholding + morphology)
    img_mask = extract_vein_mask(img_enhanced)
    
    # 5. Gabor Filter response combination
    img_gabor = np.zeros_like(img_enhanced, dtype=np.float32)
    for k in GABOR_KERNELS[:4]:
        img_gabor += np.abs(cv2.filter2D(img_enhanced, cv2.CV_32F, k))
    img_gabor = cv2.normalize(img_gabor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Plot using matplotlib
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    
    # Titles and arrays mapping
    stages = [
        ("Original (BGR)", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)),
        ("Grayscale Resized", img_resized),
        ("CLAHE Enhanced", img_enhanced),
        ("Binarized Vein Mask", img_mask),
        ("Gabor Filter Map", img_gabor)
    ]
    
    for ax, (title, data) in zip(axes, stages):
        if len(data.shape) == 3:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap='gray')
        ax.set_title(title, fontsize=10, fontweight='semibold')
        ax.axis('off')
        
    plt.tight_layout()
    
    # Ensure docs directory exists
    os.makedirs("docs", exist_ok=True)
    out_path = os.path.join("docs", "preprocessing_stages.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[Done] Preprocessing visualization successfully saved to: {out_path}")


if __name__ == '__main__':
    run_visualization()
