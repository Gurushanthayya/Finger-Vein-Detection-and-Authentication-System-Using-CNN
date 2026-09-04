"""
Preprocessing module for Finger Vein Recognition.
Handles contrast enhancement (CLAHE), thresholding, morphological cleaning, and Gabor filter banks.
"""
import base64
from io import BytesIO
import cv2
import numpy as np
from PIL import Image

def build_gabor_kernels():
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 8):
        for sigma in [2, 4]:
            kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10, 0.5, 0, cv2.CV_32F)
            s = kernel.sum()
            kernel /= s if s != 0 else 1
            kernels.append(kernel)
    return kernels

GABOR_KERNELS = build_gabor_kernels()


def to_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def resize_image(img, size=(128, 64)):
    """Resizes the image to (width, height)"""
    return cv2.resize(img, size)


def apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Enhance local contrast of finger vein patterns"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def preprocess_image(img, size=(128, 64)):
    """
    Standard preprocessing pipeline for feeding into CNN.
    Grayscale -> Resize -> CLAHE -> Normalize -> Add Channel Dim.
    """
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    gray = to_grayscale(img)
    resized = resize_image(gray, size)
    enhanced = apply_clahe(resized)
    normalized = enhanced.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=-1)


def extract_vein_mask(enhanced_img):
    """
    Segment the vein structure using adaptive thresholding and morphological filters.
    """
    blurred = cv2.GaussianBlur(enhanced_img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def to_base64_str(img):
    """Convert numpy image array to base64 string for templates rendering"""
    if len(img.shape) == 2:
        pil = Image.fromarray(img)
    else:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def get_visual_stages(img):
    """
    Get base64 images of all preprocessing pipeline stages for UI display.
    """
    gray = to_grayscale(img)
    resized = resize_image(gray, (128, 64))
    enhanced = apply_clahe(resized)
    vein_mask = extract_vein_mask(enhanced)
    
    # Generate Gabor response visualization
    gabor_comp = np.zeros_like(enhanced, dtype=np.float32)
    for k in GABOR_KERNELS[:4]:
        gabor_comp += np.abs(cv2.filter2D(enhanced, cv2.CV_32F, k))
    gabor_comp = cv2.normalize(gabor_comp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Bone colormap for vein visualization
    vein_colored = cv2.applyColorMap(vein_mask, cv2.COLORMAP_BONE)
    
    return {
        'original': to_base64_str(resized),
        'enhanced': to_base64_str(enhanced),
        'vein_mask': to_base64_str(vein_mask),
        'gabor': to_base64_str(gabor_comp),
        'vein_colored': to_base64_str(vein_colored)
    }
