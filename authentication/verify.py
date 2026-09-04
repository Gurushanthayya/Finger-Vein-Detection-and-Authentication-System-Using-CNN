"""
Finger Vein Feature Extraction and Identity Verification.
Implements dual extraction pipelines: CNN Embeddings and Handcrafted (Gabor + HOG + Topology) fallback.
"""
import os
import cv2
import logging
import numpy as np
from config import Config
from preprocessing.preprocess import to_grayscale, resize_image, apply_clahe, extract_vein_mask

logger = logging.getLogger(__name__)

# Ensure safe tensorflow deserialization
try:
    import builtins
    import tensorflow as tf
    builtins.tf = tf
    import keras
    keras.config.enable_unsafe_deserialization()
    def patched_compute_output_shape(self, input_shape):
        return input_shape
    keras.layers.Lambda.compute_output_shape = patched_compute_output_shape
except Exception:
    pass


class FingerVeinFeatureExtractor:
    """
    CNN + Handcrafted Hybrid Feature Extractor.
    Automatically detects if trained CNN model exists; otherwise, falls back to Gabor/HOG.
    """
    def __init__(self):
        self.cnn_model = None
        self.use_cnn = False
        self.gabor_kernels = self._build_gabor_kernels()
        self._try_load_cnn()

    def _try_load_cnn(self):
        cnn_path = os.path.join(Config.MODELS_DIR, 'vein_cnn_base.keras')
        if not os.path.exists(cnn_path):
            logger.info("[Extractor] CNN model not found. Using Gabor/HOG handcrafted fallback.")
            return

        try:
            import tensorflow as tf
            import builtins
            builtins.tf = tf

            logger.info(f"[Extractor] Loading CNN base model from {cnn_path}...")
            try:
                self.cnn_model = tf.keras.models.load_model(cnn_path, compile=False, safe_mode=False)
            except TypeError:
                self.cnn_model = tf.keras.models.load_model(cnn_path, compile=False)
            self.use_cnn = True
            logger.info("[Extractor] CNN model loaded successfully.")
        except Exception as e:
            logger.warning(f"[Extractor] Failed to load CNN model: {e}. Falling back to Gabor.")
            self.use_cnn = False

    def _build_gabor_kernels(self):
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 8):
            for sigma in [2, 4]:
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10, 0.5, 0, cv2.CV_32F)
                s = kernel.sum()
                kernel /= s if s != 0 else 1
                kernels.append(kernel)
        return kernels

    def extract_features(self, img):
        """
        Extract normalized features vector.
        Uses CNN if available, else Gabor + HOG.
        """
        if self.use_cnn and self.cnn_model is not None:
            return self._extract_cnn(img)
        return self._extract_handcrafted(img)

    def _extract_cnn(self, img):
        """Pass preprocessed image through CNN base network to get 128-d L2 normalized vector"""
        gray = to_grayscale(img)
        resized = resize_image(gray, (128, 64))
        enhanced = apply_clahe(resized)
        
        # Prepare for CNN: float32, /255.0, add channel and batch dimensions
        inp = enhanced.astype('float32') / 255.0
        inp = np.expand_dims(inp, axis=-1)
        inp = np.expand_dims(inp, axis=0)
        
        embedding = self.cnn_model.predict(inp, verbose=0)[0]
        
        # Ensure L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding.astype(np.float32)

    def _extract_handcrafted(self, img):
        """
        Gabor filters + Histogram of Oriented Gradients (HOG) + Topographic densities feature maps.
        """
        gray = to_grayscale(img)
        resized = resize_image(gray, (128, 64))
        enhanced = apply_clahe(resized)
        vein_mask = extract_vein_mask(enhanced)
        
        # 1. Gabor Feature Extraction
        gabor_features = []
        for kernel in self.gabor_kernels:
            filtered = cv2.filter2D(enhanced, cv2.CV_32F, kernel)
            h, w = filtered.shape
            ph, pw = h // 8, w // 8
            if ph > 0 and pw > 0:
                # Max-pool responses in 8x8 cells to reduce dimensionality
                pooled = filtered[:ph*8, :pw*8].reshape(ph, 8, pw, 8).max(axis=(1, 3))
                gabor_features.append(pooled.flatten())
        
        gabor_vec = np.concatenate(gabor_features) if gabor_features else np.zeros(512, dtype=np.float32)
        
        # 2. HOG Feature Extraction
        gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        hog_features = []
        h, w = enhanced.shape
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                hist, _ = np.histogram(
                    angle[i:i+8, j:j+8], 
                    bins=9, 
                    range=(0, 180), 
                    weights=magnitude[i:i+8, j:j+8]
                )
                hog_features.extend(hist)
        hog_vec = np.array(hog_features, dtype=np.float32)
        
        # 3. Topographical density ratios
        # Grid densities of vein mask (4x4 regions = 16 features) + total active ratio
        h, w = vein_mask.shape
        zones = []
        for r in range(4):
            for c in range(4):
                zone_pixels = vein_mask[r*h//4 : (r+1)*h//4, c*w//4 : (c+1)*w//4]
                ratio = np.sum(zone_pixels > 0) / (vein_mask.size / 16.0)
                zones.append(ratio)
        total_ratio = np.sum(vein_mask > 0) / float(vein_mask.size)
        topo_vec = np.array(zones + [total_ratio], dtype=np.float32)
        
        # Combine all handcrafted features and apply L2 normalization
        combined = np.concatenate([gabor_vec[:512], hog_vec[:256], topo_vec])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        return combined.astype(np.float32)
