"""
System Performance Graph Generator (Month 4).
Generates training history curves, biometric threshold sweeps (Precision, Recall, F1),
and Error Rate curves (FAR/FRR) to locate the Equal Error Rate (EER).
Saves layout to docs/performance_curves.png.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt

# Resolve paths when executing from script context
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from database.db import DatabaseManager
from training.train import load_dataset_split
from preprocessing.preprocess import preprocess_image

# Safe deserialization for Lambda L2 norm layer loading
try:
    import builtins
    builtins.tf = tf
    import keras
    keras.config.enable_unsafe_deserialization()
    def patched_compute_output_shape(self, input_shape):
        return input_shape
    keras.layers.Lambda.compute_output_shape = patched_compute_output_shape
except Exception:
    pass


def load_test_embeddings():
    """
    Loads test images, runs them through the trained CNN base model, 
    and returns embedding vectors and subject-hand-finger class labels.
    """
    model_path = os.path.join(Config.MODELS_DIR, 'vein_cnn_base.keras')
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained base model not found at {model_path}. Run train.py first.")
        return None, None

    print("[Plotter] Loading trained feature extractor sub-network...")
    try:
        extractor = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"[ERROR] Could not load base model: {e}")
        return None, None

    test_dir = os.path.normpath(os.path.join(Config.DATASET_FOLDER, '..', 'test'))
    if not os.path.exists(test_dir):
        print(f"[ERROR] Test split folder not found at {test_dir}. Run Week 3 split script.")
        return None, None

    # Load and preprocess all test images grouped by class
    # Format of classes: subject_side_finger (e.g. "001_left_index")
    subject_dirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    embeddings = []
    labels = []
    
    for subj in subject_dirs:
        subj_path = os.path.join(test_dir, subj)
        for side in ['left', 'right']:
            side_path = os.path.join(subj_path, side)
            if not os.path.exists(side_path):
                continue
                
            files = [f for f in os.listdir(side_path) if f.lower().endswith(('.bmp', '.jpg', '.png'))]
            for f in files:
                finger_name = f.split('_')[0]
                class_label = f"{subj}_{side}_{finger_name}"
                
                img_path = os.path.join(side_path, f)
                preprocessed = preprocess_image(img_path)
                if preprocessed is not None:
                    # Predict embedding
                    inp = np.expand_dims(preprocessed, axis=0)
                    feat = extractor.predict(inp, verbose=0)[0]
                    
                    embeddings.append(feat)
                    labels.append(class_label)
                    
    return np.array(embeddings, dtype=np.float32), labels


def generate_biometric_curves(embeddings, labels):
    """
    Simulates comparisons to sweep threshold values and compute FAR/FRR/Precision/Recall.
    """
    print("[Plotter] Performing cross-comparison sweeps...")
    # Generate positive comparison pairs (genuine matches)
    # Generate negative comparison pairs (imposter matches)
    genuine_scores = []
    imposter_scores = []
    
    n = len(embeddings)
    # Cosine matching helper
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Compare a subset to keep computation fast
    step = max(1, n // 150)
    for i in range(0, n, step):
        for j in range(i + 1, n, step):
            score = cosine_sim(embeddings[i], embeddings[j])
            if labels[i] == labels[j]:
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)
                
    genuine_scores = np.array(genuine_scores)
    imposter_scores = np.array(imposter_scores)

    thresholds = np.arange(0.60, 0.99, 0.01)
    far_list, frr_list = [], []
    precision_list, recall_list, f1_list = [], [], []

    for t in thresholds:
        # False Accept Rate (FAR): Imposter scores >= threshold
        far = np.sum(imposter_scores >= t) / len(imposter_scores) if len(imposter_scores) > 0 else 0
        # False Reject Rate (FRR): Genuine scores < threshold
        frr = np.sum(genuine_scores < t) / len(genuine_scores) if len(genuine_scores) > 0 else 0
        
        far_list.append(far * 100)
        frr_list.append(frr * 100)

        # Standard precision/recall on combined test classifications
        # Define: class positive = genuine match, class negative = imposter match
        tp = np.sum(genuine_scores >= t)
        fp = np.sum(imposter_scores >= t)
        fn = np.sum(genuine_scores < t)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_list.append(precision * 100)
        recall_list.append(recall * 100)
        f1_list.append(f1 * 100)
        
    return thresholds, far_list, frr_list, precision_list, recall_list, f1_list


def plot_performance_graphs():
    # 1. Load data and generate embeddings
    embeddings, labels = load_test_embeddings()
    if embeddings is None or len(embeddings) == 0:
        print("[ERROR] Could not extract embeddings for graphs.")
        return

    # Sweep thresholds
    thresholds, far, frr, prec, rec, f1 = generate_biometric_curves(embeddings, labels)

    # 2. Define training history metrics values (parsed from task logs)
    epochs = np.arange(1, 16)
    train_acc = [2.12, 9.98, 16.98, 25.35, 30.74, 35.85, 41.98, 44.58, 50.59, 53.14, 57.27, 61.48, 63.21, 66.43, 67.89]
    val_acc = [0.94, 0.94, 0.94, 0.94, 1.10, 4.87, 15.41, 21.07, 44.81, 44.65, 35.69, 18.55, 43.71, 41.51, 44.65]
    train_loss = [4.62, 4.35, 4.13, 3.93, 3.75, 3.58, 3.40, 3.24, 3.07, 2.92, 2.74, 2.60, 2.45, 2.30, 2.16]
    val_loss = [4.68, 4.68, 4.66, 4.64, 4.60, 4.35, 3.66, 3.57, 3.09, 2.95, 3.02, 3.50, 2.73, 2.67, 2.58]

    # Create figure layout (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Model Training History
    ax0 = axes[0]
    ax0.plot(epochs, train_acc, 'o-', color='#4361ee', linewidth=2, label='Train Acc')
    ax0.plot(epochs, val_acc, 's--', color='#ffd32a', linewidth=2, label='Val Acc')
    ax0.set_title('CNN Classifier Training History', fontsize=12, fontweight='bold', pad=10)
    ax0.set_xlabel('Epochs', fontsize=10)
    ax0.set_ylabel('Accuracy (%)', fontsize=10)
    ax0.grid(True, linestyle=':', alpha=0.6)
    
    # Dual y-axis for loss curve
    ax0_loss = ax0.twinx()
    ax0_loss.plot(epochs, train_loss, 'x-', color='#ff4757', linewidth=1.5, alpha=0.7, label='Train Loss')
    ax0_loss.plot(epochs, val_loss, '^--', color='#00e6b4', linewidth=1.5, alpha=0.7, label='Val Loss')
    ax0_loss.set_ylabel('Categorical Loss', fontsize=10)
    
    # Combined legend
    lines1, labels1 = ax0.get_legend_handles_labels()
    lines2, labels2 = ax0_loss.get_legend_handles_labels()
    ax0.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # Plot 2: Biometric Performance Curves
    ax1 = axes[1]
    ax1.plot(thresholds, prec, '-', color='#00e6b4', linewidth=2, label='Precision')
    ax1.plot(thresholds, rec, '--', color='#ff4757', linewidth=2, label='Recall')
    ax1.plot(thresholds, f1, ':', color='#ffd32a', linewidth=2.5, label='F1-Score')
    ax1.set_title('Biometric Precision / Recall vs. Cosine Threshold', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Cosine Similarity Threshold', fontsize=10)
    ax1.set_ylabel('Score (%)', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left')

    # Plot 3: Biometric Error Rates (FAR & FRR) -> EER
    ax2 = axes[2]
    ax2.plot(thresholds, far, '-', color='#ff4757', linewidth=2.5, label='FAR (False Accept Rate)')
    ax2.plot(thresholds, frr, '--', color='#4361ee', linewidth=2.5, label='FRR (False Reject Rate)')
    
    # Calculate Equal Error Rate (EER) intersection
    diffs = np.abs(np.array(far) - np.array(frr))
    eer_idx = np.argmin(diffs)
    eer_threshold = thresholds[eer_idx]
    eer_value = far[eer_idx]
    
    ax2.plot(eer_threshold, eer_value, 'ro', markersize=8, label=f'EER Checkpoint ({eer_value:.2f}%)')
    ax2.annotate(
        f'EER: {eer_value:.2f}%\nThreshold: {eer_threshold:.2f}', 
        xy=(eer_threshold, eer_value), 
        xytext=(eer_threshold - 0.12, eer_value + 10),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6)
    )

    ax2.set_title('Biometric FAR / FRR Error Rates', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Cosine Similarity Threshold', fontsize=10)
    ax2.set_ylabel('Error Rate (%)', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    
    # Save the output image
    os.makedirs(os.path.join(Config.BASE_DIR, 'docs'), exist_ok=True)
    out_path = os.path.join(Config.BASE_DIR, 'docs', 'performance_curves.png')
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    
    print(f"[Done] Performance curves successfully saved to: {out_path}")


if __name__ == '__main__':
    plot_performance_graphs()
