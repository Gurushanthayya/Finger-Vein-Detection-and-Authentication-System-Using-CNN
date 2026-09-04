"""
System Evaluation Script for Finger Vein Classification (Month 2).
Calculates test split metrics (Accuracy, Precision, Recall, F1-score)
and generates the Confusion Matrix.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')  # Headless mode for script execution
import matplotlib.pyplot as plt

# Resolve paths when executing from script context
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from training.train import load_dataset_split


def evaluate_model():
    print("=" * 60)
    print("      VEINGUARD — MODEL EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load trained classification model
    model_path = os.path.join(Config.MODELS_DIR, 'classification_cnn_model.keras')
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model not found at {model_path}. Please run train.py first.")
        return

    print("[Evaluation] Loading classification model...")
    try:
        try:
            import builtins
            builtins.tf = tf
            import keras
            keras.config.enable_unsafe_deserialization()
            # Patch Lambda shape inference for Keras 3
            def patched_compute_output_shape(self, input_shape):
                return input_shape
            keras.layers.Lambda.compute_output_shape = patched_compute_output_shape
        except Exception:
            pass
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    # Determine class mappings from folders
    train_dir = os.path.normpath(os.path.join(Config.DATASET_FOLDER, '..', 'train'))
    if not os.path.exists(train_dir):
        print("[ERROR] Training directory not found. Re-run Week 3 splitting.")
        return
        
    subjects = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_mapping = {subj: idx for idx, subj in enumerate(subjects)}
    num_classes = len(subjects)

    # 2. Load test split dataset
    print("[Evaluation] Loading test images and labels...")
    X_test, y_test = load_dataset_split('test', class_mapping)
    
    if len(X_test) == 0:
        print("[ERROR] Test set is empty. Cannot evaluate.")
        return
        
    print(f"[Evaluation] Loaded {len(X_test)} test images.")

    # 3. Perform predictions
    print("[Evaluation] Predicting test labels...")
    preds_probs = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(preds_probs, axis=1)

    # 4. Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n" + "="*40)
    print("      CLASSIFIER METRICS REPORT")
    print("="*40)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("="*40)

    # Write report file metrics.txt
    metrics_path = os.path.join(Config.BASE_DIR, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("--- CNN Classification Evaluation Metrics ---\n")
        f.write(f"Accuracy:  {acc * 100:.2f}%\n")
        f.write(f"Precision: {prec * 100:.2f}%\n")
        f.write(f"Recall:    {rec * 100:.2f}%\n")
        f.write(f"F1-Score:  {f1 * 100:.2f}%\n")
    print(f"[Done] Metrics report written to: {metrics_path}")

    # 5. Generate and plot Confusion Matrix
    print("[Evaluation] Plotting confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    # We display matrix as image. Since there are 106 classes, a dense plot is expected.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Classification Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    plt.xlabel('Predicted Label', fontsize=11, labelpad=10)
    plt.ylabel('True Label', fontsize=11, labelpad=10)
    plt.tight_layout()
    
    # Save confusion matrix plot
    os.makedirs(os.path.join(Config.BASE_DIR, 'docs'), exist_ok=True)
    cm_path = os.path.join(Config.BASE_DIR, 'docs', 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[Done] Confusion matrix plot saved to: {cm_path}")


if __name__ == '__main__':
    evaluate_model()
