"""
Training Script for Finger Vein Classification Model.
Loads training and validation sets, trains classification CNN, 
and extracts the base embedding sub-network.
"""
import os
import sys
import random
import numpy as np
import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Add project root directory to path for nested script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from preprocessing.preprocess import preprocess_image
from models.architecture import create_classification_model, extract_embedding_model

# Repro seeds
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

IMG_SHAPE = (64, 128, 1)


def load_dataset_split(split_name, class_mapping):
    """
    Loads all preprocessed images and labels for a specific split (train or validation).
    """
    split_dir = os.path.join(Config.DATASET_FOLDER, '..', split_name)
    split_dir = os.path.normpath(split_dir)
    
    if not os.path.exists(split_dir):
        print(f"[ERROR] Split directory not found: {split_dir}")
        return None, None

    X, y = [], []
    subject_dirs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    
    for subj in subject_dirs:
        if subj not in class_mapping:
            continue
        label_idx = class_mapping[subj]
        subj_path = os.path.join(split_dir, subj)
        
        for side in ['left', 'right']:
            side_path = os.path.join(subj_path, side)
            if not os.path.exists(side_path):
                continue
            
            for file_name in os.listdir(side_path):
                if file_name.lower().endswith(('.bmp', '.jpg', '.png')):
                    img_path = os.path.join(side_path, file_name)
                    processed = preprocess_image(img_path)
                    if processed is not None:
                        X.append(processed)
                        y.append(label_idx)
                        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y


def train_model():
    print("=" * 60)
    print("      VEINGUARD — CNN MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Detect subject folders and assign labels
    train_dir = os.path.normpath(os.path.join(Config.DATASET_FOLDER, '..', 'train'))
    if not os.path.exists(train_dir):
        print(f"[ERROR] Reorganized training folder not found at {train_dir}. Please complete Week 3.")
        return

    subjects = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(subjects)
    print(f"[Train] Found {num_classes} subject classes for classification.")

    # Create subject-to-index mapping
    class_mapping = {subj: idx for idx, subj in enumerate(subjects)}

    # 2. Load dataset splits
    print("[Train] Loading and preprocessing training split...")
    X_train, y_train = load_dataset_split('train', class_mapping)
    print(f"[Train] Loaded {len(X_train)} training images.")

    print("[Train] Loading and preprocessing validation split...")
    X_val, y_val = load_dataset_split('validation', class_mapping)
    print(f"[Train] Loaded {len(X_val)} validation images.")

    if len(X_train) == 0:
        print("[ERROR] Training dataset is empty. Check folders.")
        return

    # One-hot encode targets
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)

    # 3. Create Model
    model = create_classification_model(num_classes, IMG_SHAPE)
    model.summary()

    # Compile with categorical crossentropy
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    # Setup saving and checkpoints directories
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    full_model_path = os.path.join(Config.MODELS_DIR, 'classification_cnn_model.keras')
    base_model_path = os.path.join(Config.MODELS_DIR, 'vein_cnn_base.keras')

    callbacks = [
        ModelCheckpoint(full_model_path, monitor='val_loss', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
    ]

    # Data augmentation to mitigate overfitting
    # Create simple augmentation generator mapping inputs
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        fill_mode='constant',
        cval=0.0
    )
    datagen.fit(X_train)

    print("[Train] Starting model fitting...")
    model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=32),
        validation_data=(X_val, y_val_cat),
        epochs=15,
        callbacks=callbacks,
        verbose=2
    )

    # Re-load best weights to save base network
    if os.path.exists(full_model_path):
        try:
            model = tf.keras.models.load_model(full_model_path, compile=False)
        except Exception:
            pass

    # Extract base embedding model
    embedding_model = extract_embedding_model(model)
    embedding_model.save(base_model_path)
    
    print("\n" + "="*50)
    print(f"[Done] Full classifier model saved -> {full_model_path}")
    print(f"[Done] Feature embedding model saved -> {base_model_path}")
    print("="*50)


if __name__ == '__main__':
    train_model()
