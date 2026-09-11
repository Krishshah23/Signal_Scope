"""
SignalScope — Model Training Configuration
-------------------------------------------
Single source of truth for all training hyperparameters and paths.
Import this module anywhere training or inference configuration is needed.

Values here match the SIH 2026 roadmap specification:
  - 128×128 RGB input
  - MobileNetV2 backbone (frozen)
  - GlobalAveragePooling2D + Dense(1, sigmoid)
  - CIFAKE dataset
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — relative to project root, never hard-coded Windows paths
# ---------------------------------------------------------------------------

# Project root: two levels up from model/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Dataset directories — CIFAKE must be placed here locally.
# These folders are gitignored and must NOT be committed.
DATASET_TRAIN_DIR: Path = PROJECT_ROOT / "train"
DATASET_TEST_DIR: Path = PROJECT_ROOT / "test"

# Weights output — gitignored large binary files
WEIGHTS_DIR: Path = PROJECT_ROOT / "model" / "weights"
MODEL_SAVE_PATH: Path = WEIGHTS_DIR / "signalscope_baseline.keras"

# Training log / history output
HISTORY_SAVE_PATH: Path = WEIGHTS_DIR / "training_history.json"

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

IMAGE_HEIGHT: int = 128
IMAGE_WIDTH: int = 128
IMAGE_SIZE: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
NUM_CHANNELS: int = 3  # RGB
INPUT_SHAPE: tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)

# ---------------------------------------------------------------------------
# Label convention
# CIFAKE folder structure:  train/FAKE/  train/REAL/
#                           test/FAKE/   test/REAL/
#
# Label mapping:
#   FAKE  → 1   (AI-generated)
#   REAL  → 0   (real / not AI-generated)
#
# This means the model sigmoid output:
#   prob ≥ 0.5  →  "likely AI-generated"
#   prob <  0.5  →  "likely real"
# ---------------------------------------------------------------------------

CLASS_FAKE: str = "FAKE"
CLASS_REAL: str = "REAL"
LABEL_FAKE: int = 1
LABEL_REAL: int = 0
CLASS_NAMES: list[str] = [CLASS_REAL, CLASS_FAKE]  # index = label value

# Classification threshold (sigmoid output)
CLASSIFICATION_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 32
EPOCHS: int = 10
LEARNING_RATE: float = 1e-3
OPTIMIZER: str = "adam"
LOSS: str = "binary_crossentropy"

# Validation split fraction taken from training data ONLY.
# The test set is NEVER used for training, validation, or hyperparameter tuning.
VALIDATION_SPLIT: float = 0.15

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Data augmentation (applied to training data only, not validation or test)
# ---------------------------------------------------------------------------

AUGMENTATION_FLIP: str = "horizontal"     # random horizontal flip
AUGMENTATION_ROTATION: float = 0.05      # ±5% rotation
AUGMENTATION_ZOOM: float = 0.05          # ±5% zoom

# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

BACKBONE: str = "MobileNetV2"
BACKBONE_WEIGHTS: str = "imagenet"
BACKBONE_TRAINABLE: bool = False  # MUST remain False — frozen pretrained base

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def validate_paths() -> dict[str, bool]:
    """
    Check that required dataset paths exist.

    Returns
    -------
    dict
        {"train_dir": bool, "test_dir": bool, "weights_dir": bool}
    """
    return {
        "train_dir": DATASET_TRAIN_DIR.is_dir(),
        "train_fake": (DATASET_TRAIN_DIR / CLASS_FAKE).is_dir(),
        "train_real": (DATASET_TRAIN_DIR / CLASS_REAL).is_dir(),
        "test_dir": DATASET_TEST_DIR.is_dir(),
        "test_fake": (DATASET_TEST_DIR / CLASS_FAKE).is_dir(),
        "test_real": (DATASET_TEST_DIR / CLASS_REAL).is_dir(),
        "weights_dir": WEIGHTS_DIR.is_dir(),
    }


if __name__ == "__main__":
    print("SignalScope Model Configuration")
    print("=" * 40)
    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Train dir:         {DATASET_TRAIN_DIR}")
    print(f"Test dir:          {DATASET_TEST_DIR}")
    print(f"Model save path:   {MODEL_SAVE_PATH}")
    print(f"Image size:        {IMAGE_SIZE}")
    print(f"Input shape:       {INPUT_SHAPE}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Epochs:            {EPOCHS}")
    print(f"Learning rate:     {LEARNING_RATE}")
    print(f"Optimizer:         {OPTIMIZER}")
    print(f"Loss:              {LOSS}")
    print(f"Validation split:  {VALIDATION_SPLIT}")
    print(f"Random seed:       {RANDOM_SEED}")
    print(f"Backbone:          {BACKBONE}")
    print(f"Frozen base:       {not BACKBONE_TRAINABLE}")
    print()
    paths = validate_paths()
    print("Path checks:")
    for key, exists in paths.items():
        status = "✓" if exists else "✗ MISSING"
        print(f"  {key:20s}: {status}")
