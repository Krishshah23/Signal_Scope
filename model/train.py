"""
SignalScope — Transfer Learning Baseline Training
--------------------------------------------------
Trains a MobileNetV2-based binary classifier on the CIFAKE dataset.

Architecture:
    Input (128×128×3 RGB)
        ↓
    MobileNetV2 (pretrained ImageNet weights, FROZEN)
        ↓
    GlobalAveragePooling2D
        ↓
    Dense(1, activation='sigmoid')

Label convention:
    1 → FAKE (AI-generated)
    0 → REAL (real photograph)

Reproducibility:
    Random seeds are set for Python, NumPy, and TensorFlow at startup.
    Exact reproducibility across different hardware is not guaranteed
    due to non-deterministic GPU operations. On CPU-only environments
    results are more reproducible.

Data policy:
    ONLY the training directory is used for model.fit().
    Validation data is derived from the training directory only.
    The test/ directory is NEVER touched during training.

Usage:
    python model/train.py
    python model/train.py --epochs 5 --batch-size 16

From project root:
    .venv/Scripts/python model/train.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure model/ is on sys.path for relative imports
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODEL_DIR.parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("signalscope.train")


# ---------------------------------------------------------------------------
# Seed function — called BEFORE importing TensorFlow
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow.

    Note: Exact reproducibility is not guaranteed across different hardware
    configurations or TF versions due to non-deterministic GPU operations.
    CPU-only training is more deterministic.

    Parameters
    ----------
    seed : int
        The seed value to use for all random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Encourage more deterministic ops on CPU
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass

    logger.info("Random seeds set to: %d", seed)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(input_shape: tuple[int, int, int] = (128, 128, 3)) -> "tf.keras.Model":
    """
    Build the SignalScope baseline model.

    Architecture
    ------------
    Input (128×128×3)
        → MobileNetV2 (ImageNet weights, FROZEN)
        → GlobalAveragePooling2D
        → Dense(1, sigmoid)

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Model input shape as (height, width, channels). Default: (128, 128, 3).

    Returns
    -------
    tf.keras.Model
        Compiled model ready for training.

    Raises
    ------
    AssertionError
        If the architecture does not match the expected specification.
    """
    import tensorflow as tf

    logger.info("Building model — backbone: MobileNetV2, input: %s", input_shape)

    # ------------------------------------------------------------------
    # Pretrained backbone — FROZEN
    # ------------------------------------------------------------------
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,          # Remove ImageNet classification head
        weights="imagenet",         # Load pretrained ImageNet weights
    )
    base_model.trainable = False    # FREEZE all base model layers

    logger.info(
        "MobileNetV2 loaded — layers: %d, trainable layers: %d",
        len(base_model.layers),
        sum(1 for l in base_model.layers if l.trainable),
    )

    # ------------------------------------------------------------------
    # Classification head
    # ------------------------------------------------------------------
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # MobileNetV2 expects inputs in [-1, 1], which its preprocess_input
    # handles. Our pipeline normalises to [0, 1], so we apply it here.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)   # training=False keeps BN in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SignalScope_MobileNetV2")

    # ------------------------------------------------------------------
    # Architecture assertions
    # ------------------------------------------------------------------
    _assert_architecture(model, base_model)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    logger.info("Model compiled successfully.")
    model.summary(print_fn=logger.info)

    return model


def _assert_architecture(model, base_model) -> None:
    """
    Assert that the model architecture meets the roadmap specification.
    Raises AssertionError with a descriptive message if any check fails.
    """
    import tensorflow as tf

    # 1. Input shape
    expected_input = (128, 128, 3)
    actual_input = tuple(model.input_shape[1:])
    assert actual_input == expected_input, (
        f"Input shape mismatch: expected {expected_input}, got {actual_input}"
    )

    # 2. Base model is MobileNetV2
    assert "mobilenet" in base_model.name.lower(), (
        f"Expected MobileNetV2 backbone, got: {base_model.name}"
    )

    # 3. Base model is frozen
    assert not base_model.trainable, (
        "Base model must be frozen (trainable=False)."
    )
    trainable_base_layers = [l for l in base_model.layers if l.trainable]
    assert len(trainable_base_layers) == 0, (
        f"Found {len(trainable_base_layers)} trainable layers in frozen base."
    )

    # 4. GlobalAveragePooling2D exists
    gap_layers = [l for l in model.layers
                  if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)]
    assert len(gap_layers) == 1, (
        f"Expected exactly 1 GlobalAveragePooling2D layer, found {len(gap_layers)}."
    )

    # 5. Final Dense layer has 1 unit with sigmoid activation
    output_layer = model.layers[-1]
    assert isinstance(output_layer, tf.keras.layers.Dense), (
        f"Final layer must be Dense, got: {type(output_layer).__name__}"
    )
    assert output_layer.units == 1, (
        f"Final Dense layer must have 1 unit, got: {output_layer.units}"
    )
    assert output_layer.activation.__name__ == "sigmoid", (
        f"Final Dense activation must be sigmoid, got: {output_layer.activation.__name__}"
    )

    # 6. Output is a binary probability
    assert tuple(model.output_shape) == (None, 1), (
        f"Output shape must be (None, 1), got: {model.output_shape}"
    )

    logger.info(
        "Architecture assertions PASSED: "
        "input=%s, backbone=MobileNetV2(frozen), "
        "GlobalAveragePooling2D, Dense(1, sigmoid)",
        expected_input,
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    train_dir: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    validation_split: float | None = None,
    save_path: str | Path | None = None,
) -> dict:
    """
    Run the full training pipeline.

    1. Set random seeds
    2. Build the model
    3. Load CIFAKE training + validation data (test untouched)
    4. Train with model.fit()
    5. Save the trained model
    6. Save training history
    7. Return result summary

    Parameters
    ----------
    All parameters fall back to values in model/config.py if not provided.

    Returns
    -------
    dict
        Training result summary with configuration and final metrics.
    """
    from config import (
        DATASET_TRAIN_DIR,
        MODEL_SAVE_PATH,
        HISTORY_SAVE_PATH,
        INPUT_SHAPE,
        BATCH_SIZE,
        EPOCHS,
        VALIDATION_SPLIT,
        RANDOM_SEED,
        WEIGHTS_DIR,
    )
    from dataset import (
        build_train_val_datasets,
        verify_no_test_in_train,
        inspect_dataset,
    )
    import tensorflow as tf

    # Resolve parameters (arguments override config defaults)
    _train_dir = Path(train_dir) if train_dir else DATASET_TRAIN_DIR
    _epochs = epochs if epochs is not None else EPOCHS
    _batch_size = batch_size if batch_size is not None else BATCH_SIZE
    _seed = seed if seed is not None else RANDOM_SEED
    _val_split = validation_split if validation_split is not None else VALIDATION_SPLIT
    _save_path = Path(save_path) if save_path else MODEL_SAVE_PATH

    logger.info("=" * 60)
    logger.info("SignalScope Baseline Training")
    logger.info("=" * 60)
    logger.info("Train dir:         %s", _train_dir)
    logger.info("Epochs:            %d", _epochs)
    logger.info("Batch size:        %d", _batch_size)
    logger.info("Validation split:  %.0f%%", _val_split * 100)
    logger.info("Random seed:       %d", _seed)
    logger.info("Save path:         %s", _save_path)

    # ------------------------------------------------------------------
    # Set seeds
    # ------------------------------------------------------------------
    set_seeds(_seed)

    # ------------------------------------------------------------------
    # Inspect and validate training data
    # ------------------------------------------------------------------
    train_info = inspect_dataset(_train_dir, "train")
    if not train_info["exists"]:
        raise FileNotFoundError(
            f"Training directory not found: {_train_dir}\n"
            "Please download CIFAKE and place train/ at the project root."
        )

    # ------------------------------------------------------------------
    # Data isolation check — test set must be separate
    # ------------------------------------------------------------------
    from config import DATASET_TEST_DIR
    isolation_ok = verify_no_test_in_train(_train_dir, DATASET_TEST_DIR)
    if not isolation_ok:
        raise RuntimeError(
            "Data isolation check failed: train and test directories overlap. "
            "Training aborted to prevent data leakage."
        )

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    logger.info("Loading datasets...")
    train_ds, val_ds, ds_info = build_train_val_datasets(
        train_dir=_train_dir,
        image_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        batch_size=_batch_size,
        validation_split=_val_split,
        seed=_seed,
    )

    logger.info(
        "Datasets loaded: train=%d samples, val=%d samples",
        ds_info["train_samples"],
        ds_info["val_samples"],
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = build_model(input_shape=INPUT_SHAPE)

    # ------------------------------------------------------------------
    # Train — test_ds is NEVER passed here
    # ------------------------------------------------------------------
    logger.info("Starting training...")
    t_start = time.time()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    t_elapsed = time.time() - t_start
    logger.info("Training completed in %.1f seconds.", t_elapsed)

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to: %s", _save_path)
    model.save(str(_save_path))
    logger.info("Model saved successfully.")

    # ------------------------------------------------------------------
    # Verify saved model loads
    # ------------------------------------------------------------------
    logger.info("Verifying saved model can be loaded...")
    loaded = tf.keras.models.load_model(str(_save_path))
    assert loaded is not None, "Failed to load saved model."
    logger.info("Saved model verified: loads successfully.")

    # ------------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------------
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(str(HISTORY_SAVE_PATH), "w") as f:
        json.dump(history_dict, f, indent=2)
    logger.info("Training history saved to: %s", HISTORY_SAVE_PATH)

    # ------------------------------------------------------------------
    # Result summary
    # ------------------------------------------------------------------
    final_epoch = len(history.history["loss"])
    final_train_loss = history.history["loss"][-1]
    final_train_acc = history.history["accuracy"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    final_val_auc = history.history.get("val_auc", [None])[-1]

    result = {
        "status": "completed",
        "backbone": "MobileNetV2",
        "input_shape": list(INPUT_SHAPE),
        "frozen_base": True,
        "epochs_trained": final_epoch,
        "epochs_requested": _epochs,
        "batch_size": _batch_size,
        "seed": _seed,
        "train_samples": ds_info["train_samples"],
        "val_samples": ds_info["val_samples"],
        "training_time_seconds": round(t_elapsed, 1),
        "model_path": str(_save_path),
        "training_metrics": {
            "final_train_loss": round(final_train_loss, 4),
            "final_train_accuracy": round(final_train_acc, 4),
        },
        "validation_metrics": {
            "final_val_loss": round(final_val_loss, 4),
            "final_val_accuracy": round(final_val_acc, 4),
            "final_val_auc": round(final_val_auc, 4) if final_val_auc else None,
        },
        "note": (
            "These are training/validation metrics from model.fit(). "
            "Final held-out test metrics will be calculated in the "
            "evaluation milestone (Session 3+). "
            "Validation metrics must NOT be reported as test performance."
        ),
    }

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Epochs trained:      %d / %d", final_epoch, _epochs)
    logger.info("  Train loss:          %.4f", final_train_loss)
    logger.info("  Train accuracy:      %.4f", final_train_acc)
    logger.info("  Val loss:            %.4f", final_val_loss)
    logger.info("  Val accuracy:        %.4f", final_val_acc)
    if final_val_auc:
        logger.info("  Val AUC:             %.4f", final_val_auc)
    logger.info("  Model saved to:      %s", _save_path)
    logger.info("=" * 60)
    logger.info(
        "NOTE: Validation metrics above are NOT final test metrics. "
        "Held-out test evaluation is reserved for the evaluation milestone."
    )

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SignalScope baseline model on CIFAKE."
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config.py)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (default: from config.py)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config.py)"
    )
    parser.add_argument(
        "--train-dir", type=str, default=None,
        help="Path to CIFAKE train directory (default: project_root/train/)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = train(
        train_dir=args.train_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print("\nTraining result summary:")
    print(json.dumps(result, indent=2))
