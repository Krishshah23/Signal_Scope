"""
SignalScope — CIFAKE Dataset Preparation
-----------------------------------------
Loads, preprocesses, and splits the CIFAKE dataset for training.

Dataset structure expected on disk:
    <project_root>/
        train/
            FAKE/   ← AI-generated images (label = 1)
            REAL/   ← Real images         (label = 0)
        test/
            FAKE/
            REAL/

DATA ISOLATION GUARANTEE:
    The test/ directory is NEVER passed to model.fit().
    Validation data is created ONLY by splitting the training data.
    This prevents data leakage and preserves the held-out test set
    for the later evaluation milestone.

Split strategy:
    train/  ──────────────────────────────────────────────
              85% training data   |  15% validation data
    test/   (untouched until evaluation milestone)

Label convention:
    FAKE → 1  (AI-generated)
    REAL → 0  (real)

Preprocessing:
    - Load JPEG images
    - Resize to 128 × 128
    - Normalise pixel values to [0, 1] (divide by 255)
    - Batch and prefetch for efficient training
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _require_tensorflow():
    """Import TensorFlow lazily so the module can be imported for tests
    even if TF is not installed in the current Python environment."""
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for dataset loading. "
            "Install it with: pip install tensorflow==2.18.0"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_train_val_datasets(
    train_dir: str | Path,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 32,
    validation_split: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Build TensorFlow training and validation datasets from the CIFAKE
    training directory.

    The test directory is NEVER passed to this function.
    Validation data is split from training data only.

    Parameters
    ----------
    train_dir : str | Path
        Path to the CIFAKE training directory.
        Must contain subdirectories FAKE/ and REAL/.
    image_size : tuple[int, int]
        Target (height, width) for image resizing. Default: (128, 128).
    batch_size : int
        Number of samples per batch. Default: 32.
    validation_split : float
        Fraction of training data to use for validation. Default: 0.15.
        Must be in (0.0, 1.0).
    seed : int
        Random seed for reproducible train/val split. Default: 42.

    Returns
    -------
    tuple[tf.data.Dataset, tf.data.Dataset, dict]
        (train_dataset, val_dataset, info_dict)

        info_dict keys:
            "train_samples"    : int
            "val_samples"      : int
            "class_names"      : list[str]  (sorted alphabetically)
            "label_map"        : dict[str, int]
            "image_size"       : tuple[int, int]
            "batch_size"       : int
            "validation_split" : float
            "seed"             : int

    Raises
    ------
    FileNotFoundError
        If train_dir or its class subdirectories do not exist.
    ValueError
        If the directory structure is not as expected.
    """
    tf = _require_tensorflow()

    train_dir = Path(train_dir)
    _validate_dataset_dir(train_dir, label="train")

    logger.info("Building train/val datasets from: %s", train_dir)
    logger.info("Image size: %s  |  Batch: %d  |  Val split: %.0f%%",
                image_size, batch_size, validation_split * 100)

    # -----------------------------------------------------------------------
    # Load training split using Keras image_dataset_from_directory.
    # class_names will be sorted alphabetically: ["FAKE", "REAL"]
    # class_mode="binary" is not available here — we use label_mode="int"
    # and then remap labels so FAKE=1, REAL=0.
    # -----------------------------------------------------------------------

    # TF will sort class names alphabetically: FAKE=0, REAL=1 by default.
    # We explicitly remap after loading.
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        labels="inferred",
        label_mode="int",
        class_names=["FAKE", "REAL"],  # explicit order: FAKE=0, REAL=1 (TF default)
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training",
        interpolation="bilinear",
    )

    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        labels="inferred",
        label_mode="int",
        class_names=["FAKE", "REAL"],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
        interpolation="bilinear",
    )

    # -----------------------------------------------------------------------
    # Remap labels: TF assigns class index by sorted name order.
    # Sorted: ["FAKE", "REAL"] → FAKE=0, REAL=1 (TF default)
    # We want:                   FAKE=1, REAL=0  (our convention)
    # So: new_label = 1 - original_label
    # -----------------------------------------------------------------------
    def remap_labels(images, labels):
        """Invert labels: FAKE→1, REAL→0."""
        return images, tf.cast(1 - labels, tf.int32)

    # -----------------------------------------------------------------------
    # Preprocessing: normalise pixels to [0, 1]
    # -----------------------------------------------------------------------
    def preprocess(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    # -----------------------------------------------------------------------
    # Data augmentation — applied to TRAINING data ONLY, not validation
    # -----------------------------------------------------------------------
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
    ], name="augmentation")

    def augment_train(images, labels):
        images = augment(images, training=True)
        return images, labels

    # Build the pipeline
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        raw_train_ds
        .map(remap_labels, num_parallel_calls=AUTOTUNE)
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .map(augment_train, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        raw_val_ds
        .map(remap_labels, num_parallel_calls=AUTOTUNE)
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # Derive sample counts from the known total and split ratio
    # (avoids an expensive full dataset iteration just for counting)
    total_files = sum(
        len(list((Path(train_dir) / cls).iterdir()))
        for cls in ["FAKE", "REAL"]
        if (Path(train_dir) / cls).is_dir()
    )
    train_samples = int(total_files * (1.0 - validation_split))
    val_samples = total_files - train_samples

    info = {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "class_names": ["REAL", "FAKE"],   # our label convention: 0=REAL, 1=FAKE
        "label_map": {"REAL": 0, "FAKE": 1},
        "image_size": image_size,
        "batch_size": batch_size,
        "validation_split": validation_split,
        "seed": seed,
    }

    logger.info(
        "Dataset built: train=%d samples, val=%d samples",
        train_samples, val_samples,
    )

    return train_ds, val_ds, info


def inspect_dataset(
    data_dir: str | Path,
    split_name: str = "train",
) -> dict:
    """
    Inspect a dataset directory without building a training pipeline.
    Used for dataset validation and testing.

    Parameters
    ----------
    data_dir : str | Path
        Path to the dataset directory (train/ or test/).
    split_name : str
        Label for logging purposes.

    Returns
    -------
    dict
        {
            "dir": str,
            "exists": bool,
            "classes": list[str],
            "counts": dict[str, int],
            "total": int,
        }
    """
    data_dir = Path(data_dir)
    result: dict = {
        "dir": str(data_dir),
        "exists": data_dir.is_dir(),
        "classes": [],
        "counts": {},
        "total": 0,
    }

    if not data_dir.is_dir():
        logger.warning("Dataset directory not found: %s", data_dir)
        return result

    # Discover class subdirectories
    classes = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    result["classes"] = classes

    # Count images per class
    for cls in classes:
        cls_dir = data_dir / cls
        image_files = [
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        result["counts"][cls] = len(image_files)
        result["total"] += len(image_files)

    logger.info(
        "%s dataset — dir: %s | classes: %s | counts: %s | total: %d",
        split_name, data_dir, classes, result["counts"], result["total"],
    )
    return result


def verify_no_test_in_train(train_dir: str | Path, test_dir: str | Path) -> bool:
    """
    Verify that the test directory path is completely separate from
    the training directory path.

    This is a sanity check to help ensure no test data leaks into training.

    Parameters
    ----------
    train_dir : str | Path
    test_dir : str | Path

    Returns
    -------
    bool
        True if paths are distinct (no overlap), False otherwise.
    """
    train_path = Path(train_dir).resolve()
    test_path = Path(test_dir).resolve()

    if train_path == test_path:
        logger.error(
            "CRITICAL: train and test directories are identical! (%s)",
            train_path,
        )
        return False

    # Check neither is a subdirectory of the other
    try:
        test_path.relative_to(train_path)
        logger.error("CRITICAL: test directory is inside train directory!")
        return False
    except ValueError:
        pass

    try:
        train_path.relative_to(test_path)
        logger.error("CRITICAL: train directory is inside test directory!")
        return False
    except ValueError:
        pass

    logger.info("Data isolation check PASSED: train and test are separate paths.")
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_dataset_dir(data_dir: Path, label: str = "") -> None:
    """Raise FileNotFoundError if the directory or its class subdirs are missing."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset {label} directory not found: {data_dir}\n"
            "Please download CIFAKE and place it at the project root.\n"
            "See model/README.md for instructions."
        )

    for cls in ["FAKE", "REAL"]:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class directory not found: {cls_dir}\n"
                f"The CIFAKE {label} directory must contain FAKE/ and REAL/ subdirectories."
            )


# ---------------------------------------------------------------------------
# Quick inspection entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import DATASET_TRAIN_DIR, DATASET_TEST_DIR

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("CIFAKE Dataset Inspection")
    print("=" * 50)

    train_info = inspect_dataset(DATASET_TRAIN_DIR, "train")
    print(f"\nTrain directory: {train_info['dir']}")
    print(f"  Exists: {train_info['exists']}")
    print(f"  Classes: {train_info['classes']}")
    print(f"  Counts: {train_info['counts']}")
    print(f"  Total: {train_info['total']}")

    test_info = inspect_dataset(DATASET_TEST_DIR, "test")
    print(f"\nTest directory: {test_info['dir']}")
    print(f"  Exists: {test_info['exists']}")
    print(f"  Classes: {test_info['classes']}")
    print(f"  Counts: {test_info['counts']}")
    print(f"  Total: {test_info['total']}")

    isolation_ok = verify_no_test_in_train(DATASET_TRAIN_DIR, DATASET_TEST_DIR)
    print(f"\nData isolation check: {'PASSED' if isolation_ok else 'FAILED'}")
