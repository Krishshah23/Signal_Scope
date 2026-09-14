"""
SignalScope — Held-Out Test Set Evaluation
-------------------------------------------
Evaluates the trained baseline model against the COMPLETELY HELD-OUT test set.

DATA INTEGRITY GUARANTEES:
  - model.fit() is NEVER called here.
  - Model weights are NEVER modified.
  - Only test/FAKE/ and test/REAL/ are used — never train/ or validation data.
  - The classification threshold is NOT tuned using this data.

Usage:
    .venv/Scripts/python model/evaluate.py

Output files:
    report/evaluation_results.json   — all metrics in JSON format
    report/confusion_matrix.png      — 2×2 confusion matrix plot

Class mapping (must match training convention):
    REAL → 0
    FAKE → 1

Preprocessing (must match training and predict.py):
    - Load image with PIL, convert to RGB
    - Resize to 128×128 using BILINEAR interpolation
    - Normalise to [0, 1] by dividing by 255.0
    - The saved model includes MobileNetV2's preprocess_input internally,
      so NO external preprocess_input call is needed here.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or model/ directly
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODEL_DIR.parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("signalscope.evaluate")

# ---------------------------------------------------------------------------
# Constants (must stay consistent with config.py and predict.py)
# ---------------------------------------------------------------------------
from config import (
    DATASET_TEST_DIR,
    IMAGE_SIZE,
    MODEL_SAVE_PATH,
    CLASS_FAKE,
    CLASS_REAL,
    LABEL_FAKE,
    LABEL_REAL,
    CLASSIFICATION_THRESHOLD,
)

REPORT_DIR: Path = _PROJECT_ROOT / "report"
RESULTS_PATH: Path = REPORT_DIR / "evaluation_results.json"
CONFUSION_MATRIX_PATH: Path = REPORT_DIR / "confusion_matrix.png"

BATCH_SIZE: int = 32  # inference batch size


# ===========================================================================
# Core functions
# ===========================================================================


def load_model(weights_path: Path | None = None):
    """
    Load the trained Keras model from disk.

    Parameters
    ----------
    weights_path : Path, optional
        Path to the .keras model file. Defaults to MODEL_SAVE_PATH.

    Returns
    -------
    tf.keras.Model
        Loaded model (weights frozen — no fit() will ever be called here).

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    import tensorflow as tf  # lazy import — keeps module importable without TF

    path = Path(weights_path) if weights_path else MODEL_SAVE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {path}\n"
            "Run model/train.py to train the model first."
        )

    logger.info("Loading model from: %s", path)
    model = tf.keras.models.load_model(str(path))
    logger.info("Model loaded. Input shape: %s", model.input_shape)
    return model


def load_test_dataset(test_dir: Path | None = None) -> tuple[list[Path], list[int]]:
    """
    Return a list of (image_path, true_label) pairs from the held-out test set.

    Only test/FAKE/ and test/REAL/ are used — never train/ or val data.

    Class mapping:
        REAL → 0
        FAKE → 1

    Parameters
    ----------
    test_dir : Path, optional
        Root of the test dataset. Defaults to DATASET_TEST_DIR.

    Returns
    -------
    (image_paths, true_labels) : tuple
        image_paths  — list of Path objects, one per test image
        true_labels  — list of int (0 or 1), matching image_paths

    Raises
    ------
    FileNotFoundError
        If test_dir or its class sub-directories do not exist.
    """
    root = Path(test_dir) if test_dir else DATASET_TEST_DIR

    for cls in [CLASS_FAKE, CLASS_REAL]:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"Test class directory not found: {cls_dir}\n"
                "Ensure the CIFAKE dataset is placed at the project root."
            )

    # Collect paths and labels — class order is deterministic (sorted)
    image_paths: list[Path] = []
    true_labels: list[int] = []

    class_label_map = {CLASS_REAL: LABEL_REAL, CLASS_FAKE: LABEL_FAKE}

    for cls, label in sorted(class_label_map.items()):
        cls_dir = root / cls
        paths = sorted(
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        image_paths.extend(paths)
        true_labels.extend([label] * len(paths))
        logger.info("  %s: %d images (label=%d)", cls, len(paths), label)

    logger.info("Total test images: %d", len(image_paths))
    return image_paths, true_labels


def _preprocess_single(image_path: Path) -> np.ndarray:
    """
    Preprocess one image into a model-ready (1, H, W, 3) float32 array.

    Matches predict.py's _preprocess_image exactly:
      - PIL open + RGB conversion
      - Resize to IMAGE_SIZE using BILINEAR
      - Normalise to [0, 1] by dividing by 255.0

    NOTE: The saved model already includes mobilenet_v2.preprocess_input
    internally, so we only normalise to [0, 1] here — NOT to [-1, 1].
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
    arr = np.array(img, dtype="float32") / 255.0
    return arr[np.newaxis, ...]  # shape: (1, 128, 128, 3)


def collect_predictions(
    model,
    image_paths: list[Path],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Run inference on all test images and return raw sigmoid probabilities.

    This function NEVER calls model.fit() or modifies weights.

    Parameters
    ----------
    model : tf.keras.Model
        The loaded trained model.
    image_paths : list[Path]
        Paths to all test images.
    batch_size : int
        Number of images processed per model.predict() call.

    Returns
    -------
    np.ndarray
        Shape (N,) — raw sigmoid probability for each image.
        A value close to 1.0 means the model considers the image likely FAKE.
        A value close to 0.0 means the model considers it likely REAL.
    """
    n = len(image_paths)
    raw_probs = np.zeros(n, dtype="float32")

    logger.info("Running inference on %d images (batch_size=%d)...", n, batch_size)
    t0 = time.time()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_paths = image_paths[start:end]

        # Build batch tensor
        batch = np.concatenate(
            [_preprocess_single(p) for p in batch_paths], axis=0
        )  # shape: (batch_size, 128, 128, 3)

        preds = model.predict(batch, verbose=0)  # shape: (batch_size, 1)
        raw_probs[start:end] = preds[:, 0]

        if (start // batch_size + 1) % 50 == 0 or end == n:
            elapsed = time.time() - t0
            pct = end / n * 100
            logger.info("  %d / %d images (%.1f%%) — %.1fs elapsed", end, n, pct, elapsed)

    elapsed = time.time() - t0
    logger.info("Inference complete in %.1fs (%.1fms/image)", elapsed, elapsed / n * 1000)
    return raw_probs


def calculate_metrics(
    true_labels: list[int],
    raw_probs: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> dict:
    """
    Compute all evaluation metrics from true labels and model probabilities.

    Uses ONLY the held-out test labels and model outputs — no training data.

    Metrics computed:
      - accuracy
      - ROC-AUC   (uses raw probabilities, NOT thresholded predictions)
      - macro-F1  (uses thresholded predictions)
      - precision (macro average)
      - recall    (macro average)
      - confusion matrix  [[TN, FP], [FN, TP]]
                           = [[REAL→REAL, REAL→FAKE],
                              [FAKE→REAL, FAKE→FAKE]]

    Class order in confusion matrix:
        Row 0 / Col 0 = REAL (label 0)
        Row 1 / Col 1 = FAKE (label 1)

    Parameters
    ----------
    true_labels : list[int]
        Ground-truth labels (0=REAL, 1=FAKE) for every test image.
    raw_probs : np.ndarray
        Raw sigmoid output probabilities, shape (N,).
    threshold : float
        Decision threshold. Default: 0.5 (from config).

    Returns
    -------
    dict
        All metrics, confusion matrix, and per-class breakdowns.
    """
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )

    y_true = np.array(true_labels, dtype="int32")
    y_pred = (raw_probs >= threshold).astype("int32")

    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, raw_probs))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[LABEL_REAL, LABEL_FAKE])
    # cm[i, j] = number of samples with true label i predicted as j
    # cm[0,0] = REAL correctly predicted as REAL  (True Negative  for FAKE)
    # cm[0,1] = REAL incorrectly predicted as FAKE (False Positive for FAKE)
    # cm[1,0] = FAKE incorrectly predicted as REAL (False Negative for FAKE)
    # cm[1,1] = FAKE correctly predicted as FAKE  (True Positive  for FAKE)

    tn, fp, fn, tp = cm.ravel()

    n_total = len(y_true)
    n_correct = int(np.sum(y_true == y_pred))
    n_incorrect = n_total - n_correct

    metrics = {
        "accuracy":   round(acc, 6),
        "roc_auc":    round(auc, 6),
        "macro_f1":   round(macro_f1, 6),
        "precision":  round(precision, 6),
        "recall":     round(recall, 6),
        "threshold":  threshold,
    }

    logger.info("=" * 50)
    logger.info("HELD-OUT TEST SET METRICS")
    logger.info("=" * 50)
    logger.info("  Accuracy:   %.4f  (%d / %d correct)", acc, n_correct, n_total)
    logger.info("  ROC-AUC:    %.4f", auc)
    logger.info("  Macro-F1:   %.4f", macro_f1)
    logger.info("  Precision:  %.4f  (macro)", precision)
    logger.info("  Recall:     %.4f  (macro)", recall)
    logger.info("-" * 50)
    logger.info("  Confusion matrix (rows=actual, cols=predicted):")
    logger.info("               Pred REAL   Pred FAKE")
    logger.info("  Actual REAL     %5d       %5d", tn, fp)
    logger.info("  Actual FAKE     %5d       %5d", fn, tp)
    logger.info("=" * 50)

    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "counts": {
            "total":     n_total,
            "correct":   n_correct,
            "incorrect": n_incorrect,
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }


def save_results(
    metrics_dict: dict,
    model_path: Path,
    test_dir: Path,
    elapsed_seconds: float,
    results_path: Path = RESULTS_PATH,
) -> None:
    """
    Write all evaluation results to report/evaluation_results.json.

    Parameters
    ----------
    metrics_dict : dict
        Output of calculate_metrics().
    model_path : Path
        Path to the .keras model that was evaluated.
    test_dir : Path
        Path to the test dataset root.
    elapsed_seconds : float
        Total wall-clock evaluation time.
    results_path : Path
        Where to write the JSON file.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    m = metrics_dict["metrics"]
    cm = metrics_dict["confusion_matrix"]
    counts = metrics_dict["counts"]

    output = {
        "status": "completed",
        "evaluation_time_seconds": round(elapsed_seconds, 1),
        "model": {
            "name": "SignalScope MobileNetV2 baseline",
            "path": str(model_path),
            "input_shape": [IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
            "backbone": "MobileNetV2 (frozen, ImageNet pretrained)",
            "head": "GlobalAveragePooling2D → Dense(1, sigmoid)",
            "threshold": m["threshold"],
        },
        "dataset": {
            "name": "CIFAKE",
            "split": "held-out test",
            "path": str(test_dir),
            "total_samples": counts["total"],
            "real_samples": counts["tn"] + counts["fp"],
            "fake_samples": counts["fn"] + counts["tp"],
            "class_mapping": {
                CLASS_REAL: LABEL_REAL,
                CLASS_FAKE: LABEL_FAKE,
            },
        },
        "metrics": {
            "accuracy":  m["accuracy"],
            "roc_auc":   m["roc_auc"],
            "macro_f1":  m["macro_f1"],
            "precision": m["precision"],
            "recall":    m["recall"],
        },
        "confusion_matrix": {
            "matrix": cm,
            "row_labels": [CLASS_REAL, CLASS_FAKE],
            "col_labels": [CLASS_REAL, CLASS_FAKE],
            "interpretation": (
                "rows=actual class, cols=predicted class. "
                f"cm[0][0]=REAL→REAL (TN={counts['tn']}), "
                f"cm[0][1]=REAL→FAKE (FP={counts['fp']}), "
                f"cm[1][0]=FAKE→REAL (FN={counts['fn']}), "
                f"cm[1][1]=FAKE→FAKE (TP={counts['tp']})"
            ),
        },
        "prediction_counts": counts,
        "notes": [
            "These are final HELD-OUT TEST metrics. The test set was never used for "
            "training, validation, hyperparameter tuning, or model selection.",
            "model.fit() was NOT called during evaluation.",
            "Model weights were NOT modified.",
            "Session 2 validation metrics are separate and are NOT reported here.",
        ],
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("Results written to: %s", results_path)


def save_confusion_matrix(
    cm: list[list[int]],
    output_path: Path = CONFUSION_MATRIX_PATH,
    accuracy: float | None = None,
) -> None:
    """
    Save a clean 2×2 confusion matrix image.

    Rows = actual class, Columns = predicted class.
    Class order: [REAL (0), FAKE (1)].

    Parameters
    ----------
    cm : list[list[int]]
        2×2 confusion matrix as returned by sklearn (nested list).
    output_path : Path
        Where to save the PNG.
    accuracy : float, optional
        Overall accuracy to include in the title.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe for all environments
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm)
    class_labels = [CLASS_REAL, CLASS_FAKE]  # ["REAL", "FAKE"]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Heat-map — use Blues colormap
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Axis ticks
    tick_positions = [0, 1]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(class_labels, fontsize=13)
    ax.set_yticklabels(class_labels, fontsize=13)

    ax.set_xlabel("Predicted label", fontsize=13, labelpad=10)
    ax.set_ylabel("Actual label", fontsize=13, labelpad=10)

    # Title
    title = "SignalScope — Held-Out Test Confusion Matrix"
    if accuracy is not None:
        title += f"\n(Accuracy = {accuracy:.4f})"
    ax.set_title(title, fontsize=13, pad=14)

    # Annotate cells with counts and row-normalised percentages
    total = cm_arr.sum()
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = cm_arr[i, j]
            row_total = cm_arr[i, :].sum()
            pct = count / row_total * 100 if row_total > 0 else 0.0
            color = "white" if count > thresh else "black"
            ax.text(
                j, i,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=12, color=color, fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix image saved to: %s", output_path)


# ===========================================================================
# Main entry point
# ===========================================================================


def evaluate(
    weights_path: Path | None = None,
    test_dir: Path | None = None,
    results_path: Path = RESULTS_PATH,
    cm_path: Path = CONFUSION_MATRIX_PATH,
) -> dict:
    """
    Full evaluation pipeline.

    Steps:
      1. Load the trained model (NO fit()).
      2. Load the held-out test dataset (test/ only).
      3. Run inference on all test images.
      4. Compute metrics (accuracy, ROC-AUC, macro-F1, precision, recall, CM).
      5. Save JSON results to report/evaluation_results.json.
      6. Save confusion matrix PNG to report/confusion_matrix.png.

    Returns
    -------
    dict
        Full results dictionary (same content as the JSON file).
    """
    _weights = Path(weights_path) if weights_path else MODEL_SAVE_PATH
    _test_dir = Path(test_dir) if test_dir else DATASET_TEST_DIR

    logger.info("=" * 60)
    logger.info("SignalScope — Held-Out Test Evaluation")
    logger.info("=" * 60)
    logger.info("Model:     %s", _weights)
    logger.info("Test dir:  %s", _test_dir)

    t_start = time.time()

    # Step 1 — Load model
    model = load_model(_weights)

    # Step 2 — Load test dataset
    logger.info("Loading test dataset...")
    image_paths, true_labels = load_test_dataset(_test_dir)

    # Sanity check — must never include train data
    train_root = _PROJECT_ROOT / "train"
    for p in image_paths[:5]:
        assert str(train_root) not in str(p), (
            f"CRITICAL: Training path detected in test dataset! {p}"
        )

    # Step 3 — Collect predictions (no fit, no weight changes)
    raw_probs = collect_predictions(model, image_paths, batch_size=BATCH_SIZE)

    # Step 4 — Calculate metrics
    metrics_dict = calculate_metrics(true_labels, raw_probs, CLASSIFICATION_THRESHOLD)

    t_elapsed = time.time() - t_start

    # Step 5 — Save JSON
    save_results(metrics_dict, _weights, _test_dir, t_elapsed, results_path)

    # Step 6 — Save confusion matrix PNG
    save_confusion_matrix(
        metrics_dict["confusion_matrix"],
        output_path=cm_path,
        accuracy=metrics_dict["metrics"]["accuracy"],
    )

    logger.info("Evaluation complete in %.1fs", t_elapsed)
    return metrics_dict


if __name__ == "__main__":
    result = evaluate()
    print("\nSummary:")
    m = result["metrics"]
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"  Macro-F1  : {m['macro_f1']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    cm = result["confusion_matrix"]
    print(f"\nConfusion matrix (rows=actual, cols=predicted) [REAL, FAKE]:")
    print(f"  [[{cm[0][0]:6d}, {cm[0][1]:6d}],    <- Actual REAL")
    print(f"   [{cm[1][0]:6d}, {cm[1][1]:6d}]]    <- Actual FAKE")
