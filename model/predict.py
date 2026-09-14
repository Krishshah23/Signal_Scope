"""
SignalScope — Model Prediction Interface
-----------------------------------------
Public interface: predict(image_path)

This module is the single contract between the Flask backend and the
TensorFlow/Keras model. The backend ONLY calls predict() — it never
imports TensorFlow directly.

SESSION 2 STATUS:
    Model has been trained on CIFAKE (Session 2).
    Weights are saved to: model/weights/signalscope_baseline.keras
    predict() will load the model on first call and return real predictions.

WORDING RULE:
    Predictions are probabilistic estimates, not certainties.
    Results are always described as "likely AI-generated" or "likely real".
    The system never claims 100% certainty.

PUBLIC CONTRACT:
    predict(image_path: str) -> dict
    {
        "verdict":      "likely AI-generated" | "likely real",
        "confidence":   float,   # 0.0 – 1.0 (certainty of stated verdict)
        "raw_prob":     float,   # raw sigmoid output [0,1] (FAKE probability)
        "heatmap":      None,    # base64 PNG — populated in Grad-CAM milestone
        "explanation":  None,    # text template — populated in Grad-CAM milestone
    }

ARCHITECTURE:
    Model:        MobileNetV2 (frozen) + GlobalAveragePooling2D + Dense(1, sigmoid)
    Input:        128 × 128 × 3 RGB, normalised to [0, 1]
    Output:       sigmoid probability → FAKE=1, REAL=0
    Threshold:    raw_prob >= 0.5 → "likely AI-generated"
                  raw_prob <  0.5 → "likely real"
    Confidence:   raw_prob if FAKE verdict,  (1 - raw_prob) if REAL verdict
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — kept here so the backend can reference them without importing TF
# ---------------------------------------------------------------------------

IMAGE_SIZE: tuple[int, int] = (128, 128)
"""Expected spatial dimensions for model input (height, width)."""

VERDICT_AI: str = "likely AI-generated"
VERDICT_REAL: str = "likely real"
CLASSIFICATION_THRESHOLD: float = 0.5

# Default weights path — relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS_PATH = _PROJECT_ROOT / "model" / "weights" / "signalscope_baseline.keras"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModelNotTrainedError(Exception):
    """
    Raised when predict() is called but model weights cannot be found or loaded.
    """


class InvalidImageError(ValueError):
    """
    Raised when the provided image_path does not point to a readable,
    valid image file.
    """


# ---------------------------------------------------------------------------
# Module-level model cache
# Populated by _load_model() on first predict() call.
# ---------------------------------------------------------------------------
_model = None
_model_weights_path: str | None = None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def predict(image_path: str, weights_path: str | None = None) -> dict:
    """
    Analyse a single image and return a prediction result.

    This is the ONLY public function in this module.
    The Flask backend calls exactly this function — nothing else.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to a readable image file.
        Accepted formats: JPEG, PNG, WebP, BMP.
        The image will be resized to IMAGE_SIZE (128 × 128) internally.
    weights_path : str, optional
        Path to the trained model file. If None, uses the default location:
        model/weights/signalscope_baseline.keras

    Returns
    -------
    dict
        {
            "verdict":     "likely AI-generated" | "likely real",
            "confidence":  float,   # 0.0 – 1.0 (certainty of stated verdict)
            "raw_prob":    float,   # raw sigmoid output
            "heatmap":     None,    # base64 PNG (Grad-CAM milestone, not yet)
            "explanation": None,    # text description (Grad-CAM milestone, not yet)
        }

    Raises
    ------
    ModelNotTrainedError
        If model weights cannot be found at the expected location.
    InvalidImageError
        If the image file cannot be opened or preprocessed.
    FileNotFoundError
        If image_path does not exist.

    Examples
    --------
    >>> result = predict("/path/to/image.jpg")
    >>> result["verdict"]
    'likely AI-generated'
    >>> result["confidence"]
    0.87
    """
    global _model, _model_weights_path

    # Resolve weights path
    _weights = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS_PATH

    # Load model (lazy, cached in module scope)
    if _model is None or (weights_path and weights_path != _model_weights_path):
        _load_model(str(_weights))
        _model_weights_path = str(_weights)

    # Validate and preprocess image
    image_path = str(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    tensor = _preprocess_image(image_path)

    # Run inference
    raw_prob = float(_model.predict(tensor, verbose=0)[0, 0])

    # Build result
    return _build_result(raw_prob)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_model(weights_path: str) -> None:
    """
    Load a trained Keras model from disk and cache it in the module-level
    ``_model`` handle.

    Parameters
    ----------
    weights_path : str
        Path to the saved .keras file or SavedModel directory.

    Raises
    ------
    ModelNotTrainedError
        If weights_path does not exist.
    """
    global _model

    if not os.path.exists(weights_path):
        raise ModelNotTrainedError(
            f"Model weights not found at: {weights_path}\n"
            "Run model/train.py to train the model first.\n"
            "See model/README.md for instructions."
        )

    try:
        import tensorflow as tf
        logger.info("Loading SignalScope model from: %s", weights_path)
        _model = tf.keras.models.load_model(weights_path)
        logger.info("Model loaded successfully.")
    except Exception as exc:
        raise ModelNotTrainedError(
            f"Failed to load model from {weights_path}: {exc}"
        ) from exc


def _preprocess_image(image_path: str):
    """
    Load and preprocess a single image into a model-ready tensor.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Shape (1, 128, 128, 3), dtype float32, values in [0, 1].

    Raises
    ------
    InvalidImageError
        If the image cannot be opened or converted.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "PIL and numpy are required for image preprocessing. "
            "Install with: pip install Pillow numpy"
        ) from exc

    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        arr = np.array(img, dtype="float32") / 255.0
        return arr[np.newaxis, ...]  # shape: (1, 128, 128, 3)
    except Exception as exc:
        raise InvalidImageError(
            f"Could not preprocess image at {image_path}: {exc}"
        ) from exc


def _build_result(raw_prob: float) -> dict:
    """
    Convert a raw sigmoid probability to the public prediction dict.

    Parameters
    ----------
    raw_prob : float
        Output of Dense(1, activation='sigmoid'). Range [0, 1].
        Convention: higher value → more likely AI-generated (FAKE).

    Returns
    -------
    dict
        {
            "verdict":     "likely AI-generated" | "likely real",
            "confidence":  float (0.0 – 1.0),
            "raw_prob":    float,
            "heatmap":     None,
            "explanation": None,
        }

    Notes
    -----
    confidence reflects certainty of the stated verdict:
      - If verdict is FAKE: confidence = raw_prob
      - If verdict is REAL: confidence = 1 - raw_prob
    So a confidence of 0.9 always means 90% certain of the stated verdict,
    regardless of which verdict it is.
    """
    if raw_prob >= CLASSIFICATION_THRESHOLD:
        verdict = VERDICT_AI
        confidence = raw_prob
    else:
        verdict = VERDICT_REAL
        confidence = 1.0 - raw_prob

    return {
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "raw_prob": round(float(raw_prob), 4),
        "heatmap": None,       # base64 PNG — Grad-CAM milestone (future session)
        "explanation": None,   # text description — Grad-CAM milestone (future session)
    }
