"""
Model Service
--------------
Service layer between Flask routes and model/predict.py.

Responsibilities:
  1. Resolve the model weights path from Flask app config.
  2. Call model.predict.predict(image_path) — the single public inference contract.
  3. Return the prediction dict to the route handler.
  4. Translate model-layer exceptions into service-layer exceptions the route
     can map to appropriate HTTP responses.

This layer never imports TensorFlow directly.
All ML work is encapsulated in model/predict.py.

WORDING RULE:
  Predictions are probabilistic estimates, never certainties.
  Results are described as "likely AI-generated" or "likely real".
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure model/ is importable from the Flask runtime.
# When Flask runs from src/backend/, model/ is two levels up.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Signal_scope/
_MODEL_DIR = _PROJECT_ROOT / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModelNotReadyError(Exception):
    """
    Raised when inference is requested but the model weights are missing
    or cannot be loaded.
    """


class InferenceError(Exception):
    """
    Raised when inference fails for an unexpected reason (corrupt image,
    runtime error, etc.).
    """


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ModelService:
    """
    Wraps the ML prediction pipeline for use by Flask route handlers.

    Usage
    -----
    service = ModelService(weights_path="model/weights/signalscope_baseline.keras")
    result  = service.run(image_path="/tmp/uploaded_image.jpg")
    # result → {
    #     "verdict":      "likely AI-generated" | "likely real",
    #     "confidence":   0.0 – 1.0,
    #     "raw_prob":     0.0 – 1.0,
    #     "heatmap":      None,
    #     "explanation":  None,
    # }
    """

    def __init__(self, weights_path: str | None = None) -> None:
        """
        Parameters
        ----------
        weights_path : str, optional
            Path to the trained .keras model file.
            If None, model/predict.py uses its own default
            (model/weights/signalscope_baseline.keras relative to project root).
        """
        self.weights_path = weights_path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, image_path: str) -> dict:
        """
        Run inference on a single image file.

        Parameters
        ----------
        image_path : str
            Absolute path to a saved image file on disk.
            The file must exist and be a readable image (JPEG/PNG/WebP/BMP).

        Returns
        -------
        dict
            {
                "verdict":      "likely AI-generated" | "likely real",
                "confidence":   float (0.0 – 1.0),
                "raw_prob":     float (0.0 – 1.0),
                "heatmap":      None,
                "explanation":  None,
            }

        Raises
        ------
        ModelNotReadyError
            If the model weights file cannot be found or loaded.
        InferenceError
            If inference fails for any other reason (corrupt image, runtime error).
        """
        try:
            from predict import predict, ModelNotTrainedError, InvalidImageError
        except ImportError as exc:
            raise InferenceError(
                f"Could not import model/predict.py. "
                f"Ensure the project root is on sys.path. Details: {exc}"
            ) from exc

        logger.info("ModelService.run() called for image: %s", image_path)

        try:
            result = predict(image_path, weights_path=self.weights_path)
            logger.info(
                "Inference complete — verdict: %s  confidence: %.4f",
                result["verdict"],
                result["confidence"],
            )
            return result

        except ModelNotTrainedError as exc:
            raise ModelNotReadyError(str(exc)) from exc

        except (FileNotFoundError, InvalidImageError) as exc:
            # Re-raise as InferenceError so routes only need to catch one type
            raise InferenceError(str(exc)) from exc

        except Exception as exc:
            logger.exception("Unexpected inference error for %s", image_path)
            raise InferenceError(
                f"Inference failed unexpectedly: {exc}"
            ) from exc
