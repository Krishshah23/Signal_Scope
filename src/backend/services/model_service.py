"""
Model Service
--------------
Service layer between Flask routes and model/predict.py.

Responsibilities:
  1. Resolve the model weights path from Flask app config.
  2. Load model/predict.py reliably from the project root.
  3. Call model.predict.predict(image_path).
  4. Return the prediction dict to the route handler.
  5. Translate model-layer exceptions into service-layer exceptions.

This layer never imports TensorFlow directly.
All ML work is encapsulated in model/predict.py.

WORDING RULE:
  Predictions are probabilistic estimates, never certainties.
  Results are described as "likely AI-generated" or "likely real".
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# model_service.py
#     ↓ parent       = services/
#     ↓ parent       = backend/
#     ↓ parent       = src/
#     ↓ parent       = Signal_Scope/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_MODEL_DIR = _PROJECT_ROOT / "model"
_MODEL_PREDICT_FILE = _MODEL_DIR / "predict.py"

logger.info("Project root: %s", _PROJECT_ROOT)
logger.info("Model directory: %s", _MODEL_DIR)
logger.info("Model predict file: %s", _MODEL_PREDICT_FILE)


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
    Raised when inference fails for an unexpected reason.
    """


# ---------------------------------------------------------------------------
# Model module loader
# ---------------------------------------------------------------------------


def _load_model_module() -> ModuleType:
    """
    Load model/predict.py directly from its absolute filesystem path.

    This avoids relying on sys.path or the current working directory,
    both of which can differ between local Flask and Vercel.
    """

    if not _MODEL_PREDICT_FILE.exists():
        raise InferenceError(
            "model/predict.py was not found. "
            f"Expected file: {_MODEL_PREDICT_FILE}"
        )

    try:
        spec = importlib.util.spec_from_file_location(
            "signalscope_model_predict",
            _MODEL_PREDICT_FILE,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not create import specification for {_MODEL_PREDICT_FILE}"
            )

        module = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(module)

        return module

    except Exception as exc:
        logger.exception(
            "Failed to load model/predict.py from %s",
            _MODEL_PREDICT_FILE,
        )

        raise InferenceError(
            "Could not load model/predict.py. "
            f"Details: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ModelService:
    """
    Wraps the ML prediction pipeline for use by Flask route handlers.

    Usage
    -----
    service = ModelService(
        weights_path="model/weights/signalscope_baseline.keras"
    )

    result = service.run(
        image_path="/tmp/uploaded_image.jpg"
    )
    """

    def __init__(self, weights_path: str | None = None) -> None:
        """
        Parameters
        ----------
        weights_path : str, optional
            Path to the trained .keras model file.

            If None, model/predict.py uses its own default.

            Relative paths are resolved from the project root.
        """

        self.weights_path = self._resolve_weights_path(weights_path)

        logger.info(
            "ModelService initialized — weights path: %s",
            self.weights_path,
        )

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_weights_path(
        weights_path: str | None,
    ) -> str | None:
        """
        Convert a relative model path into an absolute project-root path.

        Absolute paths are preserved.
        """

        if not weights_path:
            return None

        path = Path(weights_path)

        if path.is_absolute():
            return str(path)

        return str(_PROJECT_ROOT / path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, image_path: str) -> dict:
        """
        Run inference on a single image file.

        Parameters
        ----------
        image_path : str
            Absolute path to the uploaded image.

        Returns
        -------
        dict
            Prediction result.

        Raises
        ------
        ModelNotReadyError
            If model weights are missing or cannot be loaded.

        InferenceError
            If inference fails.
        """

        logger.info(
            "ModelService.run() called for image: %s",
            image_path,
        )

        # --------------------------------------------------------------
        # Load model/predict.py
        # --------------------------------------------------------------

        model_module = _load_model_module()

        # --------------------------------------------------------------
        # Validate expected public API
        # --------------------------------------------------------------

        required_symbols = (
            "predict",
            "ModelNotTrainedError",
            "InvalidImageError",
        )

        missing_symbols = [
            symbol
            for symbol in required_symbols
            if not hasattr(model_module, symbol)
        ]

        if missing_symbols:
            raise InferenceError(
                "model/predict.py is missing required symbols: "
                + ", ".join(missing_symbols)
            )

        predict = model_module.predict
        ModelNotTrainedError = model_module.ModelNotTrainedError
        InvalidImageError = model_module.InvalidImageError

        # --------------------------------------------------------------
        # Run prediction
        # --------------------------------------------------------------

        try:
            result = predict(
                image_path,
                weights_path=self.weights_path,
            )

            logger.info(
                "Inference complete — verdict: %s confidence: %.4f",
                result["verdict"],
                result["confidence"],
            )

            return result

        # --------------------------------------------------------------
        # Model unavailable
        # --------------------------------------------------------------

        except ModelNotTrainedError as exc:
            logger.error(
                "Model is not ready: %s",
                exc,
            )

            raise ModelNotReadyError(str(exc)) from exc

        # --------------------------------------------------------------
        # Invalid image / missing file
        # --------------------------------------------------------------

        except (FileNotFoundError, InvalidImageError) as exc:
            logger.warning(
                "Invalid image or missing file: %s",
                exc,
            )

            raise InferenceError(str(exc)) from exc

        # --------------------------------------------------------------
        # Unexpected inference failure
        # --------------------------------------------------------------

        except Exception as exc:
            logger.exception(
                "Unexpected inference error for %s",
                image_path,
            )

            raise InferenceError(
                f"Inference failed unexpectedly: {exc}"
            ) from exc
