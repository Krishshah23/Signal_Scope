"""
Model Service
--------------
This service layer sits between the Flask routes and the raw model/predict.py
interface. Its job is to:

  1. Load the trained model (once weights exist).
  2. Preprocess an uploaded image into the correct tensor format.
  3. Call model.predict.predict(image_path).
  4. Format the raw model output into an API-friendly dict.
  5. Optionally attach a Grad-CAM heatmap (future milestone).

CURRENT STATUS (Session 1 — Foundation):
  The model has not been trained.
  This service raises NotImplementedError on any inference call.
  Import is safe; the error only surfaces when run() is called.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ModelNotReadyError(Exception):
    """
    Raised when inference is requested but the model has not been
    loaded or trained yet.
    """


class ModelService:
    """
    Wraps the ML prediction pipeline for use by Flask route handlers.

    Future usage (once model weights exist):
    ----------------------------------------
    service = ModelService(weights_path="model/weights/signalscope.h5")
    result  = service.run(image_path="/tmp/uploaded.jpg")
    # result → {
    #     "verdict":     "likely AI-generated",
    #     "confidence":  0.87,
    #     "heatmap":     None,   # base64 PNG in future milestone
    #     "explanation": None,   # text template in future milestone
    # }
    """

    def __init__(self, weights_path: str | None = None) -> None:
        self.weights_path = weights_path
        self._model = None  # populated in a future session

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, image_path: str) -> dict:
        """
        Run inference on a single image.

        Parameters
        ----------
        image_path : str
            Absolute path to a saved image file.

        Returns
        -------
        dict
            Prediction result (see class docstring for schema).

        Raises
        ------
        ModelNotReadyError
            Always raised during Session 1 — model not yet trained.
        """
        logger.warning(
            "ModelService.run() called but model is not yet trained. "
            "This will be implemented in a future session."
        )
        raise ModelNotReadyError(
            "The SignalScope model has not been trained yet. "
            "Inference will be available after the model training milestone."
        )

    # ------------------------------------------------------------------
    # Private helpers (stubs for future implementation)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the TensorFlow/Keras model from weights_path. (Future milestone.)"""
        raise NotImplementedError("Model loading not yet implemented.")

    def _preprocess(self, image_path: str):
        """
        Load and preprocess an image to a (1, 128, 128, 3) float32 tensor.
        (Future milestone — will use tf.keras.preprocessing or PIL.)
        """
        raise NotImplementedError("Image preprocessing not yet implemented.")

    def _build_response(self, raw_probability: float) -> dict:
        """
        Convert a raw sigmoid output probability to the API response dict.

        Parameters
        ----------
        raw_probability : float
            Output of Dense(1, sigmoid) — value in [0, 1].
            Convention: values closer to 1 → AI-generated,
                        values closer to 0 → real.

        Returns
        -------
        dict
            {
                "verdict":     "likely AI-generated" | "likely real",
                "confidence":  float (0.0 – 1.0),
                "heatmap":     None,
                "explanation": None,
            }

        NOTE: Predictions are probabilistic, never certain.
        """
        raise NotImplementedError("Response builder not yet implemented.")
