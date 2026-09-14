"""
Prediction route — POST /api/v1/predict/
------------------------------------------
Accepts an uploaded image and returns a real model prediction.

SESSION 4 STATUS:
  ✅ Upload validation — implemented
  ✅ ML model inference — connected via ModelService → model/predict.py
  ✅ Real predictions returned — verdict, confidence, raw_prob
  ⬜ Grad-CAM heatmap — future milestone

API contract:
  POST /api/v1/predict/
  Content-Type: multipart/form-data
  Field: "image" — a JPEG / PNG / WebP / BMP image file (max 16 MB)

  Success response (200 OK):
  {
      "success": true,
      "result": {
          "verdict":      "likely AI-generated" | "likely real",
          "confidence":   0.0 – 1.0,
          "raw_prob":     0.0 – 1.0,
          "heatmap":      null,
          "explanation":  null
      }
  }

  NOTE: Predictions are probabilistic estimates, not certainties.
  The system never claims to "definitely" know whether an image is AI-generated.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

predict_bp = Blueprint("predict", __name__)

# ---------------------------------------------------------------------------
# Ensure model/ is importable when running from src/backend/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MODEL_DIR = _PROJECT_ROOT / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename: str, allowed_extensions: frozenset) -> bool:
    """Return True if the file's extension is in the allowed set."""
    if not filename or "." not in filename:
        return False
    ext = os.path.splitext(filename.lower())[1]
    return ext in allowed_extensions


def _validate_upload(request_obj):
    """
    Validate an incoming multipart upload request.

    Returns
    -------
    (file_storage, None)   on success
    (None, (dict, int))    on failure — (error_body, http_status)
    """
    if "image" not in request_obj.files:
        return None, (
            {
                "success": False,
                "error": "missing_file",
                "message": (
                    "No file was attached. "
                    "Please include an image in the 'image' field of the multipart form."
                ),
            },
            400,
        )

    file = request_obj.files["image"]

    if file.filename == "" or file.filename is None:
        return None, (
            {
                "success": False,
                "error": "empty_filename",
                "message": "The uploaded file has no filename. Please select a valid image file.",
            },
            400,
        )

    allowed_ext = current_app.config["ALLOWED_EXTENSIONS"]
    if not _allowed_file(file.filename, allowed_ext):
        return None, (
            {
                "success": False,
                "error": "unsupported_file_type",
                "message": (
                    f"Unsupported file type. "
                    f"Allowed extensions: {', '.join(sorted(allowed_ext))}."
                ),
            },
            415,
        )

    allowed_mime = current_app.config["ALLOWED_MIME_TYPES"]
    if file.mimetype and file.mimetype not in allowed_mime:
        return None, (
            {
                "success": False,
                "error": "unsupported_mime_type",
                "message": (
                    f"Unsupported MIME type '{file.mimetype}'. "
                    f"Allowed types: {', '.join(sorted(allowed_mime))}."
                ),
            },
            415,
        )

    return file, None


def _get_model_service():
    """
    Lazily construct a ModelService using the weights path from Flask config.

    The weights_path is passed explicitly so tests can override it via
    app.config["MODEL_WEIGHTS_PATH"] without patching module globals.
    """
    from services.model_service import ModelService

    weights_path = current_app.config.get("MODEL_WEIGHTS_PATH")
    return ModelService(weights_path=weights_path)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@predict_bp.post("/predict/")
def predict():
    """
    POST /api/v1/predict/

    Accepts a multipart/form-data request with an 'image' file field.
    Runs the trained MobileNetV2 model and returns a probabilistic verdict.

    Returns
    -------
    200 OK
        Successful inference — real model prediction.
    400 Bad Request
        Missing image or invalid filename.
    415 Unsupported Media Type
        Unsupported file type or MIME type.
    422 Unprocessable Entity
        File passes validation but cannot be read as an image.
    503 Service Unavailable
        Model weights are missing or cannot be loaded.
    500 Internal Server Error
        Unexpected inference failure.
    """
    # --- Validate the upload ---
    file, validation_error = _validate_upload(request)
    if validation_error is not None:
        error_body, status_code = validation_error
        return jsonify(error_body), status_code

    # --- Save to a temporary file for inference ---
    # PIL/TensorFlow need a file path, not an in-memory buffer.
    # The temp file is cleaned up in the finally block regardless of outcome.
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        current_app.logger.debug(
            "Image saved to temp: %s  (%s)", tmp_path, file.filename
        )

        # --- Run real inference ---
        service = _get_model_service()
        result = service.run(tmp_path)

        return jsonify(
            {
                "success": True,
                "result": result,
            }
        ), 200

    except Exception as exc:
        # Import here to avoid circular-import issues at module level
        from services.model_service import ModelNotReadyError, InferenceError

        if isinstance(exc, ModelNotReadyError):
            current_app.logger.error("Model not available: %s", exc)
            return jsonify(
                {
                    "success": False,
                    "error": "model_not_ready",
                    "message": (
                        "The SignalScope model is not available. "
                        "Ensure model/weights/signalscope_baseline.keras exists."
                    ),
                }
            ), 503

        if isinstance(exc, InferenceError):
            msg = str(exc)
            # Distinguish corrupt/unreadable images from real server errors
            if any(k in msg.lower() for k in ("preprocess", "invalid", "cannot", "could not")):
                current_app.logger.warning("Invalid image upload: %s", exc)
                return jsonify(
                    {
                        "success": False,
                        "error": "invalid_image",
                        "message": "The uploaded file could not be read as an image.",
                    }
                ), 422
            current_app.logger.error("Inference error: %s", exc)
            return jsonify(
                {
                    "success": False,
                    "error": "inference_error",
                    "message": "An error occurred during inference. Please try again.",
                }
            ), 500

        current_app.logger.exception("Unexpected error during prediction")
        return jsonify(
            {
                "success": False,
                "error": "server_error",
                "message": "An unexpected server error occurred.",
            }
        ), 500

    finally:
        # Always clean up the temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                current_app.logger.debug("Temp file cleaned up: %s", tmp_path)
            except OSError:
                pass  # Non-fatal — OS will clean up on reboot
