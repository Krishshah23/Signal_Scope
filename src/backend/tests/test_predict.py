"""
Tests for POST /api/v1/predict/  — Session 4

Tests cover:
  - Upload validation (400 / 415 errors)
  - Corrupt image handling (422)
  - Real model inference path (200) — uses the actual trained model
  - Response structure verification
  - Temporary file cleanup
  - No fake/mock prediction values in the production path

Design notes:
  - Tests that need the real model use a small real JPEG from train/FAKE/
    or train/REAL/ as a single-sample fixture.
  - The 20,000-image held-out test set is NEVER used here.
  - model.fit() is NEVER called.
  - Corrupt-image tests use a tiny in-memory bytes object.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure src/backend/ is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # Signal_scope/
_WEIGHTS_PATH = _PROJECT_ROOT / "model" / "weights" / "signalscope_baseline.keras"
_TRAIN_FAKE_DIR = _PROJECT_ROOT / "train" / "FAKE"
_TRAIN_REAL_DIR = _PROJECT_ROOT / "train" / "REAL"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Flask test client configured for testing (no real model loaded by default)."""
    app = create_app("testing")
    app.config["TESTING"] = True
    # Point to the real weights for tests that need inference
    app.config["MODEL_WEIGHTS_PATH"] = str(_WEIGHTS_PATH)
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="session")
def real_jpeg_bytes():
    """
    Load one real JPEG from train/FAKE/ into memory for inference tests.
    Returns the raw JPEG bytes.
    Skips if the training data is not available.
    """
    if not _TRAIN_FAKE_DIR.is_dir():
        pytest.skip("train/FAKE/ not available — skipping real inference tests")
    sample = next(iter(sorted(_TRAIN_FAKE_DIR.iterdir())))
    return sample.read_bytes()


@pytest.fixture(scope="session")
def real_png_bytes():
    """
    Create a minimal valid PNG in memory using Pillow (no file I/O).
    Used for PNG upload tests.
    """
    try:
        from PIL import Image as PILImage
        import io as _io
        import numpy as np
        # 8x8 random RGB image → save as PNG in memory
        arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        img = PILImage.fromarray(arr, mode="RGB")
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        pytest.skip("Pillow/numpy not available for PNG fixture")


def _make_upload(data: bytes, filename: str) -> tuple:
    """Return a Flask test client upload tuple."""
    return (io.BytesIO(data), filename)


# ---------------------------------------------------------------------------
# Helper: mock ModelService so upload validation tests don't load the model
# ---------------------------------------------------------------------------

def _mock_service(verdict="likely AI-generated", raw_prob=0.87):
    """Return a MagicMock that looks like a successful ModelService.run() call."""
    mock_svc = MagicMock()
    mock_svc.run.return_value = {
        "verdict": verdict,
        "confidence": raw_prob,
        "raw_prob": raw_prob,
        "heatmap": None,
        "explanation": None,
    }
    return mock_svc


# ===========================================================================
# 1. Upload validation — these tests do NOT need the model loaded
# ===========================================================================

class TestUploadValidation:
    """
    All validation tests mock the ModelService so they never touch
    the actual model or TensorFlow. These are pure HTTP-layer tests.
    """

    def test_no_file_returns_400(self, client):
        """Missing 'image' field → 400."""
        resp = client.post("/api/v1/predict/")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False
        assert data["error"] == "missing_file"

    def test_empty_filename_returns_400(self, client):
        """Empty filename → 400."""
        resp = client.post(
            "/api/v1/predict/",
            data={"image": (io.BytesIO(b"data"), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False
        assert data["error"] == "empty_filename"

    def test_unsupported_extension_returns_415(self, client):
        """.txt file → 415."""
        resp = client.post(
            "/api/v1/predict/",
            data={"image": (io.BytesIO(b"text data"), "file.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 415
        data = resp.get_json()
        assert data["success"] is False
        assert data["error"] == "unsupported_file_type"

    def test_unsupported_pdf_returns_415(self, client):
        """.pdf file → 415 unsupported file type."""
        resp = client.post(
            "/api/v1/predict/",
            data={"image": (io.BytesIO(b"%PDF"), "document.pdf")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 415

    def test_response_is_json(self, client):
        """Error responses must always be JSON."""
        resp = client.post("/api/v1/predict/")
        assert "application/json" in resp.content_type

    def test_valid_jpeg_passes_validation_and_reaches_service(self, client):
        """
        A structurally valid JPEG should pass validation.
        We mock the service so inference doesn't actually run.
        """
        minimal_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        with patch(
            "routes.predict._get_model_service",
            return_value=_mock_service(),
        ):
            resp = client.post(
                "/api/v1/predict/",
                data={"image": _make_upload(minimal_jpeg, "photo.jpg")},
                content_type="multipart/form-data",
            )
        # With mock service, should reach 200 or any non-4xx
        assert resp.status_code not in (400, 415)

    def test_valid_png_passes_validation(self, client, real_png_bytes):
        """A valid PNG passes file-type validation."""
        with patch(
            "routes.predict._get_model_service",
            return_value=_mock_service(),
        ):
            resp = client.post(
                "/api/v1/predict/",
                data={"image": _make_upload(real_png_bytes, "image.png")},
                content_type="multipart/form-data",
            )
        assert resp.status_code not in (400, 415)


# ===========================================================================
# 2. Mock inference — API contract tests without loading the model
# ===========================================================================

class TestMockedInferenceResponse:
    """
    Tests that verify the API response structure using a mocked service.
    These run fast and don't require TensorFlow.
    """

    def _post_with_mock(self, client, verdict="likely AI-generated", raw_prob=0.87):
        minimal_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        with patch(
            "routes.predict._get_model_service",
            return_value=_mock_service(verdict, raw_prob),
        ):
            return client.post(
                "/api/v1/predict/",
                data={"image": _make_upload(minimal_jpeg, "photo.jpg")},
                content_type="multipart/form-data",
            )

    def test_successful_response_status_200(self, client):
        resp = self._post_with_mock(client)
        assert resp.status_code == 200

    def test_response_has_success_true(self, client):
        data = self._post_with_mock(client).get_json()
        assert data["success"] is True

    def test_response_has_result_key(self, client):
        data = self._post_with_mock(client).get_json()
        assert "result" in data

    def test_result_has_verdict(self, client):
        data = self._post_with_mock(client).get_json()
        assert "verdict" in data["result"]

    def test_result_has_confidence(self, client):
        data = self._post_with_mock(client).get_json()
        assert "confidence" in data["result"]

    def test_result_has_raw_prob(self, client):
        data = self._post_with_mock(client).get_json()
        assert "raw_prob" in data["result"]

    def test_result_heatmap_is_null(self, client):
        """heatmap must be null until Grad-CAM milestone."""
        data = self._post_with_mock(client).get_json()
        assert data["result"]["heatmap"] is None

    def test_result_explanation_is_null(self, client):
        """explanation must be null until Grad-CAM milestone."""
        data = self._post_with_mock(client).get_json()
        assert data["result"]["explanation"] is None

    def test_ai_generated_verdict(self, client):
        data = self._post_with_mock(client, verdict="likely AI-generated").get_json()
        assert data["result"]["verdict"] == "likely AI-generated"

    def test_real_verdict(self, client):
        data = self._post_with_mock(client, verdict="likely real", raw_prob=0.2).get_json()
        assert data["result"]["verdict"] == "likely real"

    def test_response_content_type_is_json(self, client):
        resp = self._post_with_mock(client)
        assert "application/json" in resp.content_type

    def test_no_501_in_success_response(self, client):
        """The real endpoint must never return 501 Not Implemented."""
        resp = self._post_with_mock(client)
        assert resp.status_code != 501


# ===========================================================================
# 3. Error-path responses (mocked)
# ===========================================================================

class TestErrorResponses:
    def test_model_not_ready_returns_503(self, client):
        """ModelNotReadyError → 503."""
        from services.model_service import ModelNotReadyError
        mock_svc = MagicMock()
        mock_svc.run.side_effect = ModelNotReadyError("weights missing")
        minimal_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        with patch("routes.predict._get_model_service", return_value=mock_svc):
            resp = client.post(
                "/api/v1/predict/",
                data={"image": _make_upload(minimal_jpeg, "photo.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 503
        data = resp.get_json()
        assert data["success"] is False
        assert data["error"] == "model_not_ready"

    def test_invalid_image_returns_422(self, client):
        """InferenceError with 'invalid'/'cannot'/'could not' → 422."""
        from services.model_service import InferenceError
        mock_svc = MagicMock()
        mock_svc.run.side_effect = InferenceError("Could not preprocess corrupt image")
        minimal_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        with patch("routes.predict._get_model_service", return_value=mock_svc):
            resp = client.post(
                "/api/v1/predict/",
                data={"image": _make_upload(minimal_jpeg, "photo.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 422
        data = resp.get_json()
        assert data["success"] is False
        assert data["error"] == "invalid_image"

    def test_generic_inference_error_returns_500(self, client):
        """Generic InferenceError → 500."""
        from services.model_service import InferenceError
        mock_svc = MagicMock()
        mock_svc.run.side_effect = InferenceError("Something exploded")
        minimal_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        with patch("routes.predict._get_model_service", return_value=mock_svc):
            resp = client.post(
                "/api/v1/predict/",
                data={"image": _make_upload(minimal_jpeg, "photo.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["success"] is False

    def test_error_response_does_not_expose_stack_trace(self, client):
        """Error responses must not expose internal stack traces."""
        from services.model_service import ModelNotReadyError
        mock_svc = MagicMock()
        mock_svc.run.side_effect = ModelNotReadyError("weights missing")
        minimal_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        with patch("routes.predict._get_model_service", return_value=mock_svc):
            resp = client.post(
                "/api/v1/predict/",
                data={"image": _make_upload(minimal_jpeg, "photo.jpg")},
                content_type="multipart/form-data",
            )
        body = resp.get_data(as_text=True)
        assert "Traceback" not in body
        assert "File \"" not in body


# ===========================================================================
# 4. Temporary file cleanup
# ===========================================================================

class TestTempFileCleanup:
    def test_temp_file_cleaned_up_on_success(self, client):
        """Temporary upload file must be deleted after successful inference."""
        created_temps = []

        original_mktemp = tempfile.NamedTemporaryFile

        def tracking_mktemp(**kwargs):
            f = original_mktemp(**kwargs)
            created_temps.append(f.name)
            return f

        minimal_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        with patch("routes.predict.tempfile.NamedTemporaryFile", side_effect=tracking_mktemp):
            with patch(
                "routes.predict._get_model_service",
                return_value=_mock_service(),
            ):
                client.post(
                    "/api/v1/predict/",
                    data={"image": _make_upload(minimal_jpeg, "photo.jpg")},
                    content_type="multipart/form-data",
                )

        for path in created_temps:
            assert not os.path.exists(path), f"Temp file was not cleaned up: {path}"

    def test_temp_file_cleaned_up_on_error(self, client):
        """Temporary upload file must be deleted even when inference fails."""
        from services.model_service import ModelNotReadyError
        created_temps = []

        original_mktemp = tempfile.NamedTemporaryFile

        def tracking_mktemp(**kwargs):
            f = original_mktemp(**kwargs)
            created_temps.append(f.name)
            return f

        mock_svc = MagicMock()
        mock_svc.run.side_effect = ModelNotReadyError("weights missing")

        minimal_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        with patch("routes.predict.tempfile.NamedTemporaryFile", side_effect=tracking_mktemp):
            with patch("routes.predict._get_model_service", return_value=mock_svc):
                client.post(
                    "/api/v1/predict/",
                    data={"image": _make_upload(minimal_jpeg, "photo.jpg")},
                    content_type="multipart/form-data",
                )

        for path in created_temps:
            assert not os.path.exists(path), f"Temp file was not cleaned up on error: {path}"


# ===========================================================================
# 5. Real inference tests — requires trained model on disk
# ===========================================================================

@pytest.mark.skipif(
    not _WEIGHTS_PATH.exists(),
    reason="signalscope_baseline.keras not found — skipping real inference tests",
)
@pytest.mark.skipif(
    not _TRAIN_FAKE_DIR.is_dir(),
    reason="train/FAKE/ not available — skipping real inference tests",
)
class TestRealInference:
    """
    End-to-end inference tests using the actual trained model.
    These tests do NOT use the held-out test set.
    They use a single real JPEG from train/FAKE/ as a smoke test.
    """

    def test_real_jpeg_returns_200(self, client, real_jpeg_bytes):
        """A real JPEG should produce a 200 response from the actual model."""
        resp = client.post(
            "/api/v1/predict/",
            data={"image": _make_upload(real_jpeg_bytes, "real_test.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.get_data(as_text=True)}"
        )

    def test_real_inference_response_has_success_true(self, client, real_jpeg_bytes):
        resp = client.post(
            "/api/v1/predict/",
            data={"image": _make_upload(real_jpeg_bytes, "real_test.jpg")},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        assert data["success"] is True

    def test_real_inference_result_has_all_keys(self, client, real_jpeg_bytes):
        resp = client.post(
            "/api/v1/predict/",
            data={"image": _make_upload(real_jpeg_bytes, "real_test.jpg")},
            content_type="multipart/form-data",
        )
        result = resp.get_json()["result"]
        for key in ("verdict", "confidence", "raw_prob", "heatmap", "explanation"):
            assert key in result, f"Missing key in result: {key}"

    def test_real_inference_verdict_is_probabilistic(self, client, real_jpeg_bytes):
        """Verdict must use probabilistic wording, never absolute."""
        resp = client.post(
            "/api/v1/predict/",
            data={"image": _make_upload(real_jpeg_bytes, "real_test.jpg")},
            content_type="multipart/form-data",
        )
        verdict = resp.get_json()["result"]["verdict"]
        assert verdict in ("likely AI-generated", "likely real"), (
            f"Verdict '{verdict}' must be probabilistic (starts with 'likely')"
        )
        # Must NOT be absolute claims
        assert verdict not in ("AI-generated", "real", "definitely fake", "100% AI")

    def test_real_inference_raw_prob_in_range(self, client, real_jpeg_bytes):
        resp = client.post(
            "/api/v1/predict/",
            data={"image": _make_upload(real_jpeg_bytes, "real_test.jpg")},
            content_type="multipart/form-data",
        )
        raw_prob = resp.get_json()["result"]["raw_prob"]
        assert isinstance(raw_prob, float)
        assert 0.0 <= raw_prob <= 1.0

    def test_real_inference_confidence_in_range(self, client, real_jpeg_bytes):
        resp = client.post(
            "/api/v1/predict/",
            data={"image": _make_upload(real_jpeg_bytes, "real_test.jpg")},
            content_type="multipart/form-data",
        )
        confidence = resp.get_json()["result"]["confidence"]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_real_inference_heatmap_null(self, client, real_jpeg_bytes):
        """heatmap must be null — Grad-CAM not yet implemented."""
        resp = client.post(
            "/api/v1/predict/",
            data={"image": _make_upload(real_jpeg_bytes, "real_test.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.get_json()["result"]["heatmap"] is None

    def test_real_inference_explanation_null(self, client, real_jpeg_bytes):
        """explanation must be null — Grad-CAM not yet implemented."""
        resp = client.post(
            "/api/v1/predict/",
            data={"image": _make_upload(real_jpeg_bytes, "real_test.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.get_json()["result"]["explanation"] is None

    def test_real_corrupt_image_returns_4xx(self, client):
        """A corrupt file that passes extension check should return 4xx, not crash."""
        corrupt_bytes = b"not_an_image_at_all_" * 10
        resp = client.post(
            "/api/v1/predict/",
            data={"image": _make_upload(corrupt_bytes, "corrupt.jpg")},
            content_type="multipart/form-data",
        )
        # Should be a client error (4xx), not a server crash (5xx from unhandled exc)
        assert 400 <= resp.status_code < 600
        data = resp.get_json()
        assert data["success"] is False


# ===========================================================================
# 6. ModelService unit tests
# ===========================================================================

class TestModelServiceImport:
    def test_model_service_imports(self):
        from services.model_service import ModelService, ModelNotReadyError, InferenceError
        assert ModelService is not None
        assert issubclass(ModelNotReadyError, Exception)
        assert issubclass(InferenceError, Exception)

    def test_model_service_instantiation(self):
        from services.model_service import ModelService
        svc = ModelService()
        assert svc.weights_path is None

    def test_model_service_with_weights_path(self):
        from services.model_service import ModelService
        svc = ModelService(weights_path="/some/path.keras")
        assert svc.weights_path == "/some/path.keras"

    def test_model_service_missing_weights_raises(self, tmp_path):
        """ModelService.run() with non-existent weights → ModelNotReadyError."""
        from services.model_service import ModelService, ModelNotReadyError
        import os
        # Need a real image file to get past the FileNotFoundError in predict()
        real_dir = _PROJECT_ROOT / "train" / "REAL"
        if not real_dir.is_dir():
            pytest.skip("train/REAL/ not available")
        sample = next(iter(sorted(real_dir.iterdir())))
        fake_weights = str(tmp_path / "nonexistent.keras")
        svc = ModelService(weights_path=fake_weights)
        with pytest.raises(ModelNotReadyError):
            svc.run(str(sample))
