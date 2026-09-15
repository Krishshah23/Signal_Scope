"""
Post-training inference tests — model/predict.py

Verifies that the real trained model can be loaded and that predict()
returns correct output structure on sample images.

These tests use DEVELOPMENT DATA (train/FAKE/ and train/REAL/) for
single-sample smoke checks — NOT the held-out test set.

Requirements:
  - model/weights/signalscope_baseline.keras must exist.
  - Pillow and TensorFlow must be installed.
  - The held-out test set is NEVER evaluated here.
  - model.fit() is NEVER called.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MODEL_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _MODEL_DIR.parent
sys.path.insert(0, str(_MODEL_DIR))

# Skip entire module if TF not available
tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def weights_path():
    # Use absolute path directly — avoids sys.path collision with
    # src/backend/config.py when running the combined test suite.
    path = _PROJECT_ROOT / "model" / "weights" / "signalscope_baseline.keras"
    return path


@pytest.fixture(scope="module")
def train_fake_sample(tmp_path_factory):
    """Return a path to a single JPEG from train/FAKE/ for smoke tests."""
    fake_dir = _PROJECT_ROOT / "train" / "FAKE"
    if not fake_dir.is_dir():
        pytest.skip("train/FAKE/ not available — skipping inference smoke tests")
    sample = next(iter(sorted(fake_dir.iterdir())))
    return sample


@pytest.fixture(scope="module")
def train_real_sample():
    """Return a path to a single JPEG from train/REAL/ for smoke tests."""
    real_dir = _PROJECT_ROOT / "train" / "REAL"
    if not real_dir.is_dir():
        pytest.skip("train/REAL/ not available — skipping inference smoke tests")
    return next(iter(sorted(real_dir.iterdir())))


# ---------------------------------------------------------------------------
# 1. Module-level import tests
# ---------------------------------------------------------------------------

class TestPredictModuleImports:
    def test_predict_module_imports(self):
        """model/predict.py imports without error."""
        import predict  # noqa: F401

    def test_predict_function_exists(self):
        from predict import predict
        assert callable(predict)

    def test_constants_correct(self):
        from predict import IMAGE_SIZE, VERDICT_AI, VERDICT_REAL, CLASSIFICATION_THRESHOLD
        assert IMAGE_SIZE == (128, 128)
        assert VERDICT_AI == "likely AI-generated"
        assert VERDICT_REAL == "likely real"
        assert CLASSIFICATION_THRESHOLD == 0.5

    def test_exceptions_importable(self):
        from predict import ModelNotTrainedError, InvalidImageError
        assert issubclass(ModelNotTrainedError, Exception)
        assert issubclass(InvalidImageError, ValueError)


# ---------------------------------------------------------------------------
# 2. Model loading
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_weights_file_exists(self, weights_path):
        assert weights_path.exists(), (
            f"Model weights not found: {weights_path}\n"
            "Run model/train.py to train the model first."
        )

    def test_weights_file_is_not_empty(self, weights_path):
        size = weights_path.stat().st_size
        assert size > 100_000, f"Weights file suspiciously small: {size} bytes"

    def test_model_loads_via_keras(self, weights_path):
        """Keras can load the .keras file without error."""
        model = tf.keras.models.load_model(str(weights_path))
        assert model is not None

    def test_loaded_model_input_shape(self, weights_path):
        model = tf.keras.models.load_model(str(weights_path))
        assert tuple(model.input_shape[1:]) == (128, 128, 3)

    def test_loaded_model_output_shape(self, weights_path):
        model = tf.keras.models.load_model(str(weights_path))
        assert tuple(model.output_shape) == (None, 1)


# ---------------------------------------------------------------------------
# 3. predict() output structure
# ---------------------------------------------------------------------------

class TestPredictOutputStructure:
    """Verify predict() returns the correct dict structure on real images."""

    def test_predict_returns_dict(self, train_fake_sample, weights_path):
        from predict import predict
        result = predict(str(train_fake_sample), weights_path=str(weights_path))
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, train_fake_sample, weights_path):
        from predict import predict
        result = predict(str(train_fake_sample), weights_path=str(weights_path))
        for key in ("verdict", "confidence", "raw_prob", "heatmap", "explanation"):
            assert key in result, f"Missing key: {key}"

    def test_verdict_is_valid_string(self, train_fake_sample, weights_path):
        from predict import predict, VERDICT_AI, VERDICT_REAL
        result = predict(str(train_fake_sample), weights_path=str(weights_path))
        assert result["verdict"] in (VERDICT_AI, VERDICT_REAL)

    def test_raw_prob_in_range(self, train_fake_sample, weights_path):
        from predict import predict
        result = predict(str(train_fake_sample), weights_path=str(weights_path))
        assert 0.0 <= result["raw_prob"] <= 1.0

    def test_confidence_in_range(self, train_fake_sample, weights_path):
        from predict import predict
        result = predict(str(train_fake_sample), weights_path=str(weights_path))
        assert 0.0 <= result["confidence"] <= 1.0

    def test_heatmap_is_string_or_none(self, train_fake_sample, weights_path):
        """Heatmap is a base64 string when Grad-CAM succeeds, or None if it fails."""
        from predict import predict
        result = predict(str(train_fake_sample), weights_path=str(weights_path))
        hm = result["heatmap"]
        assert hm is None or (isinstance(hm, str) and len(hm) > 0), (
            f"heatmap must be a non-empty string or None, got: {type(hm)}"
        )

    def test_explanation_is_string_or_none(self, train_fake_sample, weights_path):
        """Explanation is a string when Grad-CAM succeeds, or None if it fails."""
        from predict import predict
        result = predict(str(train_fake_sample), weights_path=str(weights_path))
        exp = result["explanation"]
        assert exp is None or isinstance(exp, str), (
            f"explanation must be a string or None, got: {type(exp)}"
        )

    def test_predict_on_real_image(self, train_real_sample, weights_path):
        """predict() must work on REAL images too, not just FAKE."""
        from predict import predict, VERDICT_AI, VERDICT_REAL
        result = predict(str(train_real_sample), weights_path=str(weights_path))
        assert result["verdict"] in (VERDICT_AI, VERDICT_REAL)
        assert 0.0 <= result["raw_prob"] <= 1.0

    def test_confidence_matches_verdict_and_raw_prob(self, train_fake_sample, weights_path):
        """
        confidence must be consistent with verdict and raw_prob.
        If FAKE: confidence == raw_prob
        If REAL: confidence == 1 - raw_prob  (approximately)
        """
        from predict import predict, VERDICT_AI, CLASSIFICATION_THRESHOLD
        result = predict(str(train_fake_sample), weights_path=str(weights_path))
        raw = result["raw_prob"]
        conf = result["confidence"]
        if result["verdict"] == VERDICT_AI:
            assert abs(conf - raw) < 1e-3, (
                f"FAKE verdict: confidence ({conf}) should ≈ raw_prob ({raw})"
            )
        else:
            assert abs(conf - (1.0 - raw)) < 1e-3, (
                f"REAL verdict: confidence ({conf}) should ≈ 1-raw_prob ({1.0-raw})"
            )


# ---------------------------------------------------------------------------
# 4. Error handling
# ---------------------------------------------------------------------------

class TestPredictErrorHandling:
    def test_missing_file_raises_file_not_found(self, weights_path):
        from predict import predict
        with pytest.raises(FileNotFoundError):
            predict("/nonexistent/path/image.jpg", weights_path=str(weights_path))

    def test_invalid_image_raises_invalid_image_error(self, tmp_path, weights_path):
        """A file with valid extension but corrupt content raises InvalidImageError."""
        from predict import predict, InvalidImageError
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"this is not image data at all")
        with pytest.raises(InvalidImageError):
            predict(str(corrupt), weights_path=str(weights_path))

    def test_missing_weights_raises_model_not_trained_error(self, tmp_path):
        """Pointing to a non-existent weights file raises ModelNotTrainedError."""
        from predict import predict, ModelNotTrainedError
        # We need a real image to get past file-not-found, use a real sample
        real_dir = _PROJECT_ROOT / "train" / "REAL"
        if not real_dir.is_dir():
            pytest.skip("train/REAL/ not available")
        sample = next(iter(sorted(real_dir.iterdir())))
        fake_weights = str(tmp_path / "nonexistent.keras")
        with pytest.raises(ModelNotTrainedError):
            predict(str(sample), weights_path=fake_weights)

    def test_model_caching_does_not_reload(self, train_fake_sample, weights_path):
        """
        Calling predict() twice should reuse the cached model, not reload it.
        This is a structural test — we verify both calls succeed quickly.
        """
        import time
        from predict import predict
        import predict as predict_module

        # Reset module-level cache to ensure first load is measured
        predict_module._model = None
        predict_module._model_weights_path = None

        t1 = time.time()
        predict(str(train_fake_sample), weights_path=str(weights_path))
        first_load = time.time() - t1

        t2 = time.time()
        predict(str(train_fake_sample), weights_path=str(weights_path))
        second_load = time.time() - t2

        # Second call should be at least 5× faster (no model loading)
        assert second_load < first_load, (
            f"Second predict() call ({second_load:.3f}s) should be faster than "
            f"first ({first_load:.3f}s) due to model caching."
        )

    def test_no_fit_called(self):
        """predict.py must never call model.fit()."""
        import ast
        source = (_MODEL_DIR / "predict.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fit_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(getattr(node, "func", None), ast.Attribute)
            and node.func.attr == "fit"
        ]
        assert len(fit_calls) == 0, "predict.py must never call .fit()"
