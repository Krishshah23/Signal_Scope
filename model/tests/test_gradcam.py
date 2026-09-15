"""
Tests for model/gradcam.py — Grad-CAM Explainability (Session 5)

Verifies:
  1. gradcam module imports cleanly
  2. Trained model loads
  3. Correct target layer (out_relu) identified
  4. Logit-based Grad-CAM on a FAKE image
  5. Logit-based Grad-CAM on a REAL image
  6. Heatmap dimensions are (128, 128)
  7. Heatmap values are finite
  8. Heatmap values are in [0, 1]
  9. base64 heatmap decodes to a valid 128x128 PNG
 10. overlay base64 decodes to a valid RGB 128x128 PNG
 11. Grad-CAM does NOT modify model weights
 12. model.fit() is never called in gradcam.py
 13. predict() still returns verdict/confidence/raw_prob/heatmap/explanation
 14. Invalid images handled safely

Uses a single JPEG from train/FAKE/ and train/REAL/ as smoke tests.
The held-out test set is NEVER used here.
model.fit() is NEVER called.
"""

from __future__ import annotations

import base64
import sys
import math
import io
from pathlib import Path

import numpy as np
import pytest

_MODEL_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _MODEL_DIR.parent
sys.path.insert(0, str(_MODEL_DIR))

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")

_WEIGHTS = _PROJECT_ROOT / "model" / "weights" / "signalscope_baseline.keras"
_FAKE_DIR = _PROJECT_ROOT / "train" / "FAKE"
_REAL_DIR = _PROJECT_ROOT / "train" / "REAL"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_model():
    if not _WEIGHTS.exists():
        pytest.skip(f"Weights not found: {_WEIGHTS}")
    return tf.keras.models.load_model(str(_WEIGHTS))


@pytest.fixture(scope="module")
def fake_sample():
    if not _FAKE_DIR.is_dir():
        pytest.skip("train/FAKE/ not available")
    return next(iter(sorted(_FAKE_DIR.iterdir())))


@pytest.fixture(scope="module")
def real_sample():
    if not _REAL_DIR.is_dir():
        pytest.skip("train/REAL/ not available")
    return next(iter(sorted(_REAL_DIR.iterdir())))


@pytest.fixture(scope="module")
def gradcam_fake(fake_sample, loaded_model):
    """Run generate_gradcam() on a FAKE sample, cached for the test module."""
    from gradcam import generate_gradcam
    return generate_gradcam(str(fake_sample), loaded_model)


@pytest.fixture(scope="module")
def gradcam_real(real_sample, loaded_model):
    """Run generate_gradcam() on a REAL sample, cached for the test module."""
    from gradcam import generate_gradcam
    return generate_gradcam(str(real_sample), loaded_model)


# ---------------------------------------------------------------------------
# 1. Import & constant tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_module_imports(self):
        import gradcam  # noqa: F401

    def test_generate_gradcam_callable(self):
        from gradcam import generate_gradcam
        assert callable(generate_gradcam)

    def test_find_target_layer_callable(self):
        from gradcam import find_gradcam_target_layer
        assert callable(find_gradcam_target_layer)

    def test_target_layer_constant(self):
        from gradcam import GRADCAM_TARGET_LAYER
        assert GRADCAM_TARGET_LAYER == "out_relu"

    def test_explanation_is_probabilistic(self):
        from gradcam import EXPLANATION_TEMPLATE
        forbidden = ["proves", "definitely", "certainly", "100%", "is fake", "is real"]
        lower = EXPLANATION_TEMPLATE.lower()
        for w in forbidden:
            assert w not in lower, f"EXPLANATION_TEMPLATE uses forbidden word '{w}'"

    def test_no_fit_call_in_source(self):
        import ast
        source = (_MODEL_DIR / "gradcam.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fit_calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(getattr(n, "func", None), ast.Attribute)
            and n.func.attr == "fit"
        ]
        assert len(fit_calls) == 0, "gradcam.py must never call .fit()"


# ---------------------------------------------------------------------------
# 2. Model loading & architecture
# ---------------------------------------------------------------------------

class TestModelArchitecture:
    def test_weights_file_exists(self):
        assert _WEIGHTS.exists()

    def test_model_loads(self, loaded_model):
        assert loaded_model is not None

    def test_mobilenet_submodel_present(self, loaded_model):
        from gradcam import MOBILENET_SUBMODEL_NAME
        names = [l.name for l in loaded_model.layers]
        assert MOBILENET_SUBMODEL_NAME in names

    def test_target_layer_in_mobilenet(self, loaded_model):
        from gradcam import GRADCAM_TARGET_LAYER, MOBILENET_SUBMODEL_NAME, _get_layer
        mobilenet = _get_layer(loaded_model, MOBILENET_SUBMODEL_NAME)
        layer = _get_layer(mobilenet, GRADCAM_TARGET_LAYER)
        assert layer.name == GRADCAM_TARGET_LAYER

    def test_find_target_layer_matches_constant(self, loaded_model):
        from gradcam import find_gradcam_target_layer, GRADCAM_TARGET_LAYER
        found = find_gradcam_target_layer(loaded_model)
        assert found == GRADCAM_TARGET_LAYER, (
            f"Dynamic discovery found '{found}', constant is '{GRADCAM_TARGET_LAYER}'"
        )

    def test_dense_layer_has_expected_shape(self, loaded_model):
        dense = loaded_model.layers[3]
        w, b = dense.get_weights()
        assert w.shape == (1280, 1), f"Expected Dense weights (1280,1), got {w.shape}"


# ---------------------------------------------------------------------------
# 3. Logit-based gradient verification
# ---------------------------------------------------------------------------

class TestLogitGradients:
    """Verifies that the logit approach produces non-None, non-zero gradients."""

    def _compute_logit_grads(self, loaded_model, sample_path):
        import tensorflow as tf
        from gradcam import GRADCAM_TARGET_LAYER, MOBILENET_SUBMODEL_NAME, _get_layer
        from PIL import Image

        img = np.array(
            Image.open(sample_path).convert("RGB").resize((128, 128), Image.BILINEAR),
            dtype="float32",
        ) / 255.0
        img_tensor = tf.constant(img[np.newaxis, ...])

        mobilenet = _get_layer(loaded_model, MOBILENET_SUBMODEL_NAME)
        target_layer = _get_layer(mobilenet, GRADCAM_TARGET_LAYER)
        feat_model = tf.keras.Model(
            inputs=mobilenet.input,
            outputs=target_layer.output,
        )
        dense = loaded_model.layers[3]
        dw, db = dense.get_weights()

        feats_np = feat_model(img_tensor, training=False).numpy()
        feats_var = tf.Variable(feats_np, trainable=True, dtype=tf.float32)

        with tf.GradientTape() as tape:
            gap_out = tf.reduce_mean(feats_var, axis=[1, 2])
            logit = tf.linalg.matmul(gap_out, tf.constant(dw)) + tf.constant(db)
            logit_scalar = logit[0, 0]

        grads = tape.gradient(logit_scalar, feats_var)
        return grads, feats_np

    def test_fake_image_grads_not_none(self, loaded_model, fake_sample):
        grads, _ = self._compute_logit_grads(loaded_model, fake_sample)
        assert grads is not None

    def test_fake_image_grads_not_all_zero(self, loaded_model, fake_sample):
        grads, _ = self._compute_logit_grads(loaded_model, fake_sample)
        assert np.any(grads.numpy() != 0), (
            "All gradients are zero even with logit approach. "
            "This may indicate a deeper architecture issue."
        )

    def test_real_image_grads_not_none(self, loaded_model, real_sample):
        grads, _ = self._compute_logit_grads(loaded_model, real_sample)
        assert grads is not None

    def test_real_image_grads_not_all_zero(self, loaded_model, real_sample):
        grads, _ = self._compute_logit_grads(loaded_model, real_sample)
        assert np.any(grads.numpy() != 0)


# ---------------------------------------------------------------------------
# 4. Grad-CAM output structure (FAKE image)
# ---------------------------------------------------------------------------

class TestGradcamFakeOutput:
    def test_result_is_dict(self, gradcam_fake):
        assert isinstance(gradcam_fake, dict)

    def test_has_all_keys(self, gradcam_fake):
        for k in ("heatmap_array", "heatmap_b64", "overlay_b64", "raw_prob", "target_layer"):
            assert k in gradcam_fake, f"Missing key: {k}"

    def test_target_layer_matches_constant(self, gradcam_fake):
        from gradcam import GRADCAM_TARGET_LAYER
        assert gradcam_fake["target_layer"] == GRADCAM_TARGET_LAYER

    def test_raw_prob_in_range(self, gradcam_fake):
        assert 0.0 <= gradcam_fake["raw_prob"] <= 1.0

    def test_heatmap_shape(self, gradcam_fake):
        assert gradcam_fake["heatmap_array"].shape == (128, 128)

    def test_heatmap_dtype(self, gradcam_fake):
        assert gradcam_fake["heatmap_array"].dtype == np.float32

    def test_heatmap_finite(self, gradcam_fake):
        assert np.all(np.isfinite(gradcam_fake["heatmap_array"]))

    def test_heatmap_normalised(self, gradcam_fake):
        h = gradcam_fake["heatmap_array"]
        assert float(h.min()) >= -1e-6
        assert float(h.max()) <= 1.0 + 1e-6

    def test_heatmap_b64_is_string(self, gradcam_fake):
        assert isinstance(gradcam_fake["heatmap_b64"], str)
        assert len(gradcam_fake["heatmap_b64"]) > 100

    def test_heatmap_b64_decodes_to_128x128_png(self, gradcam_fake):
        from PIL import Image
        decoded = base64.b64decode(gradcam_fake["heatmap_b64"])
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (128, 128)

    def test_overlay_b64_is_rgb_png(self, gradcam_fake):
        from PIL import Image
        decoded = base64.b64decode(gradcam_fake["overlay_b64"])
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (128, 128)
        assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# 5. Grad-CAM output (REAL image)
# ---------------------------------------------------------------------------

class TestGradcamRealOutput:
    def test_real_result_is_dict(self, gradcam_real):
        assert isinstance(gradcam_real, dict)

    def test_real_heatmap_shape(self, gradcam_real):
        assert gradcam_real["heatmap_array"].shape == (128, 128)

    def test_real_heatmap_finite(self, gradcam_real):
        assert np.all(np.isfinite(gradcam_real["heatmap_array"]))

    def test_real_heatmap_normalised(self, gradcam_real):
        h = gradcam_real["heatmap_array"]
        assert float(h.min()) >= -1e-6
        assert float(h.max()) <= 1.0 + 1e-6

    def test_real_overlay_b64_valid(self, gradcam_real):
        from PIL import Image
        decoded = base64.b64decode(gradcam_real["overlay_b64"])
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (128, 128)


# ---------------------------------------------------------------------------
# 6. Model weight integrity
# ---------------------------------------------------------------------------

class TestWeightIntegrity:
    def test_weights_unchanged_after_gradcam(self, loaded_model, fake_sample):
        """Grad-CAM must not modify model weights."""
        before = [w.numpy().copy() for w in loaded_model.trainable_weights[:5]]
        from gradcam import generate_gradcam
        generate_gradcam(str(fake_sample), loaded_model)
        after = [w.numpy() for w in loaded_model.trainable_weights[:5]]
        for b, a in zip(before, after):
            assert np.allclose(b, a, rtol=0, atol=0), "Model weights were modified by Grad-CAM!"


# ---------------------------------------------------------------------------
# 7. predict() integration
# ---------------------------------------------------------------------------

class TestPredictIntegration:
    def test_predict_returns_heatmap_key(self, fake_sample):
        from predict import predict
        result = predict(str(fake_sample), weights_path=str(_WEIGHTS))
        assert "heatmap" in result

    def test_predict_heatmap_is_string_or_none(self, fake_sample):
        from predict import predict
        result = predict(str(fake_sample), weights_path=str(_WEIGHTS))
        hm = result["heatmap"]
        assert hm is None or (isinstance(hm, str) and len(hm) > 100)

    def test_predict_core_fields_present(self, fake_sample):
        from predict import predict, VERDICT_AI, VERDICT_REAL
        result = predict(str(fake_sample), weights_path=str(_WEIGHTS))
        assert result["verdict"] in (VERDICT_AI, VERDICT_REAL)
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["raw_prob"] <= 1.0

    def test_predict_explanation_is_string_or_none(self, fake_sample):
        from predict import predict
        result = predict(str(fake_sample), weights_path=str(_WEIGHTS))
        exp = result["explanation"]
        assert exp is None or isinstance(exp, str)

    def test_predict_on_real_image(self, real_sample):
        from predict import predict, VERDICT_AI, VERDICT_REAL
        result = predict(str(real_sample), weights_path=str(_WEIGHTS))
        assert result["verdict"] in (VERDICT_AI, VERDICT_REAL)
        assert "heatmap" in result

    def test_predict_explanation_not_absolute(self, fake_sample):
        from predict import predict
        result = predict(str(fake_sample), weights_path=str(_WEIGHTS))
        if result["explanation"] is not None:
            forbidden = ["proves", "definitely", "is fake", "is real", "certainly"]
            for w in forbidden:
                assert w not in result["explanation"].lower(), \
                    f"Explanation uses forbidden word '{w}'"


# ---------------------------------------------------------------------------
# 8. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_corrupt_image_raises(self, loaded_model, tmp_path):
        from gradcam import generate_gradcam
        corrupt = tmp_path / "bad.jpg"
        corrupt.write_bytes(b"not_an_image")
        with pytest.raises(Exception):
            generate_gradcam(str(corrupt), loaded_model)

    def test_nonexistent_path_raises_file_not_found(self):
        from predict import predict
        with pytest.raises(FileNotFoundError):
            predict("/no/such/image.jpg", weights_path=str(_WEIGHTS))
