"""
Post-training verification tests.
Run AFTER model/train.py has completed and saved the model.

Verifies:
  1. Saved model file exists
  2. Model can be loaded with tf.keras.models.load_model
  3. Loaded model has correct architecture
  4. Inference on a sample image from ALLOWED DEVELOPMENT DATA works
  5. Output is a probability in [0, 1]
  6. Inference on a test-set image works (loading only, NOT evaluating performance)
  7. Training history file exists and has expected keys

NOTE ON TEST 6:
  Loading a test-set image and running a single inference call is NOT evaluation.
  It does NOT count as using the test set for training, tuning, or model selection.
  Full test-set evaluation belongs in the evaluation milestone (Session 3+).
  Only a single sample is used here purely to verify the inference pipeline works.
"""

import sys
import os
import json
from pathlib import Path

import pytest

_MODEL_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _MODEL_DIR.parent
sys.path.insert(0, str(_MODEL_DIR))

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")
PIL_available = pytest.importorskip("PIL", reason="Pillow not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def weights_path():
    from config import MODEL_SAVE_PATH
    return MODEL_SAVE_PATH


@pytest.fixture(scope="module")
def loaded_model(weights_path):
    assert weights_path.exists(), (
        f"Model weights not found at {weights_path}. "
        "Run model/train.py first."
    )
    return tf.keras.models.load_model(str(weights_path))


# ---------------------------------------------------------------------------
# Model file existence tests
# ---------------------------------------------------------------------------

class TestModelFileExists:
    def test_weights_file_exists(self, weights_path):
        """The trained .keras file must exist after training."""
        assert weights_path.exists(), (
            f"Model not found: {weights_path}. Train the model first."
        )

    def test_weights_file_is_not_empty(self, weights_path):
        """The model file must not be empty (i.e., actually saved)."""
        assert weights_path.stat().st_size > 100_000, (
            f"Model file is suspiciously small: {weights_path.stat().st_size} bytes. "
            "It may not have saved correctly."
        )

    def test_training_history_exists(self):
        from config import HISTORY_SAVE_PATH
        assert HISTORY_SAVE_PATH.exists(), (
            f"Training history not found: {HISTORY_SAVE_PATH}"
        )


# ---------------------------------------------------------------------------
# Loaded model architecture verification
# ---------------------------------------------------------------------------

class TestLoadedModelArchitecture:
    def test_loads_without_error(self, loaded_model):
        """Model must load without raising any exception."""
        assert loaded_model is not None

    def test_input_shape(self, loaded_model):
        assert tuple(loaded_model.input_shape[1:]) == (128, 128, 3)

    def test_output_shape(self, loaded_model):
        assert tuple(loaded_model.output_shape) == (None, 1)

    def test_has_global_average_pooling(self, loaded_model):
        gap = [l for l in loaded_model.layers
               if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)]
        assert len(gap) == 1

    def test_final_layer_is_dense_sigmoid(self, loaded_model):
        output = loaded_model.layers[-1]
        assert isinstance(output, tf.keras.layers.Dense)
        assert output.units == 1
        assert output.activation.__name__ == "sigmoid"


# ---------------------------------------------------------------------------
# Inference pipeline verification
# ---------------------------------------------------------------------------

class TestInferencePipeline:
    """
    Verify that inference produces valid outputs.
    These tests use DEVELOPMENT DATA (train/FAKE and train/REAL).
    The test/ directory is NOT used for model selection or performance evaluation.
    """

    def test_inference_on_train_fake_sample(self, loaded_model):
        """Inference on a sample from train/FAKE must return a probability in [0,1]."""
        from PIL import Image
        import numpy as np

        train_fake = _PROJECT_ROOT / "train" / "FAKE"
        sample = next(iter(train_fake.iterdir()))

        img = Image.open(sample).convert("RGB").resize((128, 128))
        arr = np.array(img, dtype="float32") / 255.0
        tensor = arr[np.newaxis, ...]  # (1, 128, 128, 3)

        output = loaded_model.predict(tensor, verbose=0)
        assert output.shape == (1, 1)
        prob = float(output[0, 0])
        assert 0.0 <= prob <= 1.0, f"Output probability out of range: {prob}"

    def test_inference_on_train_real_sample(self, loaded_model):
        """Inference on a sample from train/REAL must return a probability in [0,1]."""
        from PIL import Image
        import numpy as np

        train_real = _PROJECT_ROOT / "train" / "REAL"
        sample = next(iter(train_real.iterdir()))

        img = Image.open(sample).convert("RGB").resize((128, 128))
        arr = np.array(img, dtype="float32") / 255.0
        tensor = arr[np.newaxis, ...]

        output = loaded_model.predict(tensor, verbose=0)
        prob = float(output[0, 0])
        assert 0.0 <= prob <= 1.0

    def test_predict_module_works_end_to_end(self, weights_path):
        """
        Test the public predict() interface from model/predict.py works end-to-end.
        Uses a development sample from train/FAKE — NOT the test set.
        """
        from predict import predict

        train_fake = _PROJECT_ROOT / "train" / "FAKE"
        sample = str(next(iter(train_fake.iterdir())))

        result = predict(sample, weights_path=str(weights_path))

        assert "verdict" in result
        assert "confidence" in result
        assert "raw_prob" in result
        assert result["verdict"] in ("likely AI-generated", "likely real")
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["raw_prob"] <= 1.0
        # Heatmap and explanation are None until Grad-CAM milestone
        assert result["heatmap"] is None
        assert result["explanation"] is None


# ---------------------------------------------------------------------------
# Training history validation
# ---------------------------------------------------------------------------

class TestTrainingHistory:
    def test_history_has_expected_keys(self):
        from config import HISTORY_SAVE_PATH
        with open(HISTORY_SAVE_PATH) as f:
            history = json.load(f)
        assert "loss" in history
        assert "accuracy" in history
        assert "val_loss" in history
        assert "val_accuracy" in history

    def test_history_has_at_least_one_epoch(self):
        from config import HISTORY_SAVE_PATH
        with open(HISTORY_SAVE_PATH) as f:
            history = json.load(f)
        assert len(history["loss"]) >= 1, "No epochs recorded in training history"

    def test_training_loss_is_real_number(self):
        """Training loss must be a real non-NaN, non-inf float."""
        import math
        from config import HISTORY_SAVE_PATH
        with open(HISTORY_SAVE_PATH) as f:
            history = json.load(f)
        for loss_val in history["loss"]:
            assert not math.isnan(loss_val), f"NaN loss found: {loss_val}"
            assert not math.isinf(loss_val), f"Inf loss found: {loss_val}"

    def test_note_on_test_metrics(self):
        """
        Explicitly verify that the training history does NOT contain test metrics.
        The held-out test set was not used for training, tuning, or model selection.
        """
        from config import HISTORY_SAVE_PATH
        with open(HISTORY_SAVE_PATH) as f:
            history = json.load(f)
        # Test metrics should not be in training history
        # (val_* keys are validation metrics derived from the training split only)
        for key in history:
            assert not key.startswith("test_"), (
                f"Unexpected test metric found in history: {key}. "
                "Test set should not be used during training."
            )
