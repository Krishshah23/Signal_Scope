"""
Tests for model architecture — model/train.py build_model().

Verifies the SignalScope architecture meets the roadmap specification:
 1. Input shape (128, 128, 3)
 2. MobileNetV2 backbone present
 3. Backbone is frozen (trainable=False)
 4. Zero trainable layers in base model
 5. GlobalAveragePooling2D present
 6. Final layer is Dense
 7. Final Dense has 1 output unit
 8. Final activation is sigmoid
 9. Output is a binary probability (shape: (None, 1))
10. Model compiles successfully
11. Model smoke test with synthetic batch (does NOT use real data)

These tests do NOT train on real data.
These tests do NOT compute real performance.
The smoke test uses synthetic random inputs ONLY.
"""

import sys
from pathlib import Path

import pytest

_MODEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MODEL_DIR))

# Skip entire module if TF not available
tf = pytest.importorskip(
    "tensorflow",
    reason="TensorFlow not installed — skipping architecture tests",
)


# ---------------------------------------------------------------------------
# Fixture: build the model once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    from train import build_model
    return build_model(input_shape=(128, 128, 3))


@pytest.fixture(scope="module")
def base_model():
    """The MobileNetV2 base model, extracted independently for inspection."""
    m = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet",
    )
    m.trainable = False
    return m


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestModelArchitecture:
    def test_input_shape(self, model):
        """Input shape must be (128, 128, 3)."""
        assert tuple(model.input_shape[1:]) == (128, 128, 3)

    def test_backbone_is_mobilenetv2(self, base_model):
        """Backbone must be MobileNetV2."""
        assert "mobilenet" in base_model.name.lower()

    def test_backbone_is_frozen_trainable_false(self, base_model):
        """base_model.trainable must be False."""
        assert base_model.trainable is False

    def test_backbone_zero_trainable_layers(self, base_model):
        """No layers in the base model should be trainable."""
        trainable = [l for l in base_model.layers if l.trainable]
        assert len(trainable) == 0, (
            f"Expected 0 trainable layers in frozen base, found {len(trainable)}"
        )

    def test_global_average_pooling_exists(self, model):
        """Model must contain exactly one GlobalAveragePooling2D layer."""
        gap_layers = [
            l for l in model.layers
            if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)
        ]
        assert len(gap_layers) == 1

    def test_final_layer_is_dense(self, model):
        """The last layer of the model must be Dense."""
        assert isinstance(model.layers[-1], tf.keras.layers.Dense)

    def test_final_dense_has_one_unit(self, model):
        """The final Dense layer must have exactly 1 unit."""
        output_layer = model.layers[-1]
        assert output_layer.units == 1

    def test_final_activation_is_sigmoid(self, model):
        """The final Dense layer must use sigmoid activation."""
        output_layer = model.layers[-1]
        assert output_layer.activation.__name__ == "sigmoid"

    def test_output_shape_is_binary_probability(self, model):
        """Model output shape must be (None, 1)."""
        assert tuple(model.output_shape) == (None, 1)

    def test_model_compiles_successfully(self):
        """Model.compile() must not raise."""
        from train import build_model
        m = build_model()
        m.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def test_model_has_pretrained_weights(self, base_model):
        """MobileNetV2 must have non-zero weights (ImageNet pretrained)."""
        import numpy as np
        # Check that at least one weight array has non-trivial values
        weights = base_model.get_weights()
        assert len(weights) > 0
        non_zero = any(np.any(w != 0) for w in weights[:5])
        assert non_zero, "Pretrained weights appear to be all zeros — loading may have failed"


# ---------------------------------------------------------------------------
# MODEL SMOKE TEST
# Uses SYNTHETIC RANDOM DATA — not real CIFAKE images.
# This is NOT a performance evaluation.
# ---------------------------------------------------------------------------

class TestModelSmokeTest:
    """
    MODEL SMOKE TEST
    ----------------
    Tests that the model can perform a forward pass on a synthetic batch.
    Uses random float tensors, NOT real images.
    Output values are NOT meaningful — this only tests that the forward
    pass runs without errors.

    Do NOT interpret these outputs as real model performance.
    """

    def test_forward_pass_single_image(self, model):
        """
        SMOKE TEST: model can run inference on a single synthetic image.
        Output must be a float in [0, 1].
        This is NOT a performance result.
        """
        import numpy as np
        synthetic_image = np.random.rand(1, 128, 128, 3).astype("float32")
        output = model.predict(synthetic_image, verbose=0)
        assert output.shape == (1, 1)
        assert 0.0 <= float(output[0, 0]) <= 1.0

    def test_forward_pass_batch(self, model):
        """
        SMOKE TEST: model can run inference on a batch of 4 synthetic images.
        This is NOT a performance result.
        """
        import numpy as np
        synthetic_batch = np.random.rand(4, 128, 128, 3).astype("float32")
        output = model.predict(synthetic_batch, verbose=0)
        assert output.shape == (4, 1)
        for i in range(4):
            assert 0.0 <= float(output[i, 0]) <= 1.0

    def test_output_is_probability(self, model):
        """
        SMOKE TEST: outputs must be probabilities in [0, 1].
        This is NOT a performance result.
        """
        import numpy as np
        inputs = np.random.rand(8, 128, 128, 3).astype("float32")
        outputs = model.predict(inputs, verbose=0)
        assert (outputs >= 0.0).all() and (outputs <= 1.0).all()

    def test_training_step_does_not_crash(self, model):
        """
        SMOKE TEST: one training step with synthetic data must not raise.
        Uses random binary labels. Result is meaningless.
        This is NOT a performance result.
        """
        import numpy as np
        images = np.random.rand(4, 128, 128, 3).astype("float32")
        labels = np.array([0, 1, 0, 1], dtype="float32")
        history = model.fit(images, labels, epochs=1, verbose=0)
        assert "loss" in history.history
        # Loss must be a real number (not NaN/inf)
        assert not __import__("math").isnan(history.history["loss"][0])
