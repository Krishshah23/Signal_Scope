"""
SignalScope — Grad-CAM Explainability
--------------------------------------
Generates Gradient-weighted Class Activation Maps (Grad-CAM) for the trained
MobileNetV2-based SignalScope model.

Grad-CAM reference:
    Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization." ICCV 2017.
    https://arxiv.org/abs/1610.02391

MODEL ARCHITECTURE CONTEXT:
    Top-level model:
        input_image  (128×128×3)
            ↓
        mobilenetv2_1.00_128  (Functional sub-model, frozen)
            … → Conv_1 → Conv_1_bn → out_relu   ← TARGET LAYER (4×4×1280)
            ↓
        global_avg_pool  (GlobalAveragePooling2D)
            ↓
        output  (Dense(1, activation="sigmoid"))

GRAD-CAM IMPLEMENTATION DESIGN:
    1. Extract feature maps from 'out_relu' using a feature-extractor sub-model.
    2. Wrap the feature maps as a tf.Variable (so GradientTape can differentiate
       through them).
    3. Route ONLY through the classifier tail (manual GAP + Dense) inside the tape.
    4. Compute gradient of the PRE-SIGMOID LOGIT (not the sigmoid output) w.r.t.
       the feature maps.

    WHY LOGIT INSTEAD OF SIGMOID:
        sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        When the model is very confident (sigmoid → 0 or 1), the derivative
        approaches 0, causing gradients to vanish completely.
        Using the pre-sigmoid logit avoids this saturation and produces
        informative spatial gradients regardless of prediction confidence.
        The public prediction interface (predict.py) CONTINUES to use the
        sigmoid probability for verdict/confidence — only Grad-CAM uses the
        logit internally.

    5. Pool gradients globally (channel importances).
    6. Weighted sum of feature maps.
    7. ReLU: keep only positive activations.
    8. Normalise to [0, 1].
    9. Resize to 128×128.
    10. Encode as base64 PNG.

WORDING RULE:
    Grad-CAM highlights regions that influenced the model's prediction.
    It does NOT prove manipulation, AI origin, or specific generator type.
    Results are always described probabilistically.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — confirmed by live model inspection in Session 5
# ---------------------------------------------------------------------------

# 'out_relu' is the final spatial ReLU activation (after Conv_1 → Conv_1_bn)
# inside the MobileNetV2 sub-model, immediately before GlobalAveragePooling2D.
GRADCAM_TARGET_LAYER: str = "out_relu"
MOBILENET_SUBMODEL_NAME: str = "mobilenetv2_1.00_128"

EXPLANATION_TEMPLATE: str = (
    "Highlighted regions indicate areas of the image that contributed more "
    "strongly to the model's prediction. This is a probabilistic estimate "
    "and does not definitively prove the image's origin."
)


# ---------------------------------------------------------------------------
# Core Grad-CAM function
# ---------------------------------------------------------------------------

def generate_gradcam(
    image_path: str,
    model,
    target_layer_name: str = GRADCAM_TARGET_LAYER,
    submodel_name: str = MOBILENET_SUBMODEL_NAME,
) -> dict:
    """
    Generate a Grad-CAM heatmap for a single image.

    Uses the pre-sigmoid logit for gradient computation to avoid vanishing
    gradients caused by sigmoid saturation (see module docstring).

    Parameters
    ----------
    image_path : str
        Path to the input image (JPEG/PNG/WebP/BMP).
    model : tf.keras.Model
        The loaded SignalScope model.
    target_layer_name : str
        Name of the convolutional layer inside the MobileNetV2 sub-model.
        Default: 'out_relu'.
    submodel_name : str
        Name of the MobileNetV2 Functional sub-model.

    Returns
    -------
    dict
        {
            "heatmap_array":  np.ndarray (128, 128) float32 in [0, 1]
            "heatmap_b64":    str  base64 PNG of jet-coloured heatmap
            "overlay_b64":    str  base64 PNG of heatmap blended onto original
            "raw_prob":       float  sigmoid output probability
            "target_layer":   str  layer name used for Grad-CAM
        }

    Raises
    ------
    ValueError
        If the target layer cannot be found.
    RuntimeError
        If gradient computation fails or produces None.
    """
    import tensorflow as tf
    from PIL import Image

    # -----------------------------------------------------------------------
    # Step 1 — Preprocess (same pipeline as predict.py and evaluate.py)
    # -----------------------------------------------------------------------
    img_orig = Image.open(image_path).convert("RGB")
    img_resized = img_orig.resize((128, 128), Image.BILINEAR)
    img_array = np.array(img_resized, dtype="float32") / 255.0
    img_tensor = tf.constant(img_array[np.newaxis, ...])   # (1, 128, 128, 3)

    # -----------------------------------------------------------------------
    # Step 2 — Locate the MobileNetV2 sub-model and target layer
    # -----------------------------------------------------------------------
    mobilenet = _get_layer(model, submodel_name)
    target_layer = _get_layer(mobilenet, target_layer_name)

    # -----------------------------------------------------------------------
    # Step 3 — Build a feature extractor from the mobilenet sub-model's
    # input to the target layer's output
    # -----------------------------------------------------------------------
    feat_model = tf.keras.Model(
        inputs=mobilenet.input,
        outputs=target_layer.output,
        name="gradcam_feature_extractor",
    )

    # -----------------------------------------------------------------------
    # Step 4 — Extract the Dense classifier weights (after GAP)
    # We will manually apply GAP + linear transform to get the logit.
    # This bypasses the sigmoid so gradients don't vanish.
    # -----------------------------------------------------------------------
    # Top-level model structure: layers[2] = GAP, layers[3] = Dense(sigmoid)
    # Get the Dense layer weights directly — no sigmoid in our forward pass.
    dense_layer = model.layers[3]   # Dense(1, activation="sigmoid")
    dense_weights, dense_bias = dense_layer.get_weights()
    # dense_weights: (1280, 1)   dense_bias: (1,)

    # -----------------------------------------------------------------------
    # Step 5 — Compute feature maps (outside tape — just data extraction)
    # -----------------------------------------------------------------------
    feats_np = feat_model(img_tensor, training=False).numpy()   # (1, 4, 4, 1280)

    # -----------------------------------------------------------------------
    # Step 6 — Wrap features as a tf.Variable and compute logit under tape
    # GradientTape watches trainable variables by default, so wrapping as
    # tf.Variable allows us to differentiate the logit w.r.t. the features.
    # -----------------------------------------------------------------------
    feats_var = tf.Variable(feats_np, trainable=True, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Manual GAP: mean over spatial dims (H=4, W=4) → (1, 1280)
        gap_out = tf.reduce_mean(feats_var, axis=[1, 2])

        # Linear transform (no sigmoid): logit = W·gap + b
        # dense_weights is (1280,1), so transpose for matmul → (1,1)
        logit = tf.linalg.matmul(gap_out, tf.constant(dense_weights)) \
                + tf.constant(dense_bias)
        logit_scalar = logit[0, 0]

        # Compute sigmoid probability for return value
        raw_prob = float(tf.sigmoid(logit)[0, 0].numpy())

    # Gradient of the LOGIT (not sigmoid) w.r.t. feature maps
    grads = tape.gradient(logit_scalar, feats_var)   # (1, 4, 4, 1280)

    if grads is None:
        raise RuntimeError(
            "GradientTape returned None gradients. "
            "Check that the feature variable is inside the computation graph."
        )

    # -----------------------------------------------------------------------
    # Step 7 — Global average pooling of gradients → channel importances
    # -----------------------------------------------------------------------
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()   # (1280,)

    # -----------------------------------------------------------------------
    # Step 8 — Weighted sum of feature maps
    # -----------------------------------------------------------------------
    conv_np = feats_np[0]                          # (4, 4, 1280)
    weighted = conv_np * pooled_grads              # (4, 4, 1280)
    heatmap = np.sum(weighted, axis=-1)            # (4, 4)

    # -----------------------------------------------------------------------
    # Step 9 — ReLU: keep only positively contributing features
    # -----------------------------------------------------------------------
    heatmap = np.maximum(heatmap, 0.0)

    # -----------------------------------------------------------------------
    # Step 10 — Normalise to [0, 1]
    # -----------------------------------------------------------------------
    heatmap = _normalize_heatmap(heatmap)

    # -----------------------------------------------------------------------
    # Step 11 — Resize to 128×128
    # -----------------------------------------------------------------------
    heatmap_128 = _resize_heatmap(heatmap, target_size=(128, 128))

    # -----------------------------------------------------------------------
    # Step 12 — Encode for API use
    # -----------------------------------------------------------------------
    heatmap_b64 = _heatmap_to_base64(heatmap_128)
    overlay_b64 = _create_overlay_base64(img_resized, heatmap_128)

    logger.info(
        "Grad-CAM OK — layer='%s'  raw_prob=%.4f  "
        "heatmap range [%.3f, %.3f]",
        target_layer_name, raw_prob,
        float(heatmap_128.min()), float(heatmap_128.max()),
    )

    return {
        "heatmap_array": heatmap_128,
        "heatmap_b64":   heatmap_b64,
        "overlay_b64":   overlay_b64,
        "raw_prob":      raw_prob,
        "target_layer":  target_layer_name,
    }


# ---------------------------------------------------------------------------
# Public helper: find target layer programmatically
# ---------------------------------------------------------------------------

def find_gradcam_target_layer(model, submodel_name: str = MOBILENET_SUBMODEL_NAME) -> str:
    """
    Programmatically find the last spatial layer in the MobileNetV2 sub-model.
    Used to validate the hardcoded GRADCAM_TARGET_LAYER constant.
    """
    mobilenet = _get_layer(model, submodel_name)

    spatial_types = ("ReLU", "Activation", "Conv2D", "DepthwiseConv2D", "BatchNormalization")
    stop_types = ("GlobalAveragePooling2D", "Flatten", "GlobalMaxPooling2D")

    last_spatial = None
    for layer in mobilenet.layers:
        t = type(layer).__name__
        if t in stop_types:
            break
        if t in spatial_types:
            last_spatial = layer

    if last_spatial is None:
        raise ValueError(f"No spatial layer found in sub-model '{submodel_name}'.")
    return last_spatial.name


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_layer(model_or_submodel, layer_name: str):
    """Return a layer by name from a Keras model or sub-model."""
    for layer in model_or_submodel.layers:
        if layer.name == layer_name:
            return layer
    available = [l.name for l in model_or_submodel.layers[-10:]]
    raise ValueError(
        f"Layer '{layer_name}' not found in '{model_or_submodel.name}'. "
        f"Last 10 layers: {available}"
    )


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalise 2D heatmap to [0, 1]. Returns zeros if degenerate."""
    h_min, h_max = float(heatmap.min()), float(heatmap.max())
    if h_max - h_min < 1e-8:
        logger.warning(
            "Grad-CAM produced a near-zero heatmap (range %.2e). "
            "Returning zero heatmap.", h_max - h_min,
        )
        return np.zeros_like(heatmap, dtype="float32")
    return ((heatmap - h_min) / (h_max - h_min)).astype("float32")


def _resize_heatmap(heatmap: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize 2D heatmap to (height, width) via PIL bilinear interpolation."""
    from PIL import Image as PILImage
    tgt_h, tgt_w = target_size
    if heatmap.shape == (tgt_h, tgt_w):
        return heatmap.astype("float32")
    uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
    resized = PILImage.fromarray(uint8, mode="L").resize((tgt_w, tgt_h), PILImage.BILINEAR)
    return np.array(resized, dtype="float32") / 255.0


def _heatmap_to_base64(heatmap: np.ndarray, colormap: str = "jet") -> str:
    """Convert (H, W) float32 heatmap to base64-encoded PNG using jet colormap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    cmap = plt.colormaps[colormap]
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(colored, mode="RGB").save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _create_overlay_base64(
    original_rgb,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> str:
    """Blend original image with jet-coloured heatmap. Returns base64 PNG."""
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    cmap = plt.colormaps[colormap]
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    orig_arr = np.array(original_rgb, dtype="float32")
    heat_arr = colored.astype("float32")
    blended = ((1 - alpha) * orig_arr + alpha * heat_arr).clip(0, 255).astype(np.uint8)

    buf = io.BytesIO()
    PILImage.fromarray(blended, mode="RGB").save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
