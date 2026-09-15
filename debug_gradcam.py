"""Debug: Grad-CAM with logit (pre-sigmoid) and try multiple samples."""
import sys, os
sys.path.insert(0, 'c:/Users/Admin/Signal_scope/model')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

model = tf.keras.models.load_model(
    'c:/Users/Admin/Signal_scope/model/weights/signalscope_baseline.keras'
)

mobilenet = model.layers[1]
gap_layer = model.layers[2]
dense_layer = model.layers[3]
feat_model = tf.keras.Model(inputs=mobilenet.input, outputs=mobilenet.get_layer('out_relu').output)

# Build a logit model (no sigmoid) for stable gradient computation
dense_weights, dense_bias = dense_layer.get_weights()
print(f"Dense weights shape: {dense_weights.shape}  bias: {dense_bias}")

def compute_gradcam(arr, label=""):
    """Compute Grad-CAM using logit (before sigmoid activation) for stability."""
    img_tensor = tf.constant(arr[np.newaxis, ...])
    feats_np = feat_model(img_tensor, training=False).numpy()   # (1, 4, 4, 1280)
    feats_var = tf.Variable(feats_np)

    with tf.GradientTape() as tape:
        # Use manual GAP and logit computation
        gap_out = tf.reduce_mean(feats_var, axis=[1, 2])   # (1, 1280)
        logit = tf.linalg.matmul(gap_out, dense_weights) + dense_bias  # (1, 1) pre-sigmoid
        pred_sigmoid = tf.sigmoid(logit)
        # Use LOGIT for gradient (avoids saturation at sigmoid extremes)
        logit_val = logit[0, 0]

    grads = tape.gradient(logit_val, feats_var)
    pred_val = float(pred_sigmoid.numpy()[0, 0])
    print(f"  {label}: pred={pred_val:.4f}  logit={float(logit.numpy()[0,0]):.4f}  grads nonzero={np.any(grads.numpy() != 0)}")
    if np.any(grads.numpy() != 0):
        pooled = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
        weighted = feats_np[0] * pooled
        heatmap = np.maximum(np.sum(weighted, axis=-1), 0)
        print(f"    heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        return heatmap, pred_val
    return None, pred_val

# Test multiple FAKE samples
fake_dir = Path('c:/Users/Admin/Signal_scope/train/FAKE')
real_dir = Path('c:/Users/Admin/Signal_scope/train/REAL')

print("FAKE samples:")
for f in sorted(fake_dir.iterdir())[:5]:
    arr = np.array(Image.open(f).convert("RGB").resize((128,128), Image.BILINEAR), dtype="float32") / 255.0
    hm, pred = compute_gradcam(arr, f.name[:20])

print("\nREAL samples:")
for f in sorted(real_dir.iterdir())[:5]:
    arr = np.array(Image.open(f).convert("RGB").resize((128,128), Image.BILINEAR), dtype="float32") / 255.0
    hm, pred = compute_gradcam(arr, f.name[:20])
