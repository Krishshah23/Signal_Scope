"""Inspect the trained model architecture for Grad-CAM layer selection."""
import sys
sys.path.insert(0, 'c:/Users/Admin/Signal_scope/model')
import tensorflow as tf

model = tf.keras.models.load_model(
    'c:/Users/Admin/Signal_scope/model/weights/signalscope_baseline.keras'
)

print("=== TOP-LEVEL LAYERS ===")
for i, layer in enumerate(model.layers):
    print(f"  {i}  {layer.name!r:50s}  {type(layer).__name__}")

print("\n=== MobileNetV2 INTERNAL LAYERS (last 20) ===")
mobilenet = model.layers[1]  # index 1 is mobilenetv2_1.00_128
for layer in mobilenet.layers[-20:]:
    print(f"  {layer.name!r:55s}  {type(layer).__name__}")

print("\n=== Conv/BN/Activation layers near the end ===")
for layer in mobilenet.layers:
    t = type(layer).__name__
    if t in ('Conv2D', 'DepthwiseConv2D', 'BatchNormalization', 'ReLU', 'Activation'):
        pass  # collect
conv_layers = [l for l in mobilenet.layers if type(l).__name__ in ('Conv2D', 'DepthwiseConv2D')]
print(f"Total conv layers: {len(conv_layers)}")
if conv_layers:
    last = conv_layers[-1]
    print(f"Last conv layer: {last.name!r}  type={type(last).__name__}")

# Also check what layer comes just before GlobalAveragePooling2D
print("\n=== Layer just before GAP ===")
# The model output before GAP: model layers are input -> mobilenet -> GAP -> Dense
# Inside mobilenet, we want the final spatial feature map
# Find layers named like the MobileNetV2 output
for layer in mobilenet.layers[-10:]:
    print(f"  {layer.name!r:55s}  {type(layer).__name__}")
