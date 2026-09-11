# SignalScope — Model

This directory contains the machine learning model code for SignalScope.

---

## Current Status

**Session 1 — Foundation only.**

The prediction interface has been established. No model has been trained yet.
No weights exist. No inference is performed.

---

## Implemented

| Item | Status |
|---|---|
| `predict.py` — public `predict(image_path)` interface | ✅ Done |
| `ModelNotTrainedError` exception | ✅ Done |
| `InvalidImageError` exception | ✅ Done |
| Constants: `IMAGE_SIZE`, `VERDICT_AI`, `VERDICT_REAL`, `CLASSIFICATION_THRESHOLD` | ✅ Done |
| Architecture documentation in docstrings | ✅ Done |
| Private helper stubs for future implementation | ✅ Done |

---

## Planned (Future Milestones)

| Item | Milestone |
|---|---|
| Download and prepare CIFAKE dataset | Model Training |
| Collect disclosed-generator samples | Model Training |
| Implement `_preprocess_image()` — resize to 128×128, normalise | Model Training |
| Implement `_load_model()` — load `.h5` weights | Model Training |
| Train MobileNetV2 or EfficientNetB0 with frozen base | Model Training |
| Head: `GlobalAveragePooling2D → Dense(1, sigmoid)` | Model Training |
| Data augmentation (flip, rotation, brightness) | Model Training |
| Implement `_build_result()` — verdict + confidence | Model Training |
| Connect `predict()` to real inference | Model Training |
| Evaluate ROC-AUC, Macro-F1, confusion matrix | Metrics |
| Grad-CAM heatmap generation (last conv layer) | Grad-CAM |
| Robustness testing (JPEG recompression, resizing) | Robustness |
| Generalization testing on unseen generators | Robustness |

---

## Architecture

```
Input image (any size, JPEG/PNG/WebP/BMP)
        |
        v
_preprocess_image()
  - Resize to 128 × 128
  - Convert to float32
  - Normalise to [0, 1]
  - Add batch dimension → shape (1, 128, 128, 3)
        |
        v
Pretrained base (MobileNetV2 or EfficientNetB0)
  - Weights: ImageNet pretrained
  - Base layers: frozen initially
        |
        v
GlobalAveragePooling2D
        |
        v
Dense(1, activation='sigmoid')
  - Output: single probability in [0, 1]
  - ≥ 0.5 → "likely AI-generated"
  - <  0.5 → "likely real"
        |
        v
_build_result()
  - verdict: "likely AI-generated" | "likely real"
  - confidence: probability of the stated verdict
        |
        v
predict() return dict
```

**Important:** Predictions are probabilistic estimates, not certainties.
Results are always described as **"likely AI-generated"** or **"likely real"**.
The system never claims 100% certainty.

---

## Public Interface

```python
from model.predict import predict

result = predict(image_path="/path/to/image.jpg")

# result = {
#     "verdict":     "likely AI-generated",  # or "likely real"
#     "confidence":  0.87,                   # 0.0 – 1.0
#     "raw_prob":    0.87,                   # raw sigmoid output
#     "heatmap":     None,                   # base64 PNG (future)
#     "explanation": None,                   # text description (future)
# }
```

---

## Weights Directory

`model/weights/` is reserved for trained model weights.

**Do not commit weights to Git.** Add weight files to `.gitignore`.
Large model files should be shared via a separate mechanism
(Google Drive, HuggingFace Hub, or equivalent).

---

## Dataset

The model will be trained on:

- **CIFAKE** — a publicly available dataset of AI-generated and real images
  ([CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images))
- **Disclosed-generator samples** — images from known AI generators

Dataset files are **not committed to Git**. See the project README for setup instructions.

---

## References

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.*
- Howard, A. et al. (2017). *MobileNets: Efficient Convolutional Neural Networks.*
- Tan, M. & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling.*
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.*
- CIFAKE dataset: [kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
