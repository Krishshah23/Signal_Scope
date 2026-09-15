# SignalScope — Model

This directory contains the machine learning pipeline for SignalScope.

---

## Status

All milestones complete:

| Component | Status |
|---|---|
| Training pipeline (`train.py`) | ✅ Done |
| Prediction interface (`predict.py`) | ✅ Done |
| Grad-CAM explainability (`gradcam.py`) | ✅ Done |
| Held-out evaluation (`evaluate.py`) | ✅ Done |
| Robustness check (`evaluate_robustness.py`) | ✅ Done |

---

## Architecture

```
Input (128×128×3 RGB, normalised to [0,1])
        ↓
[Internal] mobilenet_v2.preprocess_input (→ [-1,1])
        ↓
MobileNetV2 (ImageNet weights, base FROZEN)
  Trainable params: 1,281
  Non-trainable params: 2,257,984
        ↓
GlobalAveragePooling2D
        ↓
Dense(1, activation='sigmoid')
        ↓
Sigmoid probability [0,1]
```

**Class convention:** `REAL = 0`, `FAKE = 1`

**Threshold:** `raw_prob >= 0.5` → `"likely AI-generated"`

---

## Prediction Interface

The public contract is:

```python
from model.predict import predict

result = predict("/path/to/image.jpg")
# or with explicit weights:
result = predict("/path/to/image.jpg", weights_path="/path/to/signalscope_baseline.keras")

# result:
# {
#   "verdict":     "likely AI-generated" | "likely real",
#   "confidence":  float,   # certainty of stated verdict (0-1)
#   "raw_prob":    float,   # raw sigmoid output (0-1)
#   "heatmap":     str | None,   # base64 PNG of Grad-CAM overlay
#   "explanation": str | None    # human-readable explanation
# }
```

**Notes:**
- Model is loaded lazily and cached on first call
- Grad-CAM is generated automatically via logit-based gradients
- If Grad-CAM fails, prediction still succeeds with `heatmap=None`

---

## Input Preprocessing

Matches the training pipeline exactly:

1. Load image with PIL, convert to RGB
2. Resize to 128×128 using BILINEAR interpolation
3. Normalise to [0, 1] by dividing by 255.0
4. Shape: `(1, 128, 128, 3)` float32
5. `mobilenet_v2.preprocess_input` is applied **inside the model graph**
   — do NOT apply it externally

---

## Grad-CAM Details

- **Target layer:** `out_relu` inside `mobilenetv2_1.00_128` (last spatial activation, 4×4×1280)
- **Gradient target:** pre-sigmoid logit (avoids vanishing gradients from sigmoid saturation)
- **Output:** base64-encoded RGB PNG overlay (heatmap blended onto resized input)
- **Interpretation:** highlighted regions influenced the model's prediction; not proof of manipulation

---

## Training Configuration

| Parameter | Value |
|---|---|
| Dataset | CIFAKE (100k train + 20k test) |
| Image size | 128×128 RGB |
| Batch size | 32 |
| Max epochs | 10 |
| Actual epochs | 6 (early stopping) |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Loss | binary crossentropy |
| Val split | 15% of train/ |
| Random seed | 42 |

---

## Held-Out Evaluation Results

Evaluated on all 20,000 images in `test/` — **not used for training or tuning**.

| Metric | Value |
|---|---|
| Accuracy | 66.4% |
| ROC-AUC | 0.760 |
| Macro-F1 | 0.649 |
| Precision (macro) | 0.699 |
| Recall (macro) | 0.664 |

Full results: `report/evaluation_results.json`

---

## Weights

`model/weights/signalscope_baseline.keras` (~9.6 MB) — gitignored.

To train from scratch:
```bash
.venv/Scripts/python model/train.py
```

---

## Dataset

- **CIFAKE** — Bird & Lotfi (2024), IEEE Access, CC BY 4.0
- [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- `train/FAKE/` and `train/REAL/` — 50,000 images each
- `test/FAKE/` and `test/REAL/` — 10,000 images each

Datasets are gitignored and must be placed at project root before training.
