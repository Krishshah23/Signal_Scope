# SignalScope — Project Documentation

**Department of Computer Science and Engineering**
**B.Tech Project Submission | Academic Year 2025–26**

---

## 1. Project Title & Overview

**Project Name:** SignalScope

**One-Line Description:** A web-based tool that analyses an uploaded image and returns a probabilistic verdict on whether it is AI-generated or a real photograph.

**Problem:** AI image generators have reached a level of realism where visual inspection alone is no longer reliable for detecting synthetic media. Journalists, researchers, and educators increasingly face situations where the authenticity of an image cannot be confirmed at a glance.

**Solution:** SignalScope uses a transfer-learned MobileNetV2 binary classifier trained on the CIFAKE dataset to estimate the probability that an image is AI-generated. Each prediction is accompanied by a Grad-CAM heatmap that visually highlights the regions of the image that most influenced the model's decision, giving the result a degree of interpretability rather than acting as a black box.

---

## 2. Objectives

- Train a binary image classifier capable of distinguishing AI-generated images from real photographs using transfer learning on a labelled benchmark dataset.
- Build a REST API backend that accepts image uploads, validates them, runs inference through the trained model, and returns structured JSON responses.
- Implement Grad-CAM (Gradient-weighted Class Activation Mapping) to produce a spatial explanation heatmap for every prediction.
- Develop a React-based frontend with drag-and-drop upload, confidence visualisation, and Grad-CAM heatmap rendering.
- Evaluate the model on a fully held-out test set of 20,000 images that was never exposed during training or validation.
- Ensure the system always describes results as probabilistic estimates and never presents a prediction as a definitive conclusion.

---

## 3. Key Features

- **Probabilistic prediction:** Every result returns a verdict ("likely AI-generated" or "likely real"), a confidence score (0–100%), and the raw sigmoid probability. The system is explicitly designed to avoid certainty language.
- **Grad-CAM explainability:** A heatmap overlay is generated on every inference call. It highlights spatial regions of the image that drove the model's prediction, making the output interpretable rather than opaque.
- **Validated REST API:** The Flask backend validates file extension, MIME type, and file size (16 MB limit) before running inference. All error paths return structured JSON with descriptive codes (400, 415, 422, 503, 500).
- **Drag-and-drop UI:** The React frontend supports drag-and-drop upload, image preview, an animated confidence bar, colour-coded verdict badge, and inline heatmap rendering.
- **Data integrity enforcement:** A `verify_no_test_in_train()` check in the training pipeline confirms the test and train directories are fully separate before any training begins, preventing data leakage.
- **Efficient model serving:** The model is loaded once on the first inference request and cached at module scope. Subsequent requests reuse the cached model with no re-loading overhead.

---

## 4. Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, Vite 5, CSS Modules |
| **Backend** | Python 3.12, Flask 3.0, flask-cors |
| **ML Framework** | TensorFlow 2.18 / Keras 3 |
| **Pretrained Model** | MobileNetV2 (ImageNet weights, frozen backbone) |
| **Explainability** | Grad-CAM — logit-based gradient computation via `tf.GradientTape` |
| **Dataset** | CIFAKE — 100k training + 20k test images (Bird & Lotfi, 2024, CC BY 4.0) |
| **Evaluation** | scikit-learn (accuracy, ROC-AUC, macro-F1, precision, recall, confusion matrix) |
| **Image I/O** | Pillow (PIL), NumPy |
| **Visualisation** | Matplotlib (headless Agg backend for PNG generation) |
| **API Communication** | Fetch API (multipart/form-data), Vite dev proxy for CORS-free development |

---

## 5. System Workflow

The system has three independent layers: a React frontend, a Flask REST API, and a Python ML model module.

**End-to-end flow:**

1. The user drags an image into the upload panel or selects one via file picker.
2. The frontend POSTs the image to `POST /api/v1/predict/` as `multipart/form-data`.
3. Flask validates the file (extension, MIME type, size), saves it to a temporary file, and calls `ModelService.run(tmp_path)`. The temp file is deleted in a `finally` block regardless of outcome.
4. `ModelService` calls `predict(image_path)` from `model/predict.py` — the sole interface between Flask and TensorFlow.
5. `predict()` lazily loads and caches the model, preprocesses the image (PIL → RGB → resize 128×128 → normalise to [0,1] → float32 tensor), runs `model.predict()`, and calls `generate_gradcam()`.
6. Grad-CAM extracts feature maps from the `out_relu` layer, computes gradient of the pre-sigmoid logit, applies weighted activation pooling, and returns a base64-encoded heatmap overlay.
7. The result dict — `verdict`, `confidence`, `raw_prob`, `heatmap` (base64 PNG), `explanation` — is returned as JSON.
8. The React `ResultPanel` renders the verdict badge, confidence bar, raw probability, and heatmap image inline.

**Key architectural isolation:** Flask never imports TensorFlow. All ML logic lives inside `model/`, and the `ModelService` layer acts as the translation boundary — mapping domain exceptions (`ModelNotTrainedError`, `InvalidImageError`) to HTTP status codes (503, 422).

---

## 6. AI/ML Implementation

**Model Architecture:**
```
Input (128 × 128 × 3 RGB)
  → mobilenet_v2.preprocess_input  [normalises to [-1, 1] internally]
  → MobileNetV2 (ImageNet pretrained — ALL 2,257,984 parameters FROZEN)
  → GlobalAveragePooling2D
  → Dense(1, activation='sigmoid')      [1,281 trainable parameters]
  → Sigmoid probability → FAKE if ≥ 0.5, REAL if < 0.5
```

**Training Setup:**
- Dataset: CIFAKE FAKE/REAL split, 85,000 training / 15,000 validation (test set held out entirely)
- Optimiser: Adam (lr = 1e-3), Loss: binary crossentropy
- Callbacks: `EarlyStopping(patience=3, restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.5, patience=2)`
- Training data augmentation: horizontal flip, ±5° rotation, ±5% zoom
- Training ran for 6 epochs before early stopping; learning rate halved at epoch 6

**Grad-CAM Implementation (`model/gradcam.py`):**
The target layer is `out_relu` inside the MobileNetV2 sub-model — the last spatial activation before `GlobalAveragePooling2D`, producing `(1, 4, 4, 1280)` feature maps. A key design decision was computing gradients against the **pre-sigmoid logit** rather than the sigmoid output. When a model is highly confident, the sigmoid derivative `σ(x)(1−σ(x))` approaches zero, causing gradients to vanish. Using the raw logit avoids this saturation and produces meaningful spatial gradients at any confidence level. The resulting heatmap is ReLU-clipped, normalised, resized to 128×128, and blended onto the original image as a jet-colourmap overlay at 45% alpha before base64 encoding.

---

## 7. Implementation Highlights

- **Transfer learning with frozen backbone:** The entire MobileNetV2 backbone remains frozen during training. Only the 1,281 parameters in the classification head are updated. Architecture correctness is enforced at build time via `_assert_architecture()`, which checks input shape, backbone name, frozen state, layer types, and output shape — failing fast if any condition is violated.
- **Data isolation enforcement:** `verify_no_test_in_train()` compares resolved absolute paths of training and test directories before `model.fit()` is ever called. Training aborts with a `RuntimeError` if they overlap.
- **Label remapping:** CIFAKE's directory-based loader maps labels alphabetically (`FAKE=0, REAL=1`). The training pipeline explicitly remaps this to the project convention (`FAKE=1, REAL=0`) via a `remap_labels` step applied to every batch, preventing silent label inversion.
- **Temp file hygiene:** The backend saves uploaded files to `tempfile.NamedTemporaryFile` and deletes them in a `finally` block, ensuring no user-uploaded image persists on disk after a request completes.
- **Non-fatal Grad-CAM:** Grad-CAM generation is wrapped in a try/except inside `predict()`. If it fails for any reason, the prediction result is still returned with `heatmap=None` and `explanation=None` — inference is never blocked by an explainability failure.
- **Versioned API surface:** All routes are registered under `/api/v1/`, keeping the frontend contract stable.

---

## 8. Results & Outcome

**Held-out test evaluation — 20,000 images (CIFAKE test split, never used during training):**

| Metric | Value |
|---|---|
| Accuracy | 66.4% |
| ROC-AUC | 0.760 |
| Macro-F1 | 0.649 |
| Precision (macro) | 0.699 |
| Recall (macro) | 0.664 |

**Confusion matrix breakdown:**
- Real images correctly identified (REAL recall): 87.4% (8,737 / 10,000)
- AI images correctly identified (FAKE recall): 45.4% (4,543 / 10,000)

The model is conservative — it leans toward predicting "real." This is a characteristic behaviour of a frozen-backbone baseline trained for limited epochs on CIFAR-10-scale images. The **ROC-AUC of 0.760** indicates the model has meaningful discriminative ability beyond random chance.

A lightweight robustness check on 10 images across four transforms (JPEG compression at q=50, 64→128 resize, ±brightness shift) showed **100% prediction stability** — all transformed images produced verdicts within 0.1 raw probability of the originals.

The final deliverable is a functional end-to-end system: a user can upload any JPEG/PNG/WebP/BMP image through the browser interface and receive a verdict with confidence score and a spatial explanation heatmap within seconds.

---

## 9. Future Scope

- **Partial fine-tuning:** Unfreeze and fine-tune the top layers of the MobileNetV2 backbone to let the model learn image-authenticity-specific features rather than relying purely on ImageNet representations, which would likely improve FAKE recall significantly.
- **Broader training data:** Extend the dataset to include outputs from modern high-resolution generators (Stable Diffusion, DALL-E, Midjourney) at native resolutions, replacing or augmenting the 32×32-origin CIFAKE images.
- **Robustness training:** Add JPEG compression and downscale/upscale augmentation during training to improve resistance to these common transforms, which are often applied to images before they circulate online.
- **Asynchronous inference:** Move model inference to a background task queue (e.g., Celery + Redis) to keep the API non-blocking under concurrent load.
- **Batch analysis:** Support multi-image uploads and batch result export (CSV / PDF) for use cases such as fact-checking workflows where multiple images need to be evaluated together.

---

## 10. Conclusion

SignalScope is a complete, working implementation of an AI-generated image detection system. It combines transfer learning on MobileNetV2 with Grad-CAM explainability to produce verdicts that are both quantified and spatially grounded. The architecture cleanly separates the ML layer, REST API, and frontend into independent modules with well-defined contracts, and the training pipeline includes explicit safeguards against data leakage. While the baseline test accuracy of 66.4% reflects the known difficulty of CIFAKE-scale detection and the constraints of a frozen backbone, the 0.760 ROC-AUC and full-stack functionality demonstrate a solid foundation that can be extended toward production-grade performance with fine-tuning and richer training data.

---

*Dataset citation: Bird, J.J. & Lotfi, A. (2024). CIFAKE: Real and AI-Generated Synthetic Images. IEEE Access. CC BY 4.0.*
