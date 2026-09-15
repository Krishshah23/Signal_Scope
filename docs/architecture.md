# SignalScope — Architecture Documentation

*Sessions 1–8 complete. This reflects the final implemented system.*

---

## Overview

SignalScope is a three-layer system:

1. **React Frontend** — image upload, result display, heatmap visualisation
2. **Flask REST API** — request validation, routing, model orchestration
3. **ML Model** — MobileNetV2 inference + Grad-CAM explainability

---

## System Diagram

```
┌─────────────────────────────────────────────────────┐
│                  User's Browser                     │
│                                                     │
│   ┌─────────────────────────────────────────────┐   │
│   │         React Frontend (Vite 5)             │   │
│   │                                             │   │
│   │  UploadPanel → api.js → ResultPanel         │   │
│   │  (drag-drop)  (fetch)  (verdict+heatmap)    │   │
│   └──────────────────┬──────────────────────────┘   │
└──────────────────────│──────────────────────────────┘
                       │
          POST /api/v1/predict/  (multipart/form-data)
          GET  /api/v1/health/
                       │
┌──────────────────────▼──────────────────────────────┐
│                  Flask REST API                      │
│                  src/backend/                        │
│                                                      │
│  app.py (factory) → routes/predict.py               │
│    1. Validate file extension + MIME + size          │
│    2. Save to temp file                              │
│    3. ModelService.run(tmp_path)                     │
│    4. Clean up temp file                             │
│    5. Return JSON response                           │
│                                                      │
│  services/model_service.py                           │
│    → calls predict(image_path)                       │
└──────────────────────│──────────────────────────────┘
                       │
                       │  Python function call
                       │
┌──────────────────────▼──────────────────────────────┐
│                  ML Model Layer                      │
│                  model/                              │
│                                                      │
│  predict.py                                          │
│    └── predict(image_path) → dict                    │
│          ├── _load_model()   [cached on first call]  │
│          ├── _preprocess_image()                     │
│          │     PIL open → RGB → resize 128×128       │
│          │     normalise /255.0 → [0,1]              │
│          ├── model.predict()                         │
│          │     raw sigmoid probability               │
│          ├── generate_gradcam()                      │
│          │     logit-based gradients                 │
│          │     out_relu feature maps                 │
│          │     base64 PNG overlay                    │
│          └── _build_result()                         │
│                verdict + confidence + heatmap        │
│                                                      │
│  weights/signalscope_baseline.keras  (~9.6 MB)       │
└─────────────────────────────────────────────────────┘
```

---

## API Contract

### `GET /api/v1/health/`
```json
{ "status": "ok", "service": "SignalScope API", "version": "v1" }
```

### `POST /api/v1/predict/`
Request: multipart/form-data, field `image`, max 16 MB, JPEG/PNG/WebP/BMP

Success (200):
```json
{
  "success": true,
  "result": {
    "verdict":     "likely AI-generated" | "likely real",
    "confidence":  0.0 – 1.0,
    "raw_prob":    0.0 – 1.0,
    "heatmap":     "<base64 PNG>" | null,
    "explanation": "<string>" | null
  }
}
```

Error (4xx / 5xx):
```json
{ "success": false, "error": "...", "message": "..." }
```

---

## ML Architecture

```
Input (128×128×3)
  → [internal] mobilenet_v2.preprocess_input
  → MobileNetV2 (frozen, ImageNet, 2.26M non-trainable params)
  → GlobalAveragePooling2D
  → Dense(1, sigmoid)  ← 1,281 trainable params

Threshold: raw_prob ≥ 0.5 → "likely AI-generated"
           raw_prob < 0.5 → "likely real"
```

### Grad-CAM
- Target layer: `out_relu` (4×4×1280 spatial features)
- Gradient source: pre-sigmoid logit (avoids sigmoid saturation)
- Output: RGB PNG overlay at 128×128

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Flask over Django | Lightweight; no ORM/DB needed |
| Versioned API `/api/v1/` | Stability for frontend contract |
| `model/predict.py` as sole ML interface | Flask never imports TensorFlow directly |
| Vite proxy `/api/*` → Flask | Avoids CORS issues in development |
| Logit-based Grad-CAM | Avoids vanishing gradients from sigmoid saturation |
| Model cached in module scope | No reload per request |
| Temp file cleanup in `finally` | No user data accumulation |
| CSS Modules | No extra UI library dependencies |

---

## Evaluation Results (Held-Out Test Set, 20,000 images)

| Metric | Value |
|---|---|
| Accuracy | 66.4% |
| ROC-AUC | 0.760 |
| Macro-F1 | 0.649 |

Confusion matrix: `report/confusion_matrix.png`
Full JSON: `report/evaluation_results.json`

---

## Session History

| Session | Milestone |
|---|---|
| 1 | Project foundation, Flask skeleton, React shell |
| 2 | CIFAKE dataset, MobileNetV2 training (6 epochs) |
| 3 | Held-out test evaluation (20k images) |
| 4 | Real model inference connected to Flask |
| 5 | Grad-CAM explainability (logit-based) |
| 6 | Frontend ↔ Flask end-to-end integration |
| 7 | Robustness QA (small sample, 4 transforms) |
| 8 | Documentation, final audit, submission |
