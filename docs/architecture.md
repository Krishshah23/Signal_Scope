# SignalScope — Architecture Documentation

> Last updated: Session 1 — Project Foundation

---

## Overview

SignalScope is a three-layer system:

1. **React Frontend** — image upload UI, result display
2. **Flask REST API** — request validation, routing, model orchestration
3. **ML Model** — TensorFlow/Keras classifier (transfer learning)

Each layer has a single, well-defined responsibility.
They communicate exclusively through the stable public interfaces described below.

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────┐
│                  User's Browser                     │
│                                                     │
│   ┌─────────────────────────────────────────────┐   │
│   │         React Frontend                      │   │
│   │         src/frontend/                       │   │
│   │                                             │   │
│   │  UploadPanel  →  api.js  →  ResultPanel     │   │
│   │  (file select)   (fetch)   (verdict/heatmap)│   │
│   └──────────────────┬──────────────────────────┘   │
└──────────────────────│──────────────────────────────┘
                       │
          HTTP / REST (multipart/form-data)
          POST /api/v1/predict/
          GET  /api/v1/health/
                       │
┌──────────────────────▼──────────────────────────────┐
│                  Flask REST API                      │
│                  src/backend/                        │
│                                                      │
│  app.py (factory)                                    │
│    │                                                 │
│    ├── GET  /api/v1/health/   → routes/health.py     │
│    │                                                 │
│    └── POST /api/v1/predict/  → routes/predict.py    │
│              │                                       │
│              │  1. Validate upload (ext, MIME, size) │
│              │  2. Save to temp file                 │
│              │  3. Call ModelService.run(path)       │  ← future
│              │  4. Return JSON response              │
│              │                                       │
│         services/model_service.py                    │
│              │                                       │
│              │  ModelService.run(image_path)         │
└──────────────│──────────────────────────────────────┘
               │
               │  Python function call only
               │  (no HTTP, no subprocess)
               │
┌──────────────▼──────────────────────────────────────┐
│                  ML Model Layer                      │
│                  model/                              │
│                                                      │
│  predict.py                                          │
│    └── predict(image_path) → dict                    │
│              │                                       │
│              ├── _preprocess_image()                 │  ← future
│              │     resize 128×128, normalise [0,1]   │
│              │                                       │
│              ├── _load_model()                       │  ← future
│              │     tf.keras.models.load_model()      │
│              │                                       │
│              └── _build_result(raw_prob)             │  ← future
│                    verdict + confidence              │
│                                                      │
│  weights/signalscope.h5   (gitignored, future)       │
└─────────────────────────────────────────────────────┘
```

---

## Implementation Status per Layer

### React Frontend — `src/frontend/`

| Component | Status | Notes |
|---|---|---|
| `index.html` / `main.jsx` | ✅ Done | Vite + React 18 entry point |
| `App.jsx` | ✅ Done | State machine: idle → loading → result/error |
| `Header.jsx` | ✅ Done | Brand + tagline |
| `UploadPanel.jsx` | ✅ Done | Drag-drop, preview, file selector, analyse button |
| `ResultPanel.jsx` | ✅ Done | Error, dev-stub (501), real result (future), heatmap slot |
| `Footer.jsx` | ✅ Done | SIH attribution, disclaimer |
| `services/api.js` | ✅ Done | `analyseImage()`, `checkHealth()` — all fetch calls centralised |
| Grad-CAM heatmap display | 🔲 Planned | `ResultPanel` has the slot ready; activated when API sends `heatmap` |
| Frontend tests | 🔲 Planned | Vitest + React Testing Library |

### Flask Backend — `src/backend/`

| Component | Status | Notes |
|---|---|---|
| `app.py` — application factory | ✅ Done | CORS, blueprints, global error handlers |
| `config.py` — env configuration | ✅ Done | Dev / Testing / Production classes |
| `GET /api/v1/health/` | ✅ Done | Always returns 200 `{"status":"ok"}` |
| `POST /api/v1/predict/` — upload validation | ✅ Done | Extension, MIME, presence, size (16 MB) |
| `POST /api/v1/predict/` — inference | 🔲 Planned (501) | Returns 501 until model is connected |
| `services/model_service.py` | ✅ Done (stub) | `ModelService.run()` raises `ModelNotReadyError` |
| Backend tests — health | ✅ Done | pytest, 3 tests |
| Backend tests — predict validation | ✅ Done | pytest, 6 tests |
| Real inference integration | 🔲 Planned | Session 3 |

### ML Model — `model/`

| Component | Status | Notes |
|---|---|---|
| `predict.py` — public interface | ✅ Done | `predict(image_path)` defined, raises `ModelNotTrainedError` |
| `ModelNotTrainedError` | ✅ Done | Clear error until training milestone |
| `InvalidImageError` | ✅ Done | For future preprocessing validation |
| Constants: `IMAGE_SIZE`, `VERDICT_*`, `THRESHOLD` | ✅ Done | Shared across modules |
| Private stubs | ✅ Done | `_load_model`, `_preprocess_image`, `_build_result` |
| Dataset preparation | 🔲 Planned | Session 2 |
| Model training | 🔲 Planned | Session 2 |
| Real `predict()` implementation | 🔲 Planned | Session 3 |
| Grad-CAM | 🔲 Planned | Session 4 |
| Robustness testing | 🔲 Planned | Session 5 |

---

## API Contract

### `GET /api/v1/health/`

```
Request:  GET /api/v1/health/

Response 200:
{
    "status":  "ok",
    "service": "SignalScope API",
    "version": "v1"
}
```

### `POST /api/v1/predict/`

```
Request:
  Content-Type: multipart/form-data
  Field "image": <image file>   JPEG | PNG | WebP | BMP, max 16 MB

Response 501 (current — model not trained):
{
    "status":   "not_implemented",
    "message":  "...",
    "filename": "<uploaded filename>"
}

Response 200 (future — once model is connected):
{
    "verdict":     "likely AI-generated" | "likely real",
    "confidence":  0.0 – 1.0,
    "heatmap":     "<base64 PNG>" | null,
    "explanation": "<string>" | null
}

Error responses:
  400  missing_file          — no 'image' field in request
  400  empty_filename        — filename is empty string
  413  file_too_large        — exceeds 16 MB
  415  unsupported_file_type — extension not in allowed list
  415  unsupported_mime_type — MIME type not in allowed list
  500  server_error          — unhandled exception
```

**Important:** `confidence` represents certainty of the stated `verdict`,
not a raw sigmoid probability. A confidence of 0.87 for `"likely real"`
means the model is 87% confident the image is real — not 87% confident it
is AI-generated. Predictions are probabilistic estimates, never certainties.

---

## ML Model Architecture

```
Input
  └── Raw image file (JPEG / PNG / WebP / BMP, any resolution)

Preprocessing  [_preprocess_image() — future]
  └── Resize to 128 × 128
  └── Convert to float32
  └── Normalise pixel values to [0, 1]
  └── Add batch dimension → shape (1, 128, 128, 3)

Pretrained Base  [future]
  └── MobileNetV2 or EfficientNetB0
  └── Weights: ImageNet pretrained
  └── Base layers: frozen (fine-tuning may be added later)

Head
  └── GlobalAveragePooling2D
  └── Dense(1, activation='sigmoid')
  └── Output: single float in [0, 1]

Decision  [_build_result() — future]
  └── raw_prob >= 0.5  →  verdict = "likely AI-generated"
  └── raw_prob <  0.5  →  verdict = "likely real"
  └── confidence = raw_prob if AI else (1 - raw_prob)
```

### Why MobileNetV2 / EfficientNetB0?

- Both are lightweight enough to run inference in real time on CPU.
- Both have strong ImageNet pretrained features that transfer well to
  image-forensics tasks.
- Final selection (MobileNetV2 vs EfficientNetB0) will be made after
  comparing validation ROC-AUC in the training milestone.

### Training Strategy (planned)

| Phase | Layers frozen | Epochs | Notes |
|---|---|---|---|
| 1 — feature extraction | All base layers | ~10 | Train head only |
| 2 — fine-tuning | Top N layers unfrozen | ~5–10 | Lower LR |

---

## Data Flow — End to End (Future State)

```
1. User selects image in browser
2. UploadPanel sends POST /api/v1/predict/ (multipart)
3. Flask validates extension, MIME type, file size
4. Flask saves file to secure temp path
5. ModelService.run(temp_path) is called
6. model/predict.py:predict(image_path) is called
7. _preprocess_image() resizes + normalises image
8. Model performs forward pass → raw sigmoid probability
9. _build_result() converts probability → verdict + confidence
10. (Grad-CAM milestone) heatmap generated, base64 encoded
11. Temp file deleted
12. JSON response returned to frontend
13. ResultPanel displays verdict, confidence bar, heatmap
```

---

## Security Decisions

| Decision | Rationale |
|---|---|
| Temp file deleted immediately after inference | No persistent user data storage |
| MIME type + extension double-check | Prevents simple extension-spoofing |
| 16 MB upload cap | Mitigates DoS via large file uploads |
| CORS restricted to `localhost:3000` in dev | Prevents cross-origin abuse |
| No secrets in source control | `.env` gitignored; `.env.example` contains placeholders only |

---

## Key Design Decisions

**1. Flask over Django**
Flask was chosen per the official roadmap. It is lighter weight and more
appropriate for a focused inference API with no database or ORM requirement.

**2. Versioned API prefix `/api/v1/`**
All routes are versioned from the start so the frontend is not broken if
breaking API changes are made in a future version.

**3. `model/predict.py` as the sole ML interface**
The Flask backend calls `predict(image_path)` only. It never imports
TensorFlow directly. This isolates the ML dependency from the web layer,
making the backend testable without a GPU or TensorFlow installation.

**4. Vite over Create React App**
Vite is faster, has a smaller footprint, and is better maintained in 2026.
CRA is no longer actively developed.

**5. CSS Modules over a UI component library**
Keeps the frontend dependency-free beyond React itself.
Avoids version conflicts and makes the UI fully hackathon-customisable.

**6. `ModelNotTrainedError` instead of fake predictions**
Returning fake confidence scores during development would violate the
project's core wording rule ("never over-claim certainty") and would
mislead evaluators. The 501 response is honest and clearly documented.

---

## Future Architecture Considerations

- **Grad-CAM:** Will require access to the last convolutional layer's
  activations and gradients. Implementation will be in `model/predict.py`
  as an optional `include_heatmap=True` parameter, keeping the public
  interface stable.
- **Robustness testing:** Will be a standalone script in `model/` that
  applies JPEG recompression and resizing transforms before calling
  `predict()`, reporting accuracy delta.
- **Production deployment:** Out of scope for the hackathon. If needed,
  the Flask app can be served with Gunicorn + Nginx. The React build
  output (`dist/`) can be served as static files.
