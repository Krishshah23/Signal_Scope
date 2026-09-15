"""
SignalScope — Lightweight Robustness Check
-------------------------------------------
Tests prediction stability on a small sample of images under common
image transformations. This is NOT the official held-out evaluation
(which is in model/evaluate.py and used all 20,000 images).

Transformations tested:
  - Original (baseline)
  - JPEG compression (quality=50)
  - Resize to 64x64 then back to 128x128 (simulates downscale)
  - Mild brightness increase (+30/255)
  - Mild brightness decrease (-30/255)

For each sample, the delta in raw_prob vs the original is recorded.
If abs(delta) < 0.10, the prediction is considered "stable".

Usage:
    .venv/Scripts/python model/evaluate_robustness.py
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("signalscope.robustness")

_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODEL_DIR.parent
sys.path.insert(0, str(_MODEL_DIR))

REPORT_PATH = _PROJECT_ROOT / "report" / "robustness_results.json"
WEIGHTS_PATH = _PROJECT_ROOT / "model" / "weights" / "signalscope_baseline.keras"

# Small sample: 5 FAKE + 5 REAL images
SAMPLE_SIZE = 5
STABILITY_THRESHOLD = 0.10   # |delta| below this → stable

TRANSFORMATIONS = ["original", "jpeg_q50", "resize_64", "brightness_plus", "brightness_minus"]


def _load_and_predict(image_path: Path, weights_path: Path) -> float:
    """Return the raw sigmoid probability for the image at image_path."""
    from predict import predict
    result = predict(str(image_path), weights_path=str(weights_path))
    return result["raw_prob"]


def _apply_transform(image_path: Path, transform: str) -> Path:
    """
    Apply a transformation to the image and save to a temporary file.
    Returns the temp path (which must be deleted by the caller).
    """
    import tempfile
    from PIL import Image, ImageEnhance

    img = Image.open(image_path).convert("RGB")

    if transform == "original":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(tmp.name, format="JPEG", quality=95)
        return Path(tmp.name)

    if transform == "jpeg_q50":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(tmp.name, format="JPEG", quality=50)
        return Path(tmp.name)

    if transform == "resize_64":
        small = img.resize((64, 64), Image.BILINEAR)
        back = small.resize((128, 128), Image.BILINEAR)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        back.save(tmp.name, format="JPEG", quality=95)
        return Path(tmp.name)

    if transform == "brightness_plus":
        arr = np.array(img, dtype="float32")
        arr = np.clip(arr + 30, 0, 255).astype(np.uint8)
        bright_img = Image.fromarray(arr, mode="RGB")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        bright_img.save(tmp.name, format="JPEG", quality=95)
        return Path(tmp.name)

    if transform == "brightness_minus":
        arr = np.array(img, dtype="float32")
        arr = np.clip(arr - 30, 0, 255).astype(np.uint8)
        dark_img = Image.fromarray(arr, mode="RGB")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        dark_img.save(tmp.name, format="JPEG", quality=95)
        return Path(tmp.name)

    raise ValueError(f"Unknown transform: {transform}")


def run_robustness(
    test_dir: Path | None = None,
    weights_path: Path | None = None,
    sample_size: int = SAMPLE_SIZE,
) -> dict:
    """
    Run the lightweight robustness check.

    Parameters
    ----------
    test_dir : Path, optional
        Root of the test dataset. Defaults to project_root/test/.
    weights_path : Path, optional
        Path to the .keras model weights.
    sample_size : int
        Number of FAKE and REAL images to sample.

    Returns
    -------
    dict  Full robustness report (also saved to report/robustness_results.json).
    """
    import os

    test_root = test_dir or (_PROJECT_ROOT / "test")
    w_path = weights_path or WEIGHTS_PATH

    if not w_path.exists():
        raise FileNotFoundError(f"Model weights not found: {w_path}")
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    logger.info("Starting robustness check — %d FAKE + %d REAL samples",
                sample_size, sample_size)

    samples = {
        "FAKE": sorted((test_root / "FAKE").iterdir())[:sample_size],
        "REAL": sorted((test_root / "REAL").iterdir())[:sample_size],
    }

    results_per_image = []
    transform_stability = {t: {"stable": 0, "unstable": 0} for t in TRANSFORMATIONS[1:]}

    for class_name, paths in samples.items():
        true_label = 1 if class_name == "FAKE" else 0
        for img_path in paths:
            image_result = {
                "file": img_path.name,
                "class": class_name,
                "true_label": true_label,
                "transforms": {},
            }

            # Baseline
            orig_tmp = _apply_transform(img_path, "original")
            try:
                baseline_prob = _load_and_predict(orig_tmp, w_path)
            finally:
                os.unlink(orig_tmp)

            image_result["transforms"]["original"] = {
                "raw_prob": round(baseline_prob, 4),
                "verdict": "likely AI-generated" if baseline_prob >= 0.5 else "likely real",
            }
            logger.info("  %s/%s  baseline=%.4f", class_name, img_path.name, baseline_prob)

            # Other transforms
            for transform in TRANSFORMATIONS[1:]:
                tmp = _apply_transform(img_path, transform)
                try:
                    prob = _load_and_predict(tmp, w_path)
                finally:
                    os.unlink(tmp)

                delta = abs(prob - baseline_prob)
                stable = delta < STABILITY_THRESHOLD
                image_result["transforms"][transform] = {
                    "raw_prob": round(prob, 4),
                    "delta_vs_original": round(delta, 4),
                    "stable": stable,
                    "verdict": "likely AI-generated" if prob >= 0.5 else "likely real",
                }
                if stable:
                    transform_stability[transform]["stable"] += 1
                else:
                    transform_stability[transform]["unstable"] += 1

            results_per_image.append(image_result)

    # Aggregate stability rates
    total_samples = sample_size * 2  # FAKE + REAL
    stability_summary = {}
    for transform, counts in transform_stability.items():
        pct = counts["stable"] / total_samples * 100 if total_samples > 0 else 0
        stability_summary[transform] = {
            "stable": counts["stable"],
            "unstable": counts["unstable"],
            "total": total_samples,
            "stability_rate_pct": round(pct, 1),
        }
        logger.info(
            "  %s stability: %d/%d (%.1f%%)",
            transform, counts["stable"], total_samples, pct,
        )

    report = {
        "status": "completed",
        "note": (
            "Lightweight robustness check on a small sample. "
            "This is NOT the official held-out evaluation (accuracy=0.664, "
            "ROC-AUC=0.760 on 20,000 images — see report/evaluation_results.json)."
        ),
        "configuration": {
            "sample_size_per_class": sample_size,
            "total_images": total_samples,
            "stability_threshold": STABILITY_THRESHOLD,
            "transformations": TRANSFORMATIONS,
        },
        "stability_summary": stability_summary,
        "per_image_results": results_per_image,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Robustness report saved to: %s", REPORT_PATH)

    return report


if __name__ == "__main__":
    result = run_robustness()
    print("\nRobustness Summary:")
    for transform, stats in result["stability_summary"].items():
        print(
            f"  {transform:20s}: {stats['stability_rate_pct']:.1f}% stable "
            f"({stats['stable']}/{stats['total']})"
        )
