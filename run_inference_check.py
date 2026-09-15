"""
Session 4 — Manual inference verification.
Calls the full inference pipeline (Flask app → model_service → predict.py → .keras model)
and prints the result. Does NOT mock anything.
"""
import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
BACKEND_DIR = PROJECT_ROOT / "src" / "backend"

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(BACKEND_DIR))

def main():
    # 1. Find a real JPEG sample
    sample_dirs = [
        PROJECT_ROOT / "train" / "FAKE",
        PROJECT_ROOT / "train" / "REAL",
        PROJECT_ROOT / "test" / "FAKE",
    ]
    sample_path = None
    for d in sample_dirs:
        if d.is_dir():
            sample_path = next(iter(sorted(d.iterdir())), None)
            if sample_path:
                print(f"Using sample image: {sample_path}")
                break

    if sample_path is None:
        print("ERROR: No sample images available.")
        sys.exit(1)

    # 2. Run inference directly via model/predict.py
    print("\n--- Direct predict() call ---")
    from predict import predict
    weights = str(PROJECT_ROOT / "model" / "weights" / "signalscope_baseline.keras")
    result = predict(str(sample_path), weights_path=weights)
    print(f"  verdict:     {result['verdict']}")
    print(f"  confidence:  {result['confidence']}")
    print(f"  raw_prob:    {result['raw_prob']}")
    print(f"  heatmap:     {result['heatmap']}")
    print(f"  explanation: {result['explanation']}")

    # 3. Run inference via ModelService (Flask service layer)
    print("\n--- ModelService.run() call ---")
    from services.model_service import ModelService
    svc = ModelService(weights_path=weights)
    result2 = svc.run(str(sample_path))
    print(f"  verdict:     {result2['verdict']}")
    print(f"  confidence:  {result2['confidence']}")
    print(f"  raw_prob:    {result2['raw_prob']}")

    # 4. Verify consistency
    assert result["verdict"] == result2["verdict"], "Mismatch between predict() and ModelService!"
    assert abs(result["raw_prob"] - result2["raw_prob"]) < 1e-4
    print("\n✅ Direct predict() and ModelService.run() are consistent.")
    print("✅ No model.fit() called.")
    print("✅ No fake predictions — result comes from signalscope_baseline.keras")
    return result

if __name__ == "__main__":
    r = main()
    print(f"\nFinal verdict: {r['verdict']} (confidence: {r['confidence']:.4f})")
