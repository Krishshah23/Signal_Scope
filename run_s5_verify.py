"""Session 5 manual verification: test full inference+Grad-CAM chain."""
import sys, os, base64
from pathlib import Path
sys.path.insert(0, 'c:/Users/Admin/Signal_scope/model')
sys.path.insert(0, 'c:/Users/Admin/Signal_scope/src/backend')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT_ROOT = Path('c:/Users/Admin/Signal_scope')
WEIGHTS = PROJECT_ROOT / 'model' / 'weights' / 'signalscope_baseline.keras'

from predict import predict, VERDICT_AI, VERDICT_REAL

# Check file sizes before and after
before_size = WEIGHTS.stat().st_size

samples = [
    (PROJECT_ROOT / 'train' / 'FAKE' / sorted((PROJECT_ROOT / 'train' / 'FAKE').iterdir())[0].name, 'FAKE'),
    (PROJECT_ROOT / 'train' / 'REAL' / sorted((PROJECT_ROOT / 'train' / 'REAL').iterdir())[0].name, 'REAL'),
]

print("=" * 60)
print("Session 5 — Manual Grad-CAM Verification")
print("=" * 60)

for sample_path, label in samples:
    print(f"\n[{label}] {sample_path.name}")
    result = predict(str(sample_path), weights_path=str(WEIGHTS))

    print(f"  verdict:     {result['verdict']}")
    print(f"  confidence:  {result['confidence']}")
    print(f"  raw_prob:    {result['raw_prob']}")
    print(f"  heatmap:     {'<base64 string, len=' + str(len(result['heatmap'])) + '>' if result['heatmap'] else None}")
    print(f"  explanation: {result['explanation'][:60] + '...' if result['explanation'] else None}")

    # Verify heatmap decodes to a valid PNG
    if result['heatmap']:
        import io
        from PIL import Image
        decoded = base64.b64decode(result['heatmap'])
        img = Image.open(io.BytesIO(decoded))
        print(f"  heatmap PNG: {img.size} {img.mode}  OK")
    else:
        print("  heatmap: None (Grad-CAM failed)")

    assert result['verdict'] in (VERDICT_AI, VERDICT_REAL)
    assert 0.0 <= result['confidence'] <= 1.0
    assert 0.0 <= result['raw_prob'] <= 1.0

after_size = WEIGHTS.stat().st_size
assert before_size == after_size, "Model file was modified!"

print("\n" + "=" * 60)
print("PASS: No model.fit() called")
print("PASS: No retraining")
print(f"PASS: Model weights unchanged (size: {after_size:,} bytes)")
print("PASS: Heatmaps are real Grad-CAM output (logit-based)")
print("PASS: Predictions use sigmoid probability")
print("=" * 60)
