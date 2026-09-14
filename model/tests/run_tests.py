"""
Test runner for all model tests (Sessions 2 & 3).
Run from project root with:
    .venv/Scripts/python model/tests/run_tests.py
"""
import subprocess
import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "model/tests/test_dataset.py",
     "model/tests/test_model_architecture.py",
     "model/tests/test_evaluation.py",
     "-v", "--tb=short", "-p", "no:cacheprovider"],
    capture_output=True,
    text=True,
)
print("STDOUT:\n", result.stdout)
if result.returncode != 0 and result.stderr:
    print("STDERR (last 3000 chars):\n", result.stderr[-3000:])
print("Return code:", result.returncode)
