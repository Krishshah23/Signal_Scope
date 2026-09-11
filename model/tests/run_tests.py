"""
Test runner for Session 2 model tests.
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
     "-v", "--tb=short", "-p", "no:cacheprovider"],
    capture_output=True,
    text=True,
)
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
print("Return code:", result.returncode)
