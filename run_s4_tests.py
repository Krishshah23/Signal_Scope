"""Session 4 test runner — runs model/tests and src/backend/tests."""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "model/tests/test_model_inference.py",
     "src/backend/tests/",
     "-v", "--tb=short", "-p", "no:cacheprovider"],
    capture_output=True, text=True,
)
print("STDOUT:\n", result.stdout[-8000:])
if result.returncode != 0:
    print("STDERR (last 2000):\n", result.stderr[-2000:])
print("Return code:", result.returncode)
