"""Final test runner for Sessions 6-8 verification.
Runs model tests and backend tests separately to avoid sys.path collision
between model/config.py and src/backend/config.py.
"""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run(label, args):
    r = subprocess.run([sys.executable, "-m", "pytest"] + args +
                       ["-q", "--tb=short", "-p", "no:cacheprovider",
                        "--ignore=model/tests/test_results.txt"],
                       capture_output=True, text=True)
    lines = r.stdout.strip().split("\n")
    # Print last summary line
    summary = [l for l in lines if "passed" in l or "failed" in l or "error" in l]
    print(f"[{label}]", summary[-1] if summary else "(no summary)")
    if r.returncode != 0:
        # Print last 20 lines of failures
        print("\n".join(lines[-20:]))
    return r.returncode

rc1 = run("MODEL TESTS", ["model/tests/"])
rc2 = run("BACKEND TESTS", ["src/backend/tests/"])

total = 0 if (rc1 == 0 and rc2 == 0) else 1
print(f"\nOverall: {'PASS' if total == 0 else 'FAIL'} (rc={total})")
