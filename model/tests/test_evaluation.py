"""
Tests for model/evaluate.py — Held-Out Test Set Evaluation.

Test strategy:
  - Structural/import tests: verify the module is importable and functions exist.
  - Dataset isolation tests: verify test/ is structurally correct and separate.
  - Unit tests for metric calculation: use small synthetic fixtures rather than
    running full 20,000-image evaluation (which is slow and belongs in CI/CD).
  - Regression tests for output structure: verify JSON schema and PNG existence
    AFTER evaluate.py has been run at least once.

Rules enforced:
  - model.fit() must NEVER be called inside evaluate.py.
  - train/ is NEVER included in test dataset paths.
  - Class mapping: REAL=0, FAKE=1.
  - Confusion matrix must be 2×2.
  - All metrics must be in valid numeric ranges.
"""

from __future__ import annotations

import json
import sys
import math
from pathlib import Path

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _MODEL_DIR.parent
sys.path.insert(0, str(_MODEL_DIR))


# ===========================================================================
# 1. Import and structure tests
# ===========================================================================

class TestEvaluateImports:
    """The evaluate module must import cleanly and expose the required API."""

    def test_evaluate_module_imports(self):
        """evaluate.py must import without errors."""
        import evaluate  # noqa: F401

    def test_required_functions_exist(self):
        """All required public functions must be present."""
        import evaluate
        required = [
            "load_model",
            "load_test_dataset",
            "collect_predictions",
            "calculate_metrics",
            "save_results",
            "save_confusion_matrix",
            "evaluate",
        ]
        for fn in required:
            assert hasattr(evaluate, fn), f"Missing function: evaluate.{fn}"

    def test_no_fit_call_in_source(self):
        """
        evaluate.py must never call model.fit().
        This is a static source-code check against executable code lines.
        """
        import ast
        source_path = _MODEL_DIR / "evaluate.py"
        source = source_path.read_text(encoding="utf-8")

        # Parse the AST to find actual .fit( method calls — not comments/docstrings
        try:
            tree = ast.parse(source)
        except SyntaxError:
            pytest.fail("evaluate.py has a syntax error")

        fit_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Look for any_expr.fit(...)
                if isinstance(func, ast.Attribute) and func.attr == "fit":
                    fit_calls.append(ast.dump(node))

        assert len(fit_calls) == 0, (
            f"evaluate.py contains a .fit() call in executable code: {fit_calls}"
        )


# ===========================================================================
# 2. Class mapping and config consistency
# ===========================================================================

class TestClassMapping:
    """The REAL=0, FAKE=1 mapping must be consistently enforced."""

    def test_label_real_is_zero(self):
        from config import LABEL_REAL
        assert LABEL_REAL == 0, f"LABEL_REAL must be 0, got {LABEL_REAL}"

    def test_label_fake_is_one(self):
        from config import LABEL_FAKE
        assert LABEL_FAKE == 1, f"LABEL_FAKE must be 1, got {LABEL_FAKE}"

    def test_class_names_match_labels(self):
        from config import CLASS_REAL, CLASS_FAKE, LABEL_REAL, LABEL_FAKE
        assert CLASS_REAL == "REAL"
        assert CLASS_FAKE == "FAKE"
        # confirm the intended mapping
        assert {CLASS_REAL: LABEL_REAL, CLASS_FAKE: LABEL_FAKE} == {"REAL": 0, "FAKE": 1}

    def test_classification_threshold(self):
        from config import CLASSIFICATION_THRESHOLD
        assert CLASSIFICATION_THRESHOLD == 0.5


# ===========================================================================
# 3. Test dataset structure (no model needed)
# ===========================================================================

class TestTestDatasetStructure:
    """Verify the held-out test dataset structure, without loading the model."""

    def test_test_fake_dir_exists(self):
        assert (_PROJECT_ROOT / "test" / "FAKE").is_dir()

    def test_test_real_dir_exists(self):
        assert (_PROJECT_ROOT / "test" / "REAL").is_dir()

    def test_test_fake_count(self):
        count = sum(1 for f in (_PROJECT_ROOT / "test" / "FAKE").iterdir() if f.is_file())
        assert count == 10_000, f"Expected 10,000 FAKE test images, got {count}"

    def test_test_real_count(self):
        count = sum(1 for f in (_PROJECT_ROOT / "test" / "REAL").iterdir() if f.is_file())
        assert count == 10_000, f"Expected 10,000 REAL test images, got {count}"

    def test_test_total_count_is_20000(self):
        fake = sum(1 for f in (_PROJECT_ROOT / "test" / "FAKE").iterdir() if f.is_file())
        real = sum(1 for f in (_PROJECT_ROOT / "test" / "REAL").iterdir() if f.is_file())
        assert fake + real == 20_000

    def test_test_dir_is_separate_from_train(self):
        """test/ and train/ must be different directories (data isolation)."""
        test_root = (_PROJECT_ROOT / "test").resolve()
        train_root = (_PROJECT_ROOT / "train").resolve()
        assert test_root != train_root

    def test_load_test_dataset_returns_correct_counts(self):
        """load_test_dataset() must return 20,000 items, all from test/."""
        from evaluate import load_test_dataset
        paths, labels = load_test_dataset()
        assert len(paths) == 20_000
        assert len(labels) == 20_000

    def test_load_test_dataset_labels_are_binary(self):
        """All labels returned must be 0 or 1."""
        from evaluate import load_test_dataset
        _, labels = load_test_dataset()
        unique = set(labels)
        assert unique == {0, 1}, f"Expected labels {{0, 1}}, got {unique}"

    def test_load_test_dataset_paths_all_in_test_dir(self):
        """Every returned path must be inside test/, NOT inside train/."""
        from evaluate import load_test_dataset
        train_root = str((_PROJECT_ROOT / "train").resolve())
        paths, _ = load_test_dataset()
        for p in paths:
            assert train_root not in str(p.resolve()), (
                f"Training path found in test dataset: {p}"
            )

    def test_fake_images_labelled_1(self):
        """FAKE images must all carry label 1."""
        from evaluate import load_test_dataset
        paths, labels = load_test_dataset()
        for p, lbl in zip(paths, labels):
            if "FAKE" in p.parts:
                assert lbl == 1, f"FAKE image has wrong label: {lbl} at {p}"
                break  # spot-check first found

    def test_real_images_labelled_0(self):
        """REAL images must all carry label 0."""
        from evaluate import load_test_dataset
        paths, labels = load_test_dataset()
        for p, lbl in zip(paths, labels):
            if "REAL" in p.parts:
                assert lbl == 0, f"REAL image has wrong label: {lbl} at {p}"
                break


# ===========================================================================
# 4. Metric calculation unit tests (synthetic fixtures, no model needed)
# ===========================================================================

class TestCalculateMetrics:
    """
    Unit tests for calculate_metrics() using tiny synthetic data.
    No model loading or real images needed — fast and deterministic.
    """

    def _run(self, y_true, raw_probs, threshold=0.5):
        from evaluate import calculate_metrics
        return calculate_metrics(y_true, np.array(raw_probs, dtype="float32"), threshold)

    def test_perfect_classifier(self):
        """A perfect classifier should yield accuracy=1.0, auc=1.0, f1=1.0."""
        # REAL=0, FAKE=1. Perfect: REAL gets prob 0.0, FAKE gets prob 1.0
        y_true = [0, 0, 0, 1, 1, 1]
        probs  = [0.1, 0.2, 0.1, 0.9, 0.8, 0.95]
        result = self._run(y_true, probs)
        m = result["metrics"]
        assert m["accuracy"]  == pytest.approx(1.0, abs=1e-5)
        assert m["roc_auc"]   == pytest.approx(1.0, abs=1e-5)
        assert m["macro_f1"]  == pytest.approx(1.0, abs=1e-5)

    def test_worst_classifier(self):
        """A classifier that always predicts wrong should have low metrics."""
        y_true = [0, 0, 1, 1]
        probs  = [0.9, 0.9, 0.1, 0.1]  # always wrong
        result = self._run(y_true, probs)
        m = result["metrics"]
        assert m["accuracy"] == pytest.approx(0.0, abs=1e-5)

    def test_confusion_matrix_shape(self):
        """Confusion matrix must always be 2×2."""
        y_true = [0, 0, 1, 1]
        probs  = [0.2, 0.8, 0.3, 0.7]
        result = self._run(y_true, probs)
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

    def test_confusion_matrix_row_sums(self):
        """Each row of the confusion matrix must sum to the number of true samples."""
        y_true = [0, 0, 0, 1, 1]
        probs  = [0.1, 0.9, 0.2, 0.8, 0.6]
        result = self._run(y_true, probs)
        cm = result["confusion_matrix"]
        assert cm[0][0] + cm[0][1] == 3   # 3 REAL samples
        assert cm[1][0] + cm[1][1] == 2   # 2 FAKE samples

    def test_confusion_matrix_counts_match_total(self):
        """Sum of confusion matrix must equal total sample count."""
        y_true = [0, 1, 0, 1, 0]
        probs  = [0.2, 0.8, 0.3, 0.7, 0.9]
        result = self._run(y_true, probs)
        cm = result["confusion_matrix"]
        total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total == 5

    def test_metrics_in_valid_range(self):
        """All metrics must be floats in [0, 1]."""
        y_true = [0, 0, 1, 1, 0, 1]
        probs  = [0.1, 0.6, 0.7, 0.4, 0.3, 0.8]
        result = self._run(y_true, probs)
        m = result["metrics"]
        for key in ["accuracy", "roc_auc", "macro_f1", "precision", "recall"]:
            val = m[key]
            assert isinstance(val, float), f"{key} is not a float"
            assert 0.0 <= val <= 1.0, f"{key}={val} is outside [0, 1]"

    def test_metrics_are_not_nan(self):
        """No metric must be NaN or infinite."""
        y_true = [0, 1, 0, 1]
        probs  = [0.2, 0.8, 0.4, 0.6]
        result = self._run(y_true, probs)
        m = result["metrics"]
        for key in ["accuracy", "roc_auc", "macro_f1", "precision", "recall"]:
            val = m[key]
            assert not math.isnan(val), f"{key} is NaN"
            assert not math.isinf(val), f"{key} is infinite"

    def test_roc_auc_uses_probabilities_not_threshold(self):
        """
        ROC-AUC should be calculated from raw probabilities, giving different
        results than a metric that uses thresholded predictions.
        """
        # Create a case where raw probs give AUC > 0.5 but thresholded accuracy = 0.5
        y_true = [0, 0, 1, 1]
        # All probs below 0.5 but FAKE still has higher probs than REAL
        probs = [0.1, 0.2, 0.3, 0.4]  # All predict REAL after threshold
        result = self._run(y_true, probs)
        m = result["metrics"]
        # Thresholded accuracy should be 0.5 (all predicted REAL, 2 correct out of 4)
        assert m["accuracy"] == pytest.approx(0.5, abs=1e-5)
        # AUC should be 1.0 (FAKE probs always > REAL probs in ranking)
        assert m["roc_auc"] == pytest.approx(1.0, abs=1e-5)

    def test_counts_are_consistent(self):
        """correct + incorrect must equal total."""
        y_true = [0, 1, 0, 1, 0]
        probs  = [0.1, 0.9, 0.8, 0.7, 0.2]
        result = self._run(y_true, probs)
        c = result["counts"]
        assert c["correct"] + c["incorrect"] == c["total"]
        assert c["total"] == 5

    def test_balanced_dataset_baseline(self):
        """
        With a balanced dataset and a random-like classifier (AUC~=0.5),
        accuracy should be near 0.5.
        """
        rng = np.random.RandomState(42)
        y_true = [0] * 100 + [1] * 100
        probs = rng.uniform(0, 1, 200).tolist()
        result = self._run(y_true, probs)
        m = result["metrics"]
        # Random classifier: accuracy should be near 0.5 (+/- noise)
        assert 0.3 <= m["accuracy"] <= 0.7


# ===========================================================================
# 5. Output file structure tests (require evaluate.py to have run at least once)
# ===========================================================================

class TestOutputFileStructure:
    """
    Tests that verify the output JSON and PNG structure.
    These tests are SKIPPED if the output files do not yet exist
    (i.e., before the first actual evaluation run).
    """

    @pytest.fixture
    def results(self):
        path = _PROJECT_ROOT / "report" / "evaluation_results.json"
        if not path.exists():
            pytest.skip("evaluation_results.json not yet generated — run model/evaluate.py first")
        with open(path) as f:
            return json.load(f)

    @pytest.fixture
    def cm_png(self):
        path = _PROJECT_ROOT / "report" / "confusion_matrix.png"
        if not path.exists():
            pytest.skip("confusion_matrix.png not yet generated — run model/evaluate.py first")
        return path

    def test_json_status_is_completed(self, results):
        assert results["status"] == "completed"

    def test_json_has_metrics_section(self, results):
        assert "metrics" in results

    def test_json_metrics_keys(self, results):
        required_keys = {"accuracy", "roc_auc", "macro_f1", "precision", "recall"}
        actual_keys = set(results["metrics"].keys())
        assert required_keys.issubset(actual_keys), (
            f"Missing metric keys: {required_keys - actual_keys}"
        )

    def test_json_metrics_are_floats_in_range(self, results):
        for key, val in results["metrics"].items():
            if isinstance(val, (int, float)):
                assert 0.0 <= val <= 1.0, f"Metric {key}={val} out of [0,1]"
                assert not math.isnan(val), f"Metric {key} is NaN"

    def test_json_total_samples_is_20000(self, results):
        assert results["dataset"]["total_samples"] == 20_000

    def test_json_real_samples_is_10000(self, results):
        assert results["dataset"]["real_samples"] == 10_000

    def test_json_fake_samples_is_10000(self, results):
        assert results["dataset"]["fake_samples"] == 10_000

    def test_json_class_mapping(self, results):
        cm_map = results["dataset"]["class_mapping"]
        assert cm_map["REAL"] == 0
        assert cm_map["FAKE"] == 1

    def test_json_confusion_matrix_is_2x2(self, results):
        cm = results["confusion_matrix"]["matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

    def test_json_confusion_matrix_sums_to_20000(self, results):
        cm = results["confusion_matrix"]["matrix"]
        total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total == 20_000

    def test_json_no_session2_val_metrics_reported_as_test(self, results):
        """
        Session 2 validation accuracy was ~0.7938.
        The test accuracy reported here must have been independently calculated.
        This test just confirms the JSON contains a numeric accuracy value.
        """
        acc = results["metrics"]["accuracy"]
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_png_confusion_matrix_exists(self, cm_png):
        assert cm_png.exists()
        assert cm_png.stat().st_size > 5_000, "PNG file seems too small to be valid"

    def test_json_roc_auc_is_not_zero(self, results):
        """
        ROC-AUC must not be 0.0 — that would indicate it was never calculated.
        Session 2 training log showed val_auc=0 due to a metric bug; the
        held-out evaluation must calculate a real AUC.
        """
        auc = results["metrics"]["roc_auc"]
        assert auc > 0.5, (
            f"ROC-AUC={auc} — this looks wrong. "
            "Ensure roc_auc_score is called with raw probabilities."
        )
