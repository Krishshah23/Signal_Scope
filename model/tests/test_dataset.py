"""
Tests for model/dataset.py — CIFAKE dataset preparation.

Tests verify:
 - Dataset directory structure (train/FAKE, train/REAL, test/FAKE, test/REAL)
 - Class discovery
 - Image file counts
 - Data isolation (test dir is separate from train dir)
 - Label mapping correctness
 - Image loading and shape (128×128×3)
 - No test data passed to training pipeline

These tests do NOT train the model.
These tests do NOT compute model performance.
"""

import sys
import os
from pathlib import Path

# Ensure model/ is on sys.path
_MODEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MODEL_DIR))

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cfg():
    """Load config once for the whole test session."""
    import config
    return config


@pytest.fixture(scope="session")
def train_dir(cfg):
    return cfg.DATASET_TRAIN_DIR


@pytest.fixture(scope="session")
def test_dir(cfg):
    return cfg.DATASET_TEST_DIR


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_image_size(self, cfg):
        assert cfg.IMAGE_SIZE == (128, 128)

    def test_input_shape(self, cfg):
        assert cfg.INPUT_SHAPE == (128, 128, 3)

    def test_label_mapping(self, cfg):
        assert cfg.LABEL_FAKE == 1
        assert cfg.LABEL_REAL == 0

    def test_classification_threshold(self, cfg):
        assert cfg.CLASSIFICATION_THRESHOLD == 0.5

    def test_backbone_frozen(self, cfg):
        assert cfg.BACKBONE_TRAINABLE is False

    def test_backbone_name(self, cfg):
        assert cfg.BACKBONE == "MobileNetV2"

    def test_validation_split_range(self, cfg):
        assert 0.0 < cfg.VALIDATION_SPLIT < 1.0

    def test_random_seed_set(self, cfg):
        assert isinstance(cfg.RANDOM_SEED, int)


# ---------------------------------------------------------------------------
# Dataset structure tests
# ---------------------------------------------------------------------------

class TestDatasetStructure:
    def test_train_dir_exists(self, train_dir):
        """Training directory must exist."""
        assert train_dir.is_dir(), (
            f"Train directory not found: {train_dir}\n"
            "Download CIFAKE and place train/ at the project root."
        )

    def test_test_dir_exists(self, test_dir):
        """Test directory must exist."""
        assert test_dir.is_dir(), (
            f"Test directory not found: {test_dir}\n"
            "Download CIFAKE and place test/ at the project root."
        )

    def test_train_fake_class_exists(self, train_dir):
        assert (train_dir / "FAKE").is_dir()

    def test_train_real_class_exists(self, train_dir):
        assert (train_dir / "REAL").is_dir()

    def test_test_fake_class_exists(self, test_dir):
        assert (test_dir / "FAKE").is_dir()

    def test_test_real_class_exists(self, test_dir):
        assert (test_dir / "REAL").is_dir()

    def test_train_has_exactly_two_classes(self, train_dir):
        classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
        assert set(classes) == {"FAKE", "REAL"}, (
            f"Expected classes FAKE and REAL, found: {classes}"
        )

    def test_test_has_exactly_two_classes(self, test_dir):
        classes = [d.name for d in test_dir.iterdir() if d.is_dir()]
        assert set(classes) == {"FAKE", "REAL"}


# ---------------------------------------------------------------------------
# Image count tests
# ---------------------------------------------------------------------------

class TestImageCounts:
    def _count_images(self, directory: Path) -> int:
        return sum(
            1 for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def test_train_fake_has_images(self, train_dir):
        count = self._count_images(train_dir / "FAKE")
        assert count > 0, "train/FAKE/ contains no image files."

    def test_train_real_has_images(self, train_dir):
        count = self._count_images(train_dir / "REAL")
        assert count > 0, "train/REAL/ contains no image files."

    def test_test_fake_has_images(self, test_dir):
        count = self._count_images(test_dir / "FAKE")
        assert count > 0, "test/FAKE/ contains no image files."

    def test_test_real_has_images(self, test_dir):
        count = self._count_images(test_dir / "REAL")
        assert count > 0, "test/REAL/ contains no image files."

    def test_train_classes_are_balanced(self, train_dir):
        """CIFAKE training data is class-balanced (50k FAKE, 50k REAL)."""
        fake = self._count_images(train_dir / "FAKE")
        real = self._count_images(train_dir / "REAL")
        # Allow up to 5% imbalance
        ratio = abs(fake - real) / max(fake, real)
        assert ratio < 0.05, (
            f"Train classes are significantly imbalanced: FAKE={fake}, REAL={real}"
        )

    def test_test_classes_are_balanced(self, test_dir):
        """CIFAKE test data is class-balanced (10k FAKE, 10k REAL)."""
        fake = self._count_images(test_dir / "FAKE")
        real = self._count_images(test_dir / "REAL")
        ratio = abs(fake - real) / max(fake, real)
        assert ratio < 0.05

    def test_train_total_count(self, train_dir):
        """Expect 100,000 total training images (50k FAKE + 50k REAL)."""
        total = (
            self._count_images(train_dir / "FAKE") +
            self._count_images(train_dir / "REAL")
        )
        assert total == 100_000, (
            f"Expected 100,000 training images, found {total}"
        )

    def test_test_total_count(self, test_dir):
        """Expect 20,000 total test images (10k FAKE + 10k REAL)."""
        total = (
            self._count_images(test_dir / "FAKE") +
            self._count_images(test_dir / "REAL")
        )
        assert total == 20_000, (
            f"Expected 20,000 test images, found {total}"
        )


# ---------------------------------------------------------------------------
# Data isolation tests
# ---------------------------------------------------------------------------

class TestDataIsolation:
    def test_train_and_test_are_different_paths(self, train_dir, test_dir):
        """Train and test directories must be completely separate."""
        assert train_dir.resolve() != test_dir.resolve(), (
            "CRITICAL: train and test directories are identical!"
        )

    def test_test_not_inside_train(self, train_dir, test_dir):
        """Test directory must not be a subdirectory of train."""
        try:
            test_dir.resolve().relative_to(train_dir.resolve())
            pytest.fail("Test directory is inside training directory — data leakage risk!")
        except ValueError:
            pass  # Expected: test is NOT inside train

    def test_verify_no_test_in_train_function(self, train_dir, test_dir):
        """The dataset.verify_no_test_in_train() helper must return True."""
        from dataset import verify_no_test_in_train
        assert verify_no_test_in_train(train_dir, test_dir) is True


# ---------------------------------------------------------------------------
# Image loading and shape tests
# ---------------------------------------------------------------------------

class TestImageLoading:
    def test_sample_train_image_loads(self, train_dir):
        """Load one image from train/FAKE/ and verify shape."""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            pytest.skip("PIL/numpy not installed — skipping image load test")

        sample_files = list((train_dir / "FAKE").iterdir())
        assert len(sample_files) > 0

        img_path = sample_files[0]
        img = Image.open(img_path).convert("RGB").resize((128, 128))
        arr = np.array(img)

        assert arr.shape == (128, 128, 3), (
            f"Expected shape (128, 128, 3), got {arr.shape}"
        )
        assert arr.dtype == "uint8"
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_sample_train_real_image_loads(self, train_dir):
        """Load one image from train/REAL/ and verify shape."""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            pytest.skip("PIL/numpy not installed")

        sample_files = list((train_dir / "REAL").iterdir())
        assert len(sample_files) > 0

        img = Image.open(sample_files[0]).convert("RGB").resize((128, 128))
        arr = __import__("numpy").array(img)
        assert arr.shape == (128, 128, 3)

    def test_images_are_jpeg_format(self, train_dir):
        """Spot check: images should be JPEG files."""
        sample = list((train_dir / "FAKE").iterdir())[:5]
        for f in sample:
            assert f.suffix.lower() in {".jpg", ".jpeg"}, (
                f"Unexpected file extension: {f.suffix}"
            )


# ---------------------------------------------------------------------------
# inspect_dataset() function tests
# ---------------------------------------------------------------------------

class TestInspectDataset:
    def test_inspect_train(self, train_dir):
        from dataset import inspect_dataset
        info = inspect_dataset(train_dir, "train")
        assert info["exists"] is True
        assert set(info["classes"]) == {"FAKE", "REAL"}
        assert info["total"] == 100_000
        assert info["counts"]["FAKE"] == 50_000
        assert info["counts"]["REAL"] == 50_000

    def test_inspect_test(self, test_dir):
        from dataset import inspect_dataset
        info = inspect_dataset(test_dir, "test")
        assert info["exists"] is True
        assert set(info["classes"]) == {"FAKE", "REAL"}
        assert info["total"] == 20_000

    def test_inspect_nonexistent_returns_exists_false(self, tmp_path):
        from dataset import inspect_dataset
        info = inspect_dataset(tmp_path / "nonexistent_dir", "test")
        assert info["exists"] is False
        assert info["total"] == 0
