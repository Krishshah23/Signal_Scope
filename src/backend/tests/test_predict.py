"""
Tests for POST /api/v1/predict/

Validates upload handling and the truthful 501 not-implemented response.
Does NOT test ML inference (model not yet trained).
"""

import io
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app


@pytest.fixture()
def client():
    app = create_app("testing")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_image(filename: str = "test.jpg", content_type: str = "image/jpeg") -> tuple:
    """Return a minimal fake image upload tuple for Flask test client."""
    data = io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 16)  # minimal JPEG-like bytes
    return (data, filename)


# ---------------------------------------------------------------------------
# Upload validation tests
# ---------------------------------------------------------------------------

def test_predict_no_file_returns_400(client):
    """Missing 'image' field must return 400."""
    resp = client.post("/api/v1/predict/")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "missing_file"


def test_predict_empty_filename_returns_400(client):
    """Empty filename must return 400."""
    resp = client.post(
        "/api/v1/predict/",
        data={"image": (io.BytesIO(b"data"), "")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "empty_filename"


def test_predict_unsupported_extension_returns_415(client):
    """A .txt file must be rejected with 415."""
    resp = client.post(
        "/api/v1/predict/",
        data={"image": (io.BytesIO(b"text data"), "file.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 415
    data = resp.get_json()
    assert data["error"] == "unsupported_file_type"


def test_predict_valid_jpeg_returns_501(client):
    """
    A valid JPEG upload must be accepted (upload pipeline works)
    but the endpoint must return 501 until the model is connected.
    """
    resp = client.post(
        "/api/v1/predict/",
        data={"image": _fake_image("photo.jpg", "image/jpeg")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 501
    data = resp.get_json()
    assert data["status"] == "not_implemented"


def test_predict_valid_png_returns_501(client):
    """PNG uploads must also be accepted and return 501."""
    resp = client.post(
        "/api/v1/predict/",
        data={"image": (io.BytesIO(b"\x89PNG\r\n" + b"\x00" * 16), "image.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 501


def test_predict_response_includes_filename(client):
    """The 501 response should echo the original filename."""
    resp = client.post(
        "/api/v1/predict/",
        data={"image": _fake_image("myimage.jpg")},
        content_type="multipart/form-data",
    )
    data = resp.get_json()
    assert data.get("filename") == "myimage.jpg"
