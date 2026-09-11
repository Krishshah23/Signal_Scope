"""
Tests for GET /api/v1/health/
"""

import pytest

import sys
import os

# Ensure the backend package is on the path when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app


@pytest.fixture()
def client():
    app = create_app("testing")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_returns_200(client):
    resp = client.get("/api/v1/health/")
    assert resp.status_code == 200


def test_health_response_body(client):
    resp = client.get("/api/v1/health/")
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "SignalScope API"
    assert data["version"] == "v1"


def test_health_content_type(client):
    resp = client.get("/api/v1/health/")
    assert "application/json" in resp.content_type
