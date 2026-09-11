"""
Health-check route — GET /api/v1/health/
-----------------------------------------
Returns a simple JSON response confirming the API service is running.
This endpoint is intentionally lightweight: no database, no model load,
no external dependencies — it must always respond quickly.
"""

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/health/")
def health_check():
    """
    GET /api/v1/health/

    Returns
    -------
    200 OK
        {
            "status": "ok",
            "service": "SignalScope API",
            "version": "v1"
        }
    """
    return jsonify(
        {
            "status": "ok",
            "service": "SignalScope API",
            "version": "v1",
        }
    ), 200
