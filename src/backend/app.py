"""
SignalScope — Flask Application Factory
-----------------------------------------
Creates and configures the Flask application.

Usage
-----
Development:
    set FLASK_ENV=development
    flask --app src/backend/app:create_app run --port 5000

Or directly:
    python src/backend/app.py
"""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS

from config import get_config
from routes.health import health_bp
from routes.predict import predict_bp


def create_app(env: str | None = None) -> Flask:
    """
    Flask application factory.

    Parameters
    ----------
    env : str, optional
        Configuration environment name: 'development', 'testing', 'production'.
        Defaults to the FLASK_ENV environment variable or 'development'.

    Returns
    -------
    Flask
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    config_class = get_config(env)
    app.config.from_object(config_class)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.DEBUG if app.config["DEBUG"] else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    app.logger.setLevel(logging.DEBUG if app.config["DEBUG"] else logging.INFO)

    # ------------------------------------------------------------------
    # CORS
    # Allow the React dev server (localhost:3000) to call the API.
    # Restricted to /api/* paths only.
    # ------------------------------------------------------------------
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": app.config["CORS_ORIGINS"],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type"],
            }
        },
    )

    # ------------------------------------------------------------------
    # Register blueprints under the versioned API prefix
    # ------------------------------------------------------------------
    api_prefix = app.config["API_PREFIX"]  # e.g. /api/v1

    app.register_blueprint(health_bp, url_prefix=api_prefix)
    app.register_blueprint(predict_bp, url_prefix=api_prefix)

    # ------------------------------------------------------------------
    # Global error handlers
    # ------------------------------------------------------------------

    @app.errorhandler(400)
    def bad_request(exc):
        return jsonify({"error": "bad_request", "message": str(exc)}), 400

    @app.errorhandler(404)
    def not_found(exc):
        return jsonify({"error": "not_found", "message": "Endpoint not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(exc):
        return jsonify(
            {"error": "method_not_allowed", "message": str(exc)}
        ), 405

    @app.errorhandler(413)
    def request_entity_too_large(exc):
        return jsonify(
            {
                "error": "file_too_large",
                "message": "Uploaded file exceeds the 16 MB size limit.",
            }
        ), 413

    @app.errorhandler(500)
    def internal_server_error(exc):
        app.logger.exception("Unhandled server error")
        return jsonify(
            {"error": "server_error", "message": "An unexpected error occurred."}
        ), 500

    app.logger.info(
        "SignalScope API ready — prefix: %s  env: %s",
        api_prefix,
        env or os.environ.get("FLASK_ENV", "development"),
    )

    return app


# ---------------------------------------------------------------------------
# Direct execution entry-point (development only)
# ---------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
