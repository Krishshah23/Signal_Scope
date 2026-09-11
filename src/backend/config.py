"""
SignalScope Backend Configuration
----------------------------------
Centralised configuration for the Flask application.
All environment-sensitive values are read from environment variables
with safe defaults for local development.

Never commit real secrets — use .env (gitignored) or the deployment environment.
"""

import os


class Config:
    """Base configuration shared across all environments."""

    # -----------------------------------------------------------------
    # Flask core
    # -----------------------------------------------------------------
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
    DEBUG: bool = False
    TESTING: bool = False

    # -----------------------------------------------------------------
    # Upload handling
    # -----------------------------------------------------------------
    # Maximum upload size: 16 MB
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024

    # Permitted MIME types for uploaded images
    ALLOWED_MIME_TYPES: frozenset = frozenset(
        [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp",
        ]
    )

    # Permitted file extensions (lowercase, with dot)
    ALLOWED_EXTENSIONS: frozenset = frozenset([".jpg", ".jpeg", ".png", ".webp", ".bmp"])

    # -----------------------------------------------------------------
    # API
    # -----------------------------------------------------------------
    API_VERSION: str = "v1"
    API_PREFIX: str = f"/api/{API_VERSION}"

    # -----------------------------------------------------------------
    # CORS — origins allowed to call the API during development.
    # In production, lock this down to the actual frontend origin.
    # -----------------------------------------------------------------
    CORS_ORIGINS: list = os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")

    # -----------------------------------------------------------------
    # Model (populated in a future session once weights are trained)
    # -----------------------------------------------------------------
    MODEL_WEIGHTS_PATH: str = os.environ.get(
        "MODEL_WEIGHTS_PATH", "model/weights/signalscope.h5"
    )


class DevelopmentConfig(Config):
    """Development-specific overrides."""

    DEBUG: bool = True


class TestingConfig(Config):
    """Testing-specific overrides."""

    TESTING: bool = True
    DEBUG: bool = True
    # Disable CSRF / upload limits that could interfere with tests
    MAX_CONTENT_LENGTH: int = 1 * 1024 * 1024


class ProductionConfig(Config):
    """Production overrides — debug must be off."""

    DEBUG: bool = False


# Registry so the factory can resolve by name
_CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}

DEFAULT_CONFIG = "development"


def get_config(env: str | None = None) -> type[Config]:
    """
    Return the appropriate config class for the given environment name.

    Parameters
    ----------
    env : str, optional
        One of 'development', 'testing', 'production'.
        Falls back to the FLASK_ENV environment variable, then to
        'development'.

    Returns
    -------
    type[Config]
        The resolved Config subclass (not an instance).
    """
    resolved = env or os.environ.get("FLASK_ENV", DEFAULT_CONFIG)
    return _CONFIG_MAP.get(resolved, DevelopmentConfig)
