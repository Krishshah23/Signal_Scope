"""
pytest configuration for the SignalScope backend.

Adds src/backend/ to sys.path so that test files can import
`app`, `config`, `routes.*`, and `services.*` without needing
relative imports or package installation.
"""

import sys
import os

# Insert the backend root (src/backend/) at the front of the module search path.
# This file is discovered automatically by pytest when running from any directory.
sys.path.insert(0, os.path.dirname(__file__))
