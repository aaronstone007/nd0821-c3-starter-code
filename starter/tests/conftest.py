"""Pytest configuration for the test suite.

Adds the inner ``starter`` package directory to ``sys.path`` so tests can import
the ML modules the same way ``train_model.py`` does (``from ml.model import ...``)
regardless of the current working directory.
"""

import os
import sys

# Directory containing this conftest: ``starter/tests``.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root that holds the inner package: ``starter``.
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
# Inner package directory that exposes the ``ml`` package: ``starter/starter``.
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "starter")

for _path in (PACKAGE_DIR, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)
