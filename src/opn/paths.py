"""Repository folders, resolved from this file (use the repo checkout, or `pip install -e .`)."""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
NETWORK_DIR = os.path.join(REPO_ROOT, "networks", "jax")
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
