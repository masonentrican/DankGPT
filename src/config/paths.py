"""
Centralized path configuration for the project.

This module provides a single source of truth for project root and common paths.
Use this instead of manually calculating project root with Path(__file__).resolve().parents[N].

Example:
    from config.paths import PROJECT_ROOT
    
    model_dir = PROJECT_ROOT / "models"
    data_dir = PROJECT_ROOT / "data"
"""
from pathlib import Path


def _find_project_root(start_path: Path) -> Path:
    """
    Walk up the directory tree from start_path to find the project root
    (directory containing pyproject.toml).
    
    This works regardless of how deep the file is in the project structure.
    """
    current = start_path.resolve()
    
    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    
    # If we reach the filesystem root without finding pyproject.toml
    raise RuntimeError(
        f"Could not find project root. Searched from {start_path} up to {current.anchor}"
    )


# Get project root by finding the directory containing pyproject.toml
# This works regardless of where this file is located in the project structure
PROJECT_ROOT = _find_project_root(Path(__file__))

# Common paths
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

