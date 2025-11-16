"""
Configuration management for LLM models.

This module provides utilities to load and manage model configurations.
"""

from pathlib import Path
from typing import Dict, Any
import importlib.util
import sys

# Project root directory
# __file__ is at src/llm/config/__init__.py
# Go up 3 levels: config -> llm -> src -> project_root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a configuration dictionary from a config file.
    
    Config files should be Python modules in the configs/ directory.
    The config_name should match the filename (without .py extension).
    The config dictionary should be named with the pattern: {CONFIG_NAME}_CONFIG
    or just be the only dictionary in the file.
    
    Args:
        config_name: Name of the config file (without .py extension).
                    Example: "gpt_config_124m" for configs/gpt_config_124m.py
    
    Returns:
        Dictionary containing the model configuration.
    
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config dictionary cannot be found in the file.
    
    Example:
        >>> config = load_config("gpt_config_124m")
        >>> print(config["vocab_size"])
        50257
    """
    config_path = CONFIGS_DIR / f"{config_name}.py"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available configs in {CONFIGS_DIR}: {list_available_configs()}"
        )
    
    # Load the config module
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config file: {config_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = module
    spec.loader.exec_module(module)
    
    # Try to find the config dictionary
    # First, try the standard naming convention: {CONFIG_NAME}_CONFIG
    expected_name = config_name.upper().replace("-", "_") + "_CONFIG"
    if hasattr(module, expected_name):
        return getattr(module, expected_name)
    
    # Try GPT_CONFIG_124M style (uppercase with numbers)
    for attr_name in dir(module):
        if attr_name.startswith("GPT_CONFIG") or attr_name.endswith("_CONFIG"):
            if isinstance(getattr(module, attr_name), dict):
                return getattr(module, attr_name)
    
    # Try any dictionary that looks like a config
    for attr_name in dir(module):
        if not attr_name.startswith("_"):
            attr = getattr(module, attr_name)
            if isinstance(attr, dict) and len(attr) > 0:
                # Check if it looks like a config (has common keys)
                if any(key in attr for key in ["vocab_size", "emb_dim", "n_layers", "context_length"]):
                    return attr
    
    raise ValueError(
        f"Could not find configuration dictionary in {config_path}.\n"
        f"Expected a dictionary named '{expected_name}' or similar."
    )


def list_available_configs() -> list[str]:
    """
    List all available configuration files in the configs directory.
    
    Returns:
        List of config names (without .py extension).
    """
    if not CONFIGS_DIR.exists():
        return []
    
    configs = []
    for file in CONFIGS_DIR.glob("*.py"):
        if file.name != "__init__.py":
            configs.append(file.stem)
    
    return sorted(configs)


def get_config_path(config_name: str) -> Path:
    """
    Get the full path to a config file.
    
    Args:
        config_name: Name of the config file (without .py extension).
    
    Returns:
        Path object pointing to the config file.
    """
    return CONFIGS_DIR / f"{config_name}.py"

