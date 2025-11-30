"""
Logging utilities.

Provides centralized logging setup and convenience functions for getting loggers.
"""

import logging
from pathlib import Path
from typing import Optional

from config.logging import setup_logging as setup_logging_config


def setup_logging(config_path: Optional[Path] = None) -> None:
    """
    Set up logging configuration.
    
    This is a convenience wrapper around config.logging.setup_logging that
    provides the same interface as before but now uses JSON configuration.
    
    Args:
        config_path: Optional path to JSON configuration file. If not provided,
                     uses default config or environment variables.
    """
    setup_logging_config(config_path)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
    
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
