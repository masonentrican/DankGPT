"""
Logging configuration module.

JSON-based logging configuration with environment variable override support.
Uses Python's logging.config.dictConfig format.
"""

import json
import os
import logging
import logging.config
import atexit
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from config.paths import PROJECT_ROOT, LOGS_DIR

# Module-level variable to track if listener has been started
_queue_listener_started = False


class DateTimeFormatter(logging.Formatter):
    """Formatter that outputs timestamps in ISO 8601 format with milliseconds and timezone."""
    
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created).astimezone()
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}" + dt.strftime("%z")


def _load_json_config(config_path: Path) -> Dict[str, Any]:
    """
    Load logging configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file.
    
    Returns:
        Dictionary containing logging configuration.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config file is invalid JSON.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Logging config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to logging configuration.
    
    Environment variables:
        LOG_LEVEL: Sets both console and file level (if individual levels not set)
        CONSOLE_LEVEL: Sets console handler log level
        FILE_LEVEL: Sets file handler log level
        LOG_FILE: Sets log file path (relative to project root)
        LOG_CONFIG: Path to alternative JSON config file
    
    Args:
        config: Logging configuration dictionary.
    
    Returns:
        Modified configuration dictionary with env var overrides applied.
    """
    config = config.copy()
    
    # Check for individual level settings
    console_level = os.getenv("CONSOLE_LEVEL")
    file_level = os.getenv("FILE_LEVEL")
    
    # If individual levels not set, check for general log level
    if not console_level and not file_level:
        general_level = os.getenv("LOG_LEVEL")
        if general_level:
            console_level = general_level.upper()
            file_level = general_level.upper()
    
    # Apply console level override
    if console_level and "handlers" in config and "console" in config["handlers"]:
        config["handlers"]["console"]["level"] = console_level.upper()
    
    # Apply file level override
    if file_level and "handlers" in config and "file" in config["handlers"]:
        config["handlers"]["file"]["level"] = file_level.upper()
    
    # Apply log file path override (applies to primary "file" handler)
    log_file = os.getenv("LOG_FILE")
    if log_file and "handlers" in config and "file" in config["handlers"]:
        log_file_path = Path(log_file)
        # If relative path, make it relative to logs directory
        if not log_file_path.is_absolute():
            log_file_path = LOGS_DIR / log_file_path
        else:
            log_file_path = Path(log_file)
        config["handlers"]["file"]["filename"] = str(log_file_path)
    
    return config


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve relative paths in configuration to absolute paths.
    
    Args:
        config: Logging configuration dictionary.
    
    Returns:
        Configuration dictionary with resolved paths.
    """
    config = config.copy()
    
    # Resolve all file handler paths
    if "handlers" in config:
        for handler_name, handler_config in config["handlers"].items():
            # Check if this handler has a filename (FileHandler, RotatingFileHandler, etc.)
            if "filename" in handler_config:
                file_path = handler_config["filename"]
                if file_path:
                    file_path_obj = Path(file_path)
                    # If relative path, make it relative to logs directory
                    if not file_path_obj.is_absolute():
                        file_path_obj = LOGS_DIR / file_path_obj
                    else:
                        file_path_obj = Path(file_path)
                    config["handlers"][handler_name]["filename"] = str(file_path_obj)
    
    return config


def get_logging_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get logging configuration with environment variable overrides applied.
    
    Configuration priority:
    1. LOG_CONFIG environment variable (path to JSON file)
    2. Provided config_path parameter
    3. Default config file at src/config/logging_config.json
    
    Args:
        config_path: Optional path to JSON configuration file.
    
    Returns:
        Logging configuration dictionary ready for logging.config.dictConfig.
    """
    # Determine which config file to use
    env_config_path = os.getenv("LOG_CONFIG")
    if env_config_path:
        config_path = Path(env_config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
    elif config_path is None:
        # Default to config/logging_config.json
        config_path = PROJECT_ROOT / "src" / "config" / "logging_config.json"
    
    # Load JSON config
    config = _load_json_config(config_path)
    
    # Resolve relative paths
    config = _resolve_paths(config)
    
    # Apply environment variable overrides
    config = _apply_env_overrides(config)
    
    # Ensure logs directory exists for all file handlers
    if "handlers" in config:
        for handler_config in config["handlers"].values():
            if "filename" in handler_config:
                file_path = Path(handler_config["filename"])
                file_path.parent.mkdir(parents=True, exist_ok=True)
    
    return config


def setup_logging(config_path: Optional[Path] = None) -> None:
    """
    Set up logging configuration from JSON file with environment variable support.
    
    This function loads the JSON configuration, applies environment variable
    overrides, and configures Python's logging system using dictConfig.
    
    If a queue handler is configured, this function will start the QueueListener
    thread to process log records asynchronously.
    
    The function is idempotent -
    calling it multiple times is safe and will only start the listener once.
    
    Args:
        config_path: Optional path to JSON configuration file. If not provided,
                     uses default or LOG_CONFIG environment variable.
    """
    global _queue_listener_started
    
    config = get_logging_config(config_path)
    
    # Apply the configuration
    logging.config.dictConfig(config)
    
    # If queue handler is configured, start its listener
    # Try getHandlerByName first (Python 3.13+), fallback to searching root logger
    queue_handler = None
    try:
        queue_handler = logging.getHandlerByName("queue_handler")
    except AttributeError:
        # Fallback: search root logger's handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.handlers.QueueHandler):
                queue_handler = handler
                break
    
    if queue_handler is not None and not _queue_listener_started:
        if hasattr(queue_handler, 'listener') and queue_handler.listener is not None:
            queue_handler.listener.start()
            # Register cleanup on exit
            atexit.register(queue_handler.listener.stop)
            _queue_listener_started = True
