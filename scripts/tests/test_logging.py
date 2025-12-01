"""
Test script for logging configuration and functionality.

Tests various aspects of the logging system including:
- Basic logging setup
- Different log levels
- Console and file output
- Logger hierarchy
- Environment variable overrides
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.utils.logging import setup_logging, get_logger
from config.paths import LOGS_DIR


def test_basic_logging():
    """Test basic logging functionality with all log levels."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: Basic Logging - All Levels")
    logger.info("=" * 70)
    
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    logger.info("\n✓ Basic logging test completed")
    logger.info(f"  Check log file at: {LOGS_DIR / 'app.log'}")


def test_logger_hierarchy():
    """Test logger hierarchy with different module loggers."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Logger Hierarchy")
    logger.info("=" * 70)
    
    # Test different logger names
    root_logger = get_logger("")
    llm_logger = get_logger("llm")
    models_logger = get_logger("llm.models")
    training_logger = get_logger("llm.training")
    generation_logger = get_logger("llm.generation")
    
    root_logger.info("Root logger message")
    llm_logger.info("LLM logger message")
    models_logger.info("Models logger message")
    training_logger.info("Training logger message")
    generation_logger.info("Generation logger message")
    
    logger.info("\n✓ Logger hierarchy test completed")


def test_custom_logger():
    """Test creating a custom logger for a specific module."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Custom Logger")
    logger.info("=" * 70)
    
    custom_logger = get_logger("test_module")
    
    custom_logger.debug("Debug message from test_module")
    custom_logger.info("Info message from test_module")
    custom_logger.warning("Warning message from test_module")
    
    logger.info("\n✓ Custom logger test completed")


def test_logging_with_context():
    """Test logging with additional context information."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Logging with Context")
    logger.info("=" * 70)
    
    # Simulate some operations
    epoch = 1
    step = 100
    loss = 0.5234
    
    logger.info(f"Training progress - Epoch: {epoch}, Step: {step}, Loss: {loss:.4f}")
    logger.debug(f"Detailed debug info - Epoch: {epoch}, Step: {step}, Loss: {loss:.4f}")
    
    # Simulate error scenario
    try:
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
    
    logger.info("\n✓ Context logging test completed")


def test_environment_override():
    """Test environment variable overrides."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Environment Variable Override")
    logger.info("=" * 70)
    
    # Save original env vars
    original_console = os.environ.get("CONSOLE_LEVEL")
    original_file = os.environ.get("FILE_LEVEL")
    
    try:
        # Set environment variables
        os.environ["CONSOLE_LEVEL"] = "WARNING"
        os.environ["FILE_LEVEL"] = "DEBUG"
        
        # Re-setup logging to pick up env vars
        setup_logging()
        
        logger.info("\n  Console level set to WARNING (should only see WARNING and above)")
        logger.info("  File level set to DEBUG (all messages should be in file)")
        
        logger.debug("This DEBUG message should NOT appear in console")
        logger.info("This INFO message should NOT appear in console")
        logger.warning("This WARNING message SHOULD appear in console")
        logger.error("This ERROR message SHOULD appear in console")
        
        logger.info("\n✓ Environment override test completed")
        logger.info("  Check log file - it should contain all messages including DEBUG")
        
    finally:
        # Restore original env vars
        if original_console:
            os.environ["CONSOLE_LEVEL"] = original_console
        elif "CONSOLE_LEVEL" in os.environ:
            del os.environ["CONSOLE_LEVEL"]
            
        if original_file:
            os.environ["FILE_LEVEL"] = original_file
        elif "FILE_LEVEL" in os.environ:
            del os.environ["FILE_LEVEL"]


def test_custom_config_path():
    """Test loading logging config from a custom path."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST 6: Custom Config Path")
    
    from config.paths import PROJECT_ROOT
    default_config_path = PROJECT_ROOT / "src" / "config" / "logging_config.json"
    logger.info("=" * 70)
    
    if default_config_path.exists():
        setup_logging(default_config_path)
        logger.info("Loaded logging config from custom path")
        logger.info("\n✓ Custom config path test completed")
    else:
        logger.error(f"\n✗ Config file not found at: {default_config_path}")


def main():
    """Run all logging tests."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("\n" + "=" * 70)
    logger.info("LOGGING SYSTEM TEST SUITE")
    logger.info("=" * 70)
    
    try:
        test_basic_logging()
        test_logger_hierarchy()
        test_custom_logger()
        test_logging_with_context()
        test_environment_override()
        test_custom_config_path()
        
        logger.info("\n" + "=" * 70)
        logger.info("ALL TESTS COMPLETED")
        logger.info("=" * 70)
        logger.info(f"\nLog files are located in: {LOGS_DIR}")
        logger.info("  - app.log (standard log file)")
        logger.info("  - app_detailed.log (rotating detailed log file)")
        logger.info("\nCheck the log files to verify all messages were written correctly.")
        
    except Exception as e:
        logger.error(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

