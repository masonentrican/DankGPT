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
    print("\n" + "=" * 70)
    print("TEST 1: Basic Logging - All Levels")
    print("=" * 70)
    
    setup_logging()
    logger = get_logger(__name__)
    
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    print("\n✓ Basic logging test completed")
    print(f"  Check log file at: {LOGS_DIR / 'app.log'}")


def test_logger_hierarchy():
    """Test logger hierarchy with different module loggers."""
    print("\n" + "=" * 70)
    print("TEST 2: Logger Hierarchy")
    print("=" * 70)
    
    setup_logging()
    
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
    
    print("\n✓ Logger hierarchy test completed")


def test_custom_logger():
    """Test creating a custom logger for a specific module."""
    print("\n" + "=" * 70)
    print("TEST 3: Custom Logger")
    print("=" * 70)
    
    setup_logging()
    logger = get_logger("test_module")
    
    logger.debug("Debug message from test_module")
    logger.info("Info message from test_module")
    logger.warning("Warning message from test_module")
    
    print("\n✓ Custom logger test completed")


def test_logging_with_context():
    """Test logging with additional context information."""
    print("\n" + "=" * 70)
    print("TEST 4: Logging with Context")
    print("=" * 70)
    
    setup_logging()
    logger = get_logger(__name__)
    
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
    
    print("\n✓ Context logging test completed")


def test_environment_override():
    """Test environment variable overrides."""
    print("\n" + "=" * 70)
    print("TEST 5: Environment Variable Override")
    print("=" * 70)
    
    # Save original env vars
    original_console = os.environ.get("CONSOLE_LEVEL")
    original_file = os.environ.get("FILE_LEVEL")
    
    try:
        # Set environment variables
        os.environ["CONSOLE_LEVEL"] = "WARNING"
        os.environ["FILE_LEVEL"] = "DEBUG"
        
        # Re-setup logging to pick up env vars
        setup_logging()
        logger = get_logger(__name__)
        
        print("\n  Console level set to WARNING (should only see WARNING and above)")
        print("  File level set to DEBUG (all messages should be in file)")
        
        logger.debug("This DEBUG message should NOT appear in console")
        logger.info("This INFO message should NOT appear in console")
        logger.warning("This WARNING message SHOULD appear in console")
        logger.error("This ERROR message SHOULD appear in console")
        
        print("\n✓ Environment override test completed")
        print("  Check log file - it should contain all messages including DEBUG")
        
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
    print("\n" + "=" * 70)
    print("TEST 6: Custom Config Path")
    print("=" * 70)
    
    from config.paths import PROJECT_ROOT
    default_config_path = PROJECT_ROOT / "src" / "config" / "logging_config.json"
    
    if default_config_path.exists():
        setup_logging(default_config_path)
        logger = get_logger(__name__)
        logger.info("Loaded logging config from custom path")
        print("\n✓ Custom config path test completed")
    else:
        print(f"\n✗ Config file not found at: {default_config_path}")


def main():
    """Run all logging tests."""
    print("\n" + "=" * 70)
    print("LOGGING SYSTEM TEST SUITE")
    print("=" * 70)
    
    try:
        test_basic_logging()
        test_logger_hierarchy()
        test_custom_logger()
        test_logging_with_context()
        test_environment_override()
        test_custom_config_path()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        print(f"\nLog files are located in: {LOGS_DIR}")
        print("  - app.log (standard log file)")
        print("  - app_detailed.log (rotating detailed log file)")
        print("\nCheck the log files to verify all messages were written correctly.")
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

