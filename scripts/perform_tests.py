"""
Script to execute all tests in the tests directory and ensure none produce errors.
"""

import subprocess
import sys
from pathlib import Path

from llm.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

def find_test_files(tests_dir: Path) -> list[Path]:
    """Find all test files in the tests directory."""
    test_files = []
    for file in tests_dir.iterdir():
        # Include only Python files that start with 'test_'
        if file.is_file() and file.name.startswith("test_") and file.suffix == ".py":
            test_files.append(file)
    return sorted(test_files)

def run_test(test_file: Path, tests_dir: Path) -> tuple[bool, str]:
    """
    Run a single test file and return (success, output).
    
    Args:
        test_file: Path to the test file to execute
        tests_dir: Path to the tests directory (for working directory)
    
    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        # Change to tests directory to ensure imports work correctly
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=str(tests_dir.parent),  # Parent directory so imports work
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout per test
        )
        
        output = result.stdout + result.stderr
        success = result.returncode == 0
        
        return success, output
    except subprocess.TimeoutExpired:
        return False, f"Test timed out after 60 seconds"
    except Exception as e:
        return False, f"Error running test: {str(e)}"

def main():
    """Main function to run all tests."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    tests_dir = script_dir / "tests"
    
    if not tests_dir.exists():
        logger.error(f"Error: Tests directory not found at {tests_dir}")
        sys.exit(1)
    
    # Find all test files
    test_files = find_test_files(tests_dir)
    
    if not test_files:
        logger.error(f"Error: No test files found in {tests_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(test_files)} test file(s):")
    for test_file in test_files:
        logger.info(f"- {test_file.name}")

    logger.info("Beginning tests...")
    logger.info("=" * 60)
    
    # Run each test
    passed = []
    failed = []
    
    for test_file in test_files:
        logger.info(f"Running {test_file.name}...")
        success, output = run_test(test_file, tests_dir)
        
        if success:
            logger.info("✓ PASSED")
            passed.append(test_file.name)
        else:
            logger.error("✗ FAILED")
            failed.append(test_file.name)
            logger.error("  Error output:")
            # Print first 20 lines of output to avoid cluttering
            output_lines = output.strip().split("\n")
            for line in output_lines[:20]:
                logger.error(f"    {line}")
            if len(output_lines) > 20:
                logger.error(f"    ... ({len(output_lines) - 20} more lines)")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary:")
    logger.info(f"  Total:  {len(test_files)}")
    logger.info(f"  Passed: {len(passed)}")
    logger.info(f"  Failed: {len(failed)}")
    logger.info("=" * 60)
    
    if failed:
        logger.error("\nFailed tests:")
        for test_name in failed:
            logger.error(f"  - {test_name}")
        sys.exit(1)
    else:
        logger.info("\nAll tests passed! ✓")
        sys.exit(0)

if __name__ == "__main__":
    main()

