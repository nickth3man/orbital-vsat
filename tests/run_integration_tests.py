#!/usr/bin/env python3
"""
Run all integration tests for the VSAT application and generate a report.

This script runs the integration tests and generates a detailed report
of the results, including test coverage and performance metrics.
"""

import os
import sys
import unittest
import time
import logging
from datetime import datetime
from pathlib import Path

import coverage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def run_tests():
    """Run all integration tests and generate a report."""
    # Start time
    start_time = time.time()
    
    # Create reports directory if it doesn't exist
    reports_dir = Path("test_reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Set up coverage
    cov = coverage.Coverage(
        source=["src"],
        omit=[
            "*/tests/*",
            "*/venv/*",
            "*/__pycache__/*",
            "*/setup.py",
        ]
    )
    
    # Start coverage measurement
    cov.start()
    
    try:
        # Discover and run tests
        logger.info("Discovering integration tests...")
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover("tests", pattern="test_integration.py")
        
        # Run tests
        logger.info("Running integration tests...")
        test_runner = unittest.TextTestRunner(verbosity=2)
        test_result = test_runner.run(test_suite)
        
        # Stop coverage measurement
        cov.stop()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"integration_test_report_{timestamp}.txt"
        
        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("VSAT Integration Test Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Test Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total tests: {test_result.testsRun}\n")
            f.write(f"Passed: {test_result.testsRun - len(test_result.failures) - len(test_result.errors)}\n")
            f.write(f"Failed: {len(test_result.failures)}\n")
            f.write(f"Errors: {len(test_result.errors)}\n")
            f.write(f"Skipped: {len(test_result.skipped)}\n")
            f.write(f"Time elapsed: {elapsed_time:.2f} seconds\n\n")
            
            if test_result.failures:
                f.write("Failed Tests:\n")
                f.write("-" * 80 + "\n")
                for test, traceback in test_result.failures:
                    f.write(f"- {test}\n")
                    f.write(f"{traceback}\n\n")
            
            if test_result.errors:
                f.write("Test Errors:\n")
                f.write("-" * 80 + "\n")
                for test, traceback in test_result.errors:
                    f.write(f"- {test}\n")
                    f.write(f"{traceback}\n\n")
            
            # Write coverage report
            f.write("Coverage Report:\n")
            f.write("-" * 80 + "\n")
            
            # Save coverage report to file
            coverage_file = reports_dir / f"coverage_report_{timestamp}.txt"
            cov.report(file=open(coverage_file, "w"))
            
            # Include summary in main report
            with open(coverage_file, "r") as cov_file:
                f.write(cov_file.read())
            
            # Generate HTML coverage report
            html_dir = reports_dir / f"coverage_html_{timestamp}"
            cov.html_report(directory=str(html_dir))
            
            f.write("\nDetailed HTML coverage report generated at:\n")
            f.write(f"{html_dir.absolute()}\n")
        
        logger.info(f"Test report generated: {report_file}")
        logger.info(f"HTML coverage report: {html_dir}")
        
        # Return success if all tests passed
        return test_result.wasSuccessful()
    
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 