#!/usr/bin/env python3
"""
Script to run database error handling tests.
"""

import unittest
import sys
from test_database import TestDatabaseErrorHandling

if __name__ == "__main__":
    print("Starting database error handling tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDatabaseErrorHandling)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    if result.wasSuccessful():
        print("All database error handling tests passed!")
        sys.exit(0)
    else:
        print(f"Tests failed: {len(result.errors)} errors, {len(result.failures)} failures")
        sys.exit(1) 