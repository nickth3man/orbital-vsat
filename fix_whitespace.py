#!/usr/bin/env python
"""
Fix whitespace issues in Python files.
- Removes trailing whitespace from all lines
- Ensures file ends with a newline
"""

import sys
import os


def fix_whitespace(filename: str) -> None:
    """
    Fix whitespace issues in a file.

    Args:
        filename: Path to the file to fix
    """
    # Read file content
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Remove trailing whitespace
    clean_lines = [line.rstrip() + '\n' for line in lines]

    # Ensure the file ends with a single newline
    if clean_lines and clean_lines[-1] == '\n':
        clean_lines[-1] = ''

    # Write back to file
    with open(filename, 'w') as f:
        f.writelines(clean_lines)
        f.write('\n')  # Add final newline

    print(f"Fixed whitespace in {filename}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_whitespace.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        sys.exit(1)

    fix_whitespace(filename)
