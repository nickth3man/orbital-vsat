"""
Utility module for VSAT.

This module provides utility functions and classes for the application.
"""

from src.utils.error_handler import (
    ErrorSeverity,
    VSATError,
    FileError,
    AudioError,
    ProcessingError,
    DatabaseError,
    ExportError,
    UIError,
    ErrorHandler,
    install_global_error_handler
)

__all__ = [
    # Error handling
    'ErrorSeverity',
    'VSATError',
    'FileError',
    'AudioError',
    'ProcessingError',
    'DatabaseError',
    'ExportError',
    'UIError',
    'ErrorHandler',
    'install_global_error_handler'
] 