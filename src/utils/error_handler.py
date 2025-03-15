"""
Error handling utilities for VSAT.

This module provides error handling utilities for the VSAT application.
"""

import logging
import traceback
import sys
from enum import Enum
from typing import Dict, Any, Optional, Callable, Type, List, Union
from PyQt6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


class VSATError(Exception):
    """Base exception class for VSAT-specific errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            severity: Error severity level
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.details = details or {}
        
        # Log the error
        self._log_error()
    
    def _log_error(self):
        """Log the error based on its severity."""
        log_funcs = {
            ErrorSeverity.INFO: logger.info,
            ErrorSeverity.WARNING: logger.warning,
            ErrorSeverity.ERROR: logger.error,
            ErrorSeverity.CRITICAL: logger.critical
        }
        
        log_func = log_funcs.get(self.severity, logger.error)
        log_func(f"{self.__class__.__name__}: {self.message}")
        
        if self.details:
            log_func(f"Error details: {self.details}")


class FileError(VSATError):
    """Exception for file-related errors."""
    pass


class AudioError(VSATError):
    """Exception for audio processing errors."""
    pass


class ProcessingError(VSATError):
    """Exception for audio processing pipeline errors."""
    pass


class DatabaseError(VSATError):
    """Exception for database-related errors."""
    pass


class ExportError(VSATError):
    """Exception for export-related errors."""
    pass


class UIError(VSATError):
    """Exception for UI-related errors."""
    pass


class DiarizationError(VSATError):
    """Error raised when speaker diarization fails."""
    pass


class TranscriptionError(VSATError):
    """Error raised when transcription fails."""
    pass


class ConfigError(VSATError):
    """Error raised when configuration operations fail."""
    pass


class NetworkError(VSATError):
    """Error raised when network operations fail."""
    pass


class ModelError(VSATError):
    """Error raised when model operations fail."""
    pass


class ChunkingError(VSATError):
    """Error raised when chunking operations fail."""
    pass


class ErrorHandler:
    """Class for handling errors in the VSAT application."""
    
    # Mapping of exception types to user-friendly messages
    ERROR_MESSAGES = {
        FileError: "File Operation Error",
        AudioError: "Audio Processing Error",
        ProcessingError: "Processing Pipeline Error",
        DatabaseError: "Database Error",
        ExportError: "Export Error",
        UIError: "User Interface Error",
        Exception: "Unexpected Error"
    }
    
    @staticmethod
    def handle_exception(
        exception: Exception, 
        show_dialog: bool = True,
        parent=None,
        callback: Optional[Callable[[Exception], None]] = None
    ) -> bool:
        """Handle an exception.
        
        Args:
            exception: The exception to handle
            show_dialog: Whether to show a dialog to the user
            parent: Parent widget for the dialog
            callback: Optional callback function to call after handling
            
        Returns:
            bool: True if the exception was handled, False otherwise
        """
        # Log the exception
        if isinstance(exception, VSATError):
            # Already logged in VSATError.__init__
            pass
        else:
            logger.error(f"Unhandled exception: {str(exception)}")
            logger.error(traceback.format_exc())
        
        # Get error message
        error_type = type(exception)
        title = ErrorHandler.ERROR_MESSAGES.get(error_type, ErrorHandler.ERROR_MESSAGES[Exception])
        
        message = str(exception)
        details = traceback.format_exc() if not isinstance(exception, VSATError) else None
        
        # Show dialog if requested
        if show_dialog:
            ErrorHandler.show_error_dialog(title, message, details, parent)
        
        # Call callback if provided
        if callback:
            callback(exception)
        
        return True
    
    @staticmethod
    def show_error_dialog(title: str, message: str, details: Optional[str] = None, parent=None):
        """Show an error dialog to the user.
        
        Args:
            title: Dialog title
            message: Error message
            details: Additional error details
            parent: Parent widget for the dialog
        """
        dialog = QMessageBox(parent)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        
        if details:
            dialog.setDetailedText(details)
        
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        dialog.exec()
    
    @staticmethod
    def global_exception_hook(exc_type: Type[Exception], exc_value: Exception, exc_traceback):
        """Global exception hook for uncaught exceptions.
        
        This function can be set as sys.excepthook to catch all uncaught exceptions.
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        logger.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Format exception for display
        error_msg = str(exc_value)
        error_details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Show error dialog
        ErrorHandler.show_error_dialog("Critical Error", error_msg, error_details)


def install_global_error_handler():
    """Install the global exception hook."""
    sys.excepthook = ErrorHandler.global_exception_hook
    logger.info("Global error handler installed") 