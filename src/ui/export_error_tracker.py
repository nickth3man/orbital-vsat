"""
Export error tracking and recovery module for VSAT.

This module provides a tracking system for export errors, with recovery options
and comprehensive logging to improve user experience during export operations.
"""

import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from src.utils.error_handler import ExportError, FileError, ErrorSeverity

logger = logging.getLogger(__name__)

class ExportOperation(Enum):
    """Types of export operations."""
    TRANSCRIPT = "transcript"
    AUDIO_SEGMENT = "audio_segment"
    SPEAKER_AUDIO = "speaker_audio"
    SELECTION = "selection"
    BATCH = "batch"

@dataclass
class ExportAttempt:
    """Represents a single export attempt with error tracking."""
    operation: ExportOperation
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    target_path: Optional[str] = None
    
    def duration(self) -> float:
        """Get the duration of the export attempt in seconds."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def mark_success(self):
        """Mark the attempt as successful."""
        self.end_time = time.time()
        self.success = True
    
    def mark_failure(self, error: Exception):
        """Mark the attempt as failed."""
        self.end_time = time.time()
        self.success = False
        
        if isinstance(error, ExportError) or isinstance(error, FileError):
            self.error_message = str(error)
            self.error_details = error.context
        else:
            self.error_message = str(error)
            self.error_details = {"exception_type": type(error).__name__}
    
    def can_retry(self, max_retries: int = 3) -> bool:
        """Check if the operation can be retried."""
        return self.retry_count < max_retries and not self.success
    
    def increment_retry(self):
        """Increment the retry count."""
        self.retry_count += 1
        

class ExportErrorTracker(QObject):
    """Tracks export errors and provides recovery options."""
    
    # Signal emitted when an export operation starts
    exportStarted = pyqtSignal(ExportOperation, str)  # operation, target path
    
    # Signal emitted when an export operation completes
    exportCompleted = pyqtSignal(ExportOperation, str, bool)  # operation, target path, success
    
    # Signal emitted when an export operation fails
    exportFailed = pyqtSignal(ExportOperation, str, str)  # operation, error message, target path
    
    # Signal emitted when a retry is attempted
    retryAttempted = pyqtSignal(ExportOperation, int)  # operation, retry count
    
    def __init__(self, parent=None):
        """Initialize the export error tracker.
        
        Args:
            parent: Parent QObject
        """
        super().__init__(parent)
        self.attempts = []  # List of ExportAttempt objects
        self.current_attempt = None
        self.max_retries = 3
        
    def start_tracking(self, operation: ExportOperation, target_path: str = None) -> ExportAttempt:
        """Start tracking a new export operation.
        
        Args:
            operation: Type of export operation
            target_path: Path to the export target file
            
        Returns:
            ExportAttempt: The new tracking object
        """
        attempt = ExportAttempt(
            operation=operation,
            start_time=time.time(),
            target_path=target_path
        )
        
        self.attempts.append(attempt)
        self.current_attempt = attempt
        
        # Emit signal
        self.exportStarted.emit(operation, target_path or "")
        
        logger.debug(f"Started tracking export operation: {operation.value}")
        return attempt
    
    def mark_success(self):
        """Mark the current export operation as successful."""
        if self.current_attempt:
            self.current_attempt.mark_success()
            
            # Emit signal
            self.exportCompleted.emit(
                self.current_attempt.operation,
                self.current_attempt.target_path or "",
                True
            )
            
            logger.info(f"Export operation completed successfully: {self.current_attempt.operation.value}")
            
            # Clear current attempt
            self.current_attempt = None
    
    def mark_failure(self, error: Exception):
        """Mark the current export operation as failed.
        
        Args:
            error: The exception that caused the failure
        """
        if self.current_attempt:
            self.current_attempt.mark_failure(error)
            
            # Emit signal
            self.exportFailed.emit(
                self.current_attempt.operation,
                str(error),
                self.current_attempt.target_path or ""
            )
            
            logger.error(
                f"Export operation failed: {self.current_attempt.operation.value} - {str(error)}",
                exc_info=True
            )
    
    def attempt_retry(self, retry_func: Callable[[], None]) -> bool:
        """Attempt to retry the current export operation.
        
        Args:
            retry_func: Function to call to retry the operation
            
        Returns:
            bool: True if retry was attempted, False otherwise
        """
        if not self.current_attempt:
            logger.warning("Attempted retry with no active export operation")
            return False
        
        if not self.current_attempt.can_retry(self.max_retries):
            logger.warning(
                f"Max retries ({self.max_retries}) reached for {self.current_attempt.operation.value}"
            )
            return False
        
        # Increment retry count
        self.current_attempt.increment_retry()
        
        # Emit signal
        self.retryAttempted.emit(
            self.current_attempt.operation,
            self.current_attempt.retry_count
        )
        
        logger.info(
            f"Retrying export operation ({self.current_attempt.retry_count}/{self.max_retries}): "
            f"{self.current_attempt.operation.value}"
        )
        
        # Call retry function
        try:
            retry_func()
            return True
        except Exception as e:
            logger.error(f"Retry attempt failed: {str(e)}", exc_info=True)
            self.mark_failure(e)
            return False
    
    def get_recent_failures(self, limit: int = 10) -> List[ExportAttempt]:
        """Get the most recent failed export attempts.
        
        Args:
            limit: Maximum number of failures to return
            
        Returns:
            List[ExportAttempt]: List of failed attempts
        """
        failures = [a for a in self.attempts if not a.success]
        return sorted(failures, key=lambda a: a.start_time, reverse=True)[:limit]
    
    def get_failure_stats(self) -> Dict[ExportOperation, Dict[str, Any]]:
        """Get statistics about export failures.
        
        Returns:
            Dict: Dictionary with stats for each operation type
        """
        stats = {}
        
        for op in ExportOperation:
            op_attempts = [a for a in self.attempts if a.operation == op]
            op_failures = [a for a in op_attempts if not a.success]
            
            if not op_attempts:
                continue
                
            stats[op] = {
                "total": len(op_attempts),
                "failures": len(op_failures),
                "failure_rate": len(op_failures) / len(op_attempts) if op_attempts else 0,
                "avg_duration": sum(a.duration() for a in op_attempts) / len(op_attempts) if op_attempts else 0
            }
            
        return stats 