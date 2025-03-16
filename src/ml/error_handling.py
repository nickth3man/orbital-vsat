"""
Error handling for ML components.

This module provides specialized error handling for ML components in VSAT.
"""

import logging
import traceback
import torch
import psutil
import os
import time
from typing import Dict, Any, Optional, Callable, List, Tuple, Union

from ..utils.error_handler import VSATError, ErrorSeverity

logger = logging.getLogger(__name__)

# Dictionary of error types and their descriptions
ERROR_TYPES = {
    # Model-related errors
    "model_load": "Failed to load model",
    "model_download": "Failed to download model",
    "model_initialization": "Failed to initialize model",
    "model_not_found": "Model not found",
    
    # Resource-related errors
    "cuda_out_of_memory": "CUDA out of memory",
    "cpu_out_of_memory": "CPU out of memory",
    "disk_space": "Insufficient disk space",
    
    # Processing-related errors
    "audio_too_short": "Audio too short for processing",
    "audio_too_long": "Audio too long for efficient processing",
    "invalid_audio": "Invalid audio format or content",
    "processing_timeout": "Processing timeout",
    "inference_failure": "Model inference failed",
    
    # System-related errors
    "file_access": "Failed to access file",
    "network_error": "Network error",
    "authentication_failure": "Authentication failure",
    
    # Other errors
    "invalid_parameters": "Invalid parameters",
    "unexpected_error": "Unexpected error"
}

# Recovery strategies for different error types
RECOVERY_STRATEGIES = {
    "cuda_out_of_memory": [
        "Try using a smaller model",
        "Try processing in smaller chunks",
        "Try using CPU instead of GPU",
        "Free up GPU memory by closing other applications"
    ],
    "audio_too_long": [
        "Process the audio in chunks",
        "Use a more efficient model",
        "Reduce audio quality"
    ],
    "network_error": [
        "Check internet connection",
        "Try again later",
        "Use cached models"
    ],
    "invalid_audio": [
        "Check audio format",
        "Convert to a supported format (WAV, MP3, FLAC)",
        "Check for corruption",
        "Try preprocessing the audio"
    ],
    "processing_timeout": [
        "Try processing in smaller chunks",
        "Try a faster model",
        "Increase timeout threshold"
    ]
}

class ModelLoadError(VSATError):
    """Error raised when a model fails to load."""
    
    def __init__(self, message: str, model_name: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            model_name: Name of the model that failed to load
            details: Additional error details
        """
        if details is None:
            details = {}
        details["model_name"] = model_name
        super().__init__(message, ErrorSeverity.ERROR, details)
        
        # Log the error
        logger.error(f"Model load error: {message} (model: {model_name})")
        if "traceback" in details:
            logger.debug(f"Traceback: {details['traceback']}")


class InferenceError(VSATError):
    """Error raised when model inference fails."""
    
    def __init__(self, message: str, model_name: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            model_name: Name of the model that failed during inference
            details: Additional error details
        """
        if details is None:
            details = {}
        details["model_name"] = model_name
        super().__init__(message, ErrorSeverity.ERROR, details)
        
        # Log the error
        logger.error(f"Inference error: {message} (model: {model_name})")
        if "traceback" in details:
            logger.debug(f"Traceback: {details['traceback']}")
        
        # Add recovery strategies
        error_type = details.get("error_type", "unexpected_error")
        if error_type in RECOVERY_STRATEGIES:
            details["recovery_strategies"] = RECOVERY_STRATEGIES[error_type]


class DiarizationError(VSATError):
    """Error raised when speaker diarization fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, ErrorSeverity.ERROR, details)
        
        # Log the error
        logger.error(f"Diarization error: {message}")
        if details and "traceback" in details:
            logger.debug(f"Traceback: {details['traceback']}")
        
        # Add recovery strategies based on error type
        if details and "error_type" in details:
            error_type = details["error_type"]
            if error_type in RECOVERY_STRATEGIES:
                details["recovery_strategies"] = RECOVERY_STRATEGIES[error_type]
                
                # Add automatic retry suggestion for specific errors
                if error_type in ["cuda_out_of_memory", "processing_timeout", "network_error"]:
                    details["can_retry"] = True
                    details["retry_with_different_params"] = True


class SpeakerIdentificationError(VSATError):
    """Error raised when speaker identification fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, ErrorSeverity.ERROR, details)
        
        # Log the error
        logger.error(f"Speaker identification error: {message}")
        if details and "traceback" in details:
            logger.debug(f"Traceback: {details['traceback']}")
        
        # Add recovery strategies based on error type
        if details and "error_type" in details:
            error_type = details["error_type"]
            if error_type in RECOVERY_STRATEGIES:
                details["recovery_strategies"] = RECOVERY_STRATEGIES[error_type]


class AudioProcessingError(VSATError):
    """Error raised when audio processing fails."""
    
    def __init__(self, message: str, error_type: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            error_type: Type of error (e.g., "processing", "file_format")
            details: Additional error details
        """
        if details is None:
            details = {}
        details["error_type"] = error_type
        super().__init__(message, ErrorSeverity.ERROR, details)
        
        # Log the error
        logger.error(f"Audio processing error: {message}")
        if "traceback" in details:
            logger.debug(f"Traceback: {details['traceback']}")
        
        # Add recovery strategies based on error type
        if error_type in RECOVERY_STRATEGIES:
            details["recovery_strategies"] = RECOVERY_STRATEGIES[error_type]


class ResourceExhaustionError(VSATError):
    """Error raised when system resources are exhausted during ML processing."""
    
    def __init__(self, message: str, resource_type: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            resource_type: Type of resource that was exhausted (e.g., "memory", "GPU memory")
            details: Additional error details
        """
        if details is None:
            details = {}
        details["resource_type"] = resource_type
        super().__init__(message, ErrorSeverity.ERROR, details)
        
        # Log the error
        logger.error(f"Resource exhaustion: {message} (resource: {resource_type})")
        if "traceback" in details:
            logger.debug(f"Traceback: {details['traceback']}")
        
        # Add recovery strategies
        if resource_type == "GPU memory":
            details["recovery_strategies"] = RECOVERY_STRATEGIES["cuda_out_of_memory"]
            details["can_retry"] = True
            details["retry_with_different_params"] = True
            details["retry_on_cpu"] = True
        elif resource_type == "memory":
            details["recovery_strategies"] = [
                "Process in smaller chunks",
                "Close other applications",
                "Reduce model size"
            ]
            details["can_retry"] = True
            details["retry_with_different_params"] = True


class TranscriptionError(VSATError):
    """Error raised when transcription fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, ErrorSeverity.ERROR, details)
        
        # Log the error
        logger.error(f"Transcription error: {message}")
        if details and "traceback" in details:
            logger.debug(f"Traceback: {details['traceback']}")
        
        # Add recovery strategies based on error type
        if details and "error_type" in details:
            error_type = details["error_type"]
            if error_type in RECOVERY_STRATEGIES:
                details["recovery_strategies"] = RECOVERY_STRATEGIES[error_type]


def handle_model_load_error(model_name: str, original_error: Exception) -> ModelLoadError:
    """Create a ModelLoadError from an original exception.
    
    Args:
        model_name: Name of the model that failed to load
        original_error: Original exception that was raised
        
    Returns:
        ModelLoadError: A new ModelLoadError instance
    """
    error_details = {
        "original_error": str(original_error),
        "original_error_type": type(original_error).__name__,
        "traceback": traceback.format_exc()
    }
    
    # Determine error type for more specific handling
    error_message = str(original_error).lower()
    if "cuda" in error_message and "memory" in error_message:
        error_details["error_type"] = "cuda_out_of_memory"
        error_details["recovery_strategies"] = RECOVERY_STRATEGIES["cuda_out_of_memory"]
        error_details["can_retry"] = True
        error_details["retry_on_cpu"] = True
        
        # Add GPU memory info if available
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    error_details[f"gpu_{i}_memory_allocated"] = f"{torch.cuda.memory_allocated(i) / (1024**2):.2f} MB"
                    error_details[f"gpu_{i}_memory_reserved"] = f"{torch.cuda.memory_reserved(i) / (1024**2):.2f} MB"
        except Exception:
            pass
    elif "network" in error_message or "download" in error_message or "connect" in error_message:
        error_details["error_type"] = "network_error"
        error_details["recovery_strategies"] = RECOVERY_STRATEGIES["network_error"]
        error_details["can_retry"] = True
    elif "auth" in error_message or "token" in error_message or "permission" in error_message:
        error_details["error_type"] = "authentication_failure"
        error_details["recovery_strategies"] = [
            "Check your HuggingFace token",
            "Verify API key permissions",
            "Check credentials"
        ]
    elif "not found" in error_message or "does not exist" in error_message:
        error_details["error_type"] = "model_not_found"
        error_details["recovery_strategies"] = [
            "Check model name for typos",
            "Verify the model exists on the hub",
            "Try an alternative model"
        ]
    else:
        # Generic model load error
        error_details["error_type"] = "model_load"
        
    return ModelLoadError(
        f"Failed to load model '{model_name}': {str(original_error)}",
        model_name,
        error_details
    )


def handle_inference_error(model_name: str, original_error: Exception, input_shape: Optional[Any] = None) -> InferenceError:
    """Create an InferenceError from an original exception.
    
    Args:
        model_name: Name of the model that failed during inference
        original_error: Original exception that was raised
        input_shape: Shape of the input data that caused the error
        
    Returns:
        InferenceError: A new InferenceError instance
    """
    error_details = {
        "original_error": str(original_error),
        "original_error_type": type(original_error).__name__,
        "traceback": traceback.format_exc()
    }
    
    if input_shape is not None:
        error_details["input_shape"] = str(input_shape)
    
    # Determine error type for more specific handling
    error_message = str(original_error).lower()
    if "cuda" in error_message and "memory" in error_message:
        error_details["error_type"] = "cuda_out_of_memory"
        error_details["recovery_strategies"] = RECOVERY_STRATEGIES["cuda_out_of_memory"]
        error_details["can_retry"] = True
        error_details["retry_on_cpu"] = True
        
        # Add GPU memory info if available
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    error_details[f"gpu_{i}_memory_allocated"] = f"{torch.cuda.memory_allocated(i) / (1024**2):.2f} MB"
                    error_details[f"gpu_{i}_memory_reserved"] = f"{torch.cuda.memory_reserved(i) / (1024**2):.2f} MB"
        except Exception:
            pass
    elif "shape" in error_message or "dimension" in error_message or "size" in error_message:
        error_details["error_type"] = "input_shape_mismatch"
        error_details["recovery_strategies"] = [
            "Check input dimensions",
            "Resize or reshape input data",
            "Check model input requirements"
        ]
    elif "timeout" in error_message or "timed out" in error_message:
        error_details["error_type"] = "processing_timeout"
        error_details["recovery_strategies"] = RECOVERY_STRATEGIES["processing_timeout"]
        error_details["can_retry"] = True
    else:
        error_details["error_type"] = "inference_failure"
        
    return InferenceError(
        f"Model inference failed for '{model_name}': {str(original_error)}",
        model_name,
        error_details
    )


def handle_errors(error: Exception, context: str, fallback_severity: ErrorSeverity):
    """Handle errors from ML components with enhanced diagnostics and recovery.
    
    Args:
        error: The exception that occurred
        context: Context description where the error occurred
        fallback_severity: Severity to use if not specified by the error
    """
    # Basic error logging
    logger.error(f"{context}: {str(error)}")
    
    # Enhanced error diagnostics for ML errors
    if isinstance(error, VSATError):
        # Log detailed error information for ML errors
        logger.error(f"Error type: {error.__class__.__name__}")
        if hasattr(error, "details") and error.details:
            logger.error(f"Error details: {error.details}")
            
            # Log recovery strategies if available
            if "recovery_strategies" in error.details:
                strategies = error.details["recovery_strategies"]
                strategies_str = ", ".join(strategies)
                logger.info(f"Recovery strategies: {strategies_str}")
            
            # Log if an automatic retry is possible
            if error.details.get("can_retry", False):
                logger.info("This error can be retried automatically")
                
                if error.details.get("retry_on_cpu", False):
                    logger.info("Consider retrying on CPU")
                
                if error.details.get("retry_with_different_params", False):
                    logger.info("Consider retrying with different parameters")
    
    # Log system resource information for diagnostic purposes
    try:
        # Log CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        logger.debug(f"System diagnostics - CPU: {cpu_percent}%, Memory: {memory.percent}% used ({memory.available / (1024**3):.2f} GB available)")
        
        # Log GPU usage if available
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                logger.debug(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / (1024**2):.2f} MB")
                logger.debug(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / (1024**2):.2f} MB")
    except Exception as e:
        logger.debug(f"Error collecting system diagnostics: {e}")


def retry_operation(operation: Callable, max_retries: int = 3, 
                   initial_delay: float = 1.0, backoff_factor: float = 2.0, 
                   fallback_fn: Optional[Callable] = None,
                   allowed_exceptions: Optional[List[type]] = None) -> Any:
    """Retry an operation with exponential backoff.
    
    Args:
        operation: The function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before the first retry
        backoff_factor: Factor by which to increase the delay on each retry
        fallback_fn: Function to call if all retries fail
        allowed_exceptions: List of exception types that trigger retry (defaults to all exceptions)
        
    Returns:
        Result of the operation or fallback function
        
    Raises:
        Exception: The last exception encountered if all retries fail and no fallback_fn is provided
    """
    if allowed_exceptions is None:
        allowed_exceptions = [Exception]
        
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except tuple(allowed_exceptions) as e:
            last_exception = e
            
            if attempt < max_retries:
                logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"Operation failed after {max_retries + 1} attempts: {str(e)}")
                
    # If we reach here, all retries have failed
    if fallback_fn is not None:
        logger.info("Trying fallback function...")
        return fallback_fn()
    else:
        # Re-raise the last exception
        raise last_exception 