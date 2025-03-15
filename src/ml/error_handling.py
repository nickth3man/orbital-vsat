"""
Error handling for ML components.

This module provides specialized error handling for ML components in VSAT.
"""

from typing import Dict, Any, Optional
from ..utils.error_handler import VSATError, ErrorSeverity

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


class DiarizationError(VSATError):
    """Error raised when speaker diarization fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, ErrorSeverity.ERROR, details)


class SpeakerIdentificationError(VSATError):
    """Error raised when speaker identification fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, ErrorSeverity.ERROR, details)


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


def handle_model_load_error(model_name: str, original_error: Exception) -> ModelLoadError:
    """Create a ModelLoadError from an original exception.
    
    Args:
        model_name: Name of the model that failed to load
        original_error: Original exception that was raised
        
    Returns:
        ModelLoadError: A new ModelLoadError instance
    """
    return ModelLoadError(
        f"Failed to load model '{model_name}': {str(original_error)}",
        model_name,
        {"original_error": str(original_error), "original_error_type": type(original_error).__name__}
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
    details = {
        "original_error": str(original_error),
        "original_error_type": type(original_error).__name__
    }
    
    if input_shape is not None:
        details["input_shape"] = str(input_shape)
        
    return InferenceError(
        f"Model inference failed for '{model_name}': {str(original_error)}",
        model_name,
        details
    ) 