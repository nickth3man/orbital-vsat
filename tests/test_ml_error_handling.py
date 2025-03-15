"""
Unit tests for ML error handling.
"""

import unittest
import logging
from unittest.mock import Mock, patch

from src.ml.error_handling import (
    ModelLoadError,
    InferenceError,
    DiarizationError,
    SpeakerIdentificationError,
    ResourceExhaustionError,
    handle_model_load_error,
    handle_inference_error
)
from src.utils.error_handler import ErrorSeverity

class TestMLErrorHandling(unittest.TestCase):
    """Test case for ML error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure logging to avoid actual logging during tests
        logging.basicConfig(level=logging.CRITICAL)
    
    def test_model_load_error(self):
        """Test ModelLoadError creation."""
        # Create a basic error
        error = ModelLoadError("Failed to load model", "whisper-large-v3")
        self.assertEqual(str(error), "Failed to load model")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.details["model_name"], "whisper-large-v3")
        
        # Create an error with details
        error = ModelLoadError(
            "Failed to load model with CUDA", 
            "pyannote/speaker-diarization-3.1",
            {"cuda_version": "11.7", "device": "cuda:0"}
        )
        self.assertEqual(str(error), "Failed to load model with CUDA")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.details["model_name"], "pyannote/speaker-diarization-3.1")
        self.assertEqual(error.details["cuda_version"], "11.7")
        self.assertEqual(error.details["device"], "cuda:0")
    
    def test_inference_error(self):
        """Test InferenceError creation."""
        # Create a basic error
        error = InferenceError("Inference failed", "whisper-large-v3")
        self.assertEqual(str(error), "Inference failed")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.details["model_name"], "whisper-large-v3")
        
        # Create an error with details
        error = InferenceError(
            "Out of memory during inference", 
            "pyannote/speaker-diarization-3.1",
            {"input_shape": "(1, 16000, 10)", "device": "cuda:0"}
        )
        self.assertEqual(str(error), "Out of memory during inference")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.details["model_name"], "pyannote/speaker-diarization-3.1")
        self.assertEqual(error.details["input_shape"], "(1, 16000, 10)")
        self.assertEqual(error.details["device"], "cuda:0")
    
    def test_diarization_error(self):
        """Test DiarizationError creation."""
        # Create a basic error
        error = DiarizationError("Diarization failed")
        self.assertEqual(str(error), "Diarization failed")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        
        # Create an error with details
        error = DiarizationError(
            "Too many speakers detected", 
            {"num_speakers": 15, "max_supported": 10}
        )
        self.assertEqual(str(error), "Too many speakers detected")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.details["num_speakers"], 15)
        self.assertEqual(error.details["max_supported"], 10)
    
    def test_speaker_identification_error(self):
        """Test SpeakerIdentificationError creation."""
        # Create a basic error
        error = SpeakerIdentificationError("Speaker identification failed")
        self.assertEqual(str(error), "Speaker identification failed")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        
        # Create an error with details
        error = SpeakerIdentificationError(
            "Voice print generation failed", 
            {"audio_duration": 1.5, "min_duration": 3.0}
        )
        self.assertEqual(str(error), "Voice print generation failed")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.details["audio_duration"], 1.5)
        self.assertEqual(error.details["min_duration"], 3.0)
    
    def test_resource_exhaustion_error(self):
        """Test ResourceExhaustionError creation."""
        # Create a basic error
        error = ResourceExhaustionError("Out of memory", "GPU memory")
        self.assertEqual(str(error), "Out of memory")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.details["resource_type"], "GPU memory")
        
        # Create an error with details
        error = ResourceExhaustionError(
            "CPU memory exhausted", 
            "RAM",
            {"available": "2GB", "required": "8GB"}
        )
        self.assertEqual(str(error), "CPU memory exhausted")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.details["resource_type"], "RAM")
        self.assertEqual(error.details["available"], "2GB")
        self.assertEqual(error.details["required"], "8GB")
    
    def test_handle_model_load_error(self):
        """Test handle_model_load_error function."""
        # Create an original exception
        original_error = ValueError("CUDA out of memory")
        
        # Handle the error
        error = handle_model_load_error("whisper-large-v3", original_error)
        
        # Check the result
        self.assertIsInstance(error, ModelLoadError)
        self.assertEqual(error.details["model_name"], "whisper-large-v3")
        self.assertEqual(error.details["original_error"], "CUDA out of memory")
        self.assertEqual(error.details["original_error_type"], "ValueError")
    
    def test_handle_inference_error(self):
        """Test handle_inference_error function."""
        # Create an original exception
        original_error = RuntimeError("Invalid shape")
        
        # Handle the error without input shape
        error = handle_inference_error("whisper-large-v3", original_error)
        
        # Check the result
        self.assertIsInstance(error, InferenceError)
        self.assertEqual(error.details["model_name"], "whisper-large-v3")
        self.assertEqual(error.details["original_error"], "Invalid shape")
        self.assertEqual(error.details["original_error_type"], "RuntimeError")
        self.assertNotIn("input_shape", error.details)
        
        # Handle the error with input shape
        error = handle_inference_error("whisper-large-v3", original_error, "(1, 16000)")
        
        # Check the result
        self.assertIsInstance(error, InferenceError)
        self.assertEqual(error.details["model_name"], "whisper-large-v3")
        self.assertEqual(error.details["original_error"], "Invalid shape")
        self.assertEqual(error.details["original_error_type"], "RuntimeError")
        self.assertEqual(error.details["input_shape"], "(1, 16000)")


if __name__ == '__main__':
    unittest.main() 