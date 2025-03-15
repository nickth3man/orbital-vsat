"""
Machine learning components for VSAT.

This package contains modules for speaker diarization, speaker identification,
and other ML-related functionality.
"""

from .diarization import Diarizer
from .speaker_identification import SpeakerIdentifier
from .voice_print_processor import VoicePrintProcessor
from .voice_activity_detection import VoiceActivityDetector
from .error_handling import (
    ModelLoadError, 
    InferenceError, 
    DiarizationError, 
    SpeakerIdentificationError,
    ResourceExhaustionError,
    handle_model_load_error,
    handle_inference_error
)

__all__ = [
    'Diarizer', 
    'SpeakerIdentifier', 
    'VoicePrintProcessor',
    'VoiceActivityDetector',
    'ModelLoadError',
    'InferenceError',
    'DiarizationError',
    'SpeakerIdentificationError',
    'ResourceExhaustionError',
    'handle_model_load_error',
    'handle_inference_error'
] 