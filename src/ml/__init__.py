"""
Machine learning module for VSAT.

This module provides ML-based functionality for the VSAT application:
- Speaker diarization
- Speaker identification
- Voice print processing
- Error handling for ML operations
- Voice activity detection
- Content analysis
"""

# Import diarization components
from src.ml.diarization import (
    Diarizer,
    DiarizationResult,
    Speaker,
    Segment,
    DiarizationConfig
)

# Import speaker identification components
from src.ml.speaker_identification import (
    SpeakerIdentifier,
    SpeakerMatch
)

# Import voice print processor
from src.ml.voice_print_processor import (
    VoicePrintProcessor
)

# Import ML error handling
from src.ml.error_handling import (
    ModelLoadError,
    InferenceError,
    DiarizationError,
    SpeakerIdentificationError,
    ResourceExhaustionError,
    handle_model_load_error,
    handle_inference_error
)

# Import voice activity detection
from src.ml.voice_activity_detection import (
    VoiceActivityDetector,
    SensitivityPreset
)

# Import content analysis
from src.ml.content_analysis import (
    ContentAnalyzer,
    TopicModeler,
    KeywordExtractor,
    Summarizer,
    ImportantMomentDetector,
    ContentAnalysisError
)

__all__ = [
    'Diarizer',
    'DiarizationResult',
    'Speaker',
    'Segment',
    'DiarizationConfig',
    'SpeakerIdentifier',
    'SpeakerMatch',
    'VoicePrintProcessor',
    'ModelLoadError',
    'InferenceError', 
    'DiarizationError',
    'SpeakerIdentificationError',
    'ResourceExhaustionError',
    'handle_model_load_error',
    'handle_inference_error',
    'VoiceActivityDetector',
    'SensitivityPreset',
    'ContentAnalyzer',
    'TopicModeler',
    'KeywordExtractor',
    'Summarizer',
    'ImportantMomentDetector',
    'ContentAnalysisError'
] 