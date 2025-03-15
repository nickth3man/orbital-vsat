"""
Transcription module for VSAT.

This module provides functionality for transcribing audio and aligning word timestamps.
"""

from src.transcription.whisper_transcriber import WhisperTranscriber
from src.transcription.word_aligner import WordAligner

__all__ = ['WhisperTranscriber', 'WordAligner'] 