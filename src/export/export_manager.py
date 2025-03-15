"""
Export functionality for VSAT.

This module provides functionality for exporting audio segments, transcripts, and other data.
"""

import os
import json
import logging
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from src.audio.file_handler import AudioFileHandler
from src.utils.error_handler import ExportError, FileError, ErrorSeverity
from src.export.transcript_exporter import TranscriptExporter
from src.export.audio_exporter import AudioExporter

logger = logging.getLogger(__name__)

class ExportManager:
    """Class for managing export operations."""
    
    # Supported transcript export formats
    TRANSCRIPT_FORMATS = {
        'txt': 'Plain Text',
        'srt': 'SubRip Subtitle',
        'vtt': 'WebVTT Subtitle',
        'json': 'JSON',
        'csv': 'CSV'
    }
    
    # Supported audio export formats
    AUDIO_FORMATS = {
        'wav': 'WAV Audio',
        'mp3': 'MP3 Audio',
        'flac': 'FLAC Audio'
    }
    
    def __init__(self):
        """Initialize the export manager."""
        self.file_handler = AudioFileHandler()
        self.transcript_exporter = TranscriptExporter()
        self.audio_exporter = AudioExporter()
        logger.debug("ExportManager initialized")
    
    def export_transcript(self, segments: List[Dict[str, Any]], output_path: str, 
                         format_type: str = 'txt', include_speaker: bool = True,
                         include_timestamps: bool = True) -> bool:
        """Export transcript segments to a file.
        
        Args:
            segments: List of transcript segments to export
            output_path: Path to save the exported transcript
            format_type: Format to export (txt, srt, vtt, json, csv)
            include_speaker: Whether to include speaker information
            include_timestamps: Whether to include timestamps
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Check if format is supported
            if format_type not in self.TRANSCRIPT_FORMATS:
                raise ExportError(
                    f"Unsupported transcript format: {format_type}",
                    severity=ErrorSeverity.ERROR
                )
            
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Export based on format
            return self.transcript_exporter.export_transcript(
                segments, output_path, format_type, include_speaker, include_timestamps
            )
            
        except (ExportError, FileError) as e:
            # Re-raise export errors
            raise e
        except Exception as e:
            # Wrap other exceptions
            raise ExportError(
                f"Failed to export transcript: {str(e)}",
                severity=ErrorSeverity.ERROR
            ) from e
    
    def export_audio_segment(self, audio_file: str, output_path: str, 
                            start: float, end: float, format_type: str = 'wav') -> bool:
        """Export a segment of audio to a file.
        
        Args:
            audio_file: Path to the source audio file
            output_path: Path to save the exported audio
            start: Start time in seconds
            end: End time in seconds
            format_type: Format to export (wav, mp3, flac)
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Check if format is supported
            if format_type not in self.AUDIO_FORMATS:
                raise ExportError(
                    f"Unsupported audio format: {format_type}",
                    severity=ErrorSeverity.ERROR
                )
            
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            return self.audio_exporter.export_audio_segment(
                audio_file, output_path, start, end, format_type
            )
            
        except (ExportError, FileError) as e:
            # Re-raise export errors
            raise e
        except Exception as e:
            # Wrap other exceptions
            raise ExportError(
                f"Failed to export audio segment: {str(e)}",
                severity=ErrorSeverity.ERROR
            ) from e
    
    def export_speaker_audio(self, audio_file: str, segments: List[Dict[str, Any]], 
                            output_dir: str, speaker_id: Union[int, str], 
                            format_type: str = 'wav') -> bool:
        """Export all segments for a specific speaker to separate audio files.
        
        Args:
            audio_file: Path to the source audio file
            segments: List of transcript segments
            output_dir: Directory to save the exported audio files
            speaker_id: ID of the speaker to export
            format_type: Format to export (wav, mp3, flac)
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Check if format is supported
            if format_type not in self.AUDIO_FORMATS:
                raise ExportError(
                    f"Unsupported audio format: {format_type}",
                    severity=ErrorSeverity.ERROR
                )
            
            # Create directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            return self.audio_exporter.export_speaker_audio(
                audio_file, segments, output_dir, speaker_id, format_type
            )
            
        except (ExportError, FileError) as e:
            # Re-raise export errors
            raise e
        except Exception as e:
            # Wrap other exceptions
            raise ExportError(
                f"Failed to export speaker audio: {str(e)}",
                severity=ErrorSeverity.ERROR
            ) from e
    
    def export_word_audio(self, audio_file: str, word: Dict[str, Any], 
                         output_path: str, format_type: str = 'wav',
                         padding_ms: int = 50) -> bool:
        """Export a single word to an audio file.
        
        Args:
            audio_file: Path to the source audio file
            word: Word data with start and end times
            output_path: Path to save the exported audio
            format_type: Format to export (wav, mp3, flac)
            padding_ms: Padding in milliseconds to add before and after the word
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Check if format is supported
            if format_type not in self.AUDIO_FORMATS:
                raise ExportError(
                    f"Unsupported audio format: {format_type}",
                    severity=ErrorSeverity.ERROR
                )
            
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            return self.audio_exporter.export_word_audio(
                audio_file, word, output_path, format_type, padding_ms
            )
            
        except (ExportError, FileError) as e:
            # Re-raise export errors
            raise e
        except Exception as e:
            # Wrap other exceptions
            raise ExportError(
                f"Failed to export word audio: {str(e)}",
                severity=ErrorSeverity.ERROR
            ) from e
    
    def export_selection(self, audio_file: str, words: List[Dict[str, Any]], 
                        output_path: str, format_type: str = 'wav',
                        include_transcript: bool = True) -> bool:
        """Export a selection of words to an audio file.
        
        Args:
            audio_file: Path to the source audio file
            words: List of word data with start and end times
            output_path: Path to save the exported audio
            format_type: Format to export (wav, mp3, flac)
            include_transcript: Whether to include a transcript file
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Check if format is supported
            if format_type not in self.AUDIO_FORMATS:
                raise ExportError(
                    f"Unsupported audio format: {format_type}",
                    severity=ErrorSeverity.ERROR
                )
            
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            return self.audio_exporter.export_selection(
                audio_file, words, output_path, format_type, include_transcript
            )
            
        except (ExportError, FileError) as e:
            # Re-raise export errors
            raise e
        except Exception as e:
            # Wrap other exceptions
            raise ExportError(
                f"Failed to export selection: {str(e)}",
                severity=ErrorSeverity.ERROR
            ) from e 