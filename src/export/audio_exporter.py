"""
Audio export functionality for VSAT.

This module provides functionality for exporting audio segments in various formats.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union

import numpy as np
import soundfile as sf

from src.audio.file_handler import AudioFileHandler
from src.utils.error_handler import ExportError, FileError, ErrorSeverity

logger = logging.getLogger(__name__)

class AudioExporter:
    """Class for exporting audio in various formats."""
    
    def __init__(self):
        """Initialize the audio exporter."""
        self.file_handler = AudioFileHandler()
        logger.debug("AudioExporter initialized")
    
    def export_audio_segment(self, audio_file: str, output_path: str, 
                            start: float, end: float, format_type: str = 'wav') -> bool:
        """Export a segment of an audio file.
        
        Args:
            audio_file: Path to the audio file
            output_path: Path to save the exported file
            start: Start time in seconds
            end: End time in seconds
            format_type: Export format type ('wav', 'mp3', 'flac')
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ExportError: If there's an error during export
            FileError: If there's an error with file operations
        """
        try:
            # Check if audio file exists
            if not os.path.exists(audio_file):
                raise FileError(
                    f"Audio file not found: {audio_file}",
                    ErrorSeverity.ERROR,
                    {"audio_file": audio_file}
                )
            
            # Load audio file
            audio_data, sample_rate, _ = self.file_handler.load_audio(audio_file)
            
            # Convert stereo to mono if needed
            if audio_data.ndim > 1 and audio_data.shape[0] > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            # Calculate start and end samples
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            
            # Validate time range
            if start >= end:
                raise ExportError(
                    f"Invalid time range: start ({start}) >= end ({end})",
                    ErrorSeverity.ERROR,
                    {"start": start, "end": end}
                )
            
            # Ensure valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            # Check if range is valid
            if start_sample >= end_sample:
                raise ExportError(
                    "Invalid sample range for export",
                    ErrorSeverity.ERROR,
                    {"start_sample": start_sample, "end_sample": end_sample}
                )
            
            # Extract segment
            segment_data = audio_data[start_sample:end_sample]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save segment
            sf.write(output_path, segment_data, sample_rate)
            
            logger.info(f"Exported audio segment to {output_path} ({start:.2f}s to {end:.2f}s)")
            return True
            
        except (PermissionError, IOError) as e:
            raise FileError(
                f"File operation error: {str(e)}",
                ErrorSeverity.ERROR,
                {"audio_file": audio_file, "output_path": output_path, "error": str(e)}
            ) from e
        except Exception as e:
            # Catch-all for unexpected errors
            raise ExportError(
                f"Unexpected error during audio segment export: {str(e)}",
                ErrorSeverity.ERROR,
                {"audio_file": audio_file, "output_path": output_path, "start": start, "end": end}
            ) from e
    
    def export_speaker_audio(self, audio_file: str, segments: List[Dict[str, Any]], 
                            output_dir: str, speaker_id: Union[int, str], 
                            format_type: str = 'wav') -> bool:
        """Export all segments for a specific speaker.
        
        Args:
            audio_file: Path to the audio file
            segments: List of transcript segments
            output_dir: Directory to save the exported files
            speaker_id: ID of the speaker to export
            format_type: Export format type ('wav', 'mp3', 'flac')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Filter segments for the specified speaker
            speaker_segments = [s for s in segments if s.get('speaker') == speaker_id]
            
            if not speaker_segments:
                logger.warning(f"No segments found for speaker {speaker_id}")
                return False
            
            # Get speaker name
            speaker_name = speaker_segments[0].get('speaker_name', f"Speaker_{speaker_id}")
            
            # Export each segment
            for i, segment in enumerate(speaker_segments):
                # Generate output filename
                filename = f"{speaker_name}_segment_{i+1:03d}.{format_type}"
                output_path = os.path.join(output_dir, filename)
                
                # Export segment
                success = self.export_audio_segment(
                    audio_file, 
                    output_path, 
                    segment['start'], 
                    segment['end'], 
                    format_type
                )
                
                if not success:
                    logger.warning(f"Failed to export segment {i+1} for speaker {speaker_id}")
            
            logger.info(f"Exported {len(speaker_segments)} segments for speaker {speaker_id} to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting speaker audio: {e}")
            return False
    
    def export_word_audio(self, audio_file: str, word: Dict[str, Any], 
                         output_path: str, format_type: str = 'wav',
                         padding_ms: int = 50) -> bool:
        """Export audio for a specific word.
        
        Args:
            audio_file: Path to the audio file
            word: Word data with 'start' and 'end' times
            output_path: Path to save the exported file
            format_type: Export format type ('wav', 'mp3', 'flac')
            padding_ms: Padding in milliseconds to add before and after the word
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ExportError: If there's an error during export
            FileError: If there's an error with file operations
        """
        try:
            # Check if word has required fields
            if 'start' not in word or 'end' not in word:
                raise ExportError(
                    "Word data missing required fields (start, end)",
                    ErrorSeverity.ERROR,
                    {"word": word}
                )
            
            # Calculate start and end times with padding
            padding_sec = padding_ms / 1000.0
            start = max(0, word['start'] - padding_sec)
            end = word['end'] + padding_sec
            
            # Export segment
            return self.export_audio_segment(audio_file, output_path, start, end, format_type)
            
        except Exception as e:
            if isinstance(e, (ExportError, FileError)):
                # Re-raise existing VSAT errors
                raise
            else:
                # Wrap generic errors
                raise ExportError(
                    f"Error exporting word audio: {str(e)}",
                    ErrorSeverity.ERROR,
                    {"audio_file": audio_file, "word": word, "output_path": output_path}
                ) from e
    
    def export_selection(self, audio_file: str, words: List[Dict[str, Any]], 
                        output_path: str, format_type: str = 'wav',
                        include_transcript: bool = True) -> bool:
        """Export audio for a selection of words.
        
        Args:
            audio_file: Path to the audio file
            words: List of word data with 'start' and 'end' times
            output_path: Path to save the exported file
            format_type: Export format type ('wav', 'mp3', 'flac')
            include_transcript: Whether to include a transcript file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not words:
                logger.error("No words provided for export")
                return False
            
            # Get start and end times
            start = min(word['start'] for word in words)
            end = max(word['end'] for word in words)
            
            # Export audio segment
            success = self.export_audio_segment(audio_file, output_path, start, end, format_type)
            
            if success and include_transcript:
                # Create transcript
                transcript = " ".join(word['text'] for word in words)
                
                # Generate transcript filename
                transcript_path = os.path.splitext(output_path)[0] + ".txt"
                
                # Write transcript
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                
                logger.info(f"Exported transcript for selection to {transcript_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error exporting selection: {e}")
            return False 