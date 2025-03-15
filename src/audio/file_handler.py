"""
Audio file handling module for VSAT.

This module provides functionality for loading, saving, and basic processing of audio files.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import librosa
import soundfile as sf

from src.utils.error_handler import AudioError, FileError, ErrorSeverity

logger = logging.getLogger(__name__)

class AudioFileHandler:
    """Class for handling audio file operations."""
    
    SUPPORTED_FORMATS = {
        '.wav': 'WAV',
        '.mp3': 'MP3',
        '.flac': 'FLAC'
    }
    
    def __init__(self):
        """Initialize the audio file handler."""
        logger.debug("Initializing AudioFileHandler")
    
    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        """Check if the file format is supported.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            bool: True if the format is supported, False otherwise
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in AudioFileHandler.SUPPORTED_FORMATS
    
    @staticmethod
    def load_audio(file_path: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Load an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple containing:
                np.ndarray: Audio data as a numpy array
                int: Sample rate
                Dict[str, Any]: Metadata
                
        Raises:
            FileError: If the file does not exist or cannot be accessed
            AudioError: If the file format is not supported or there's an error processing the audio
        """
        if not os.path.exists(file_path):
            raise FileError(
                f"File not found: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path}
            )
        
        if not AudioFileHandler.is_supported_format(file_path):
            raise AudioError(
                f"Unsupported file format: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path, "extension": os.path.splitext(file_path)[1].lower()}
            )
        
        logger.info(f"Loading audio file: {file_path}")
        
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
            
            # Extract metadata
            metadata = {
                'filename': os.path.basename(file_path),
                'path': file_path,
                'format': os.path.splitext(file_path)[1].lower()[1:],
                'channels': 1 if audio_data.ndim == 1 else audio_data.shape[0],
                'duration': librosa.get_duration(y=audio_data, sr=sample_rate),
                'sample_rate': sample_rate
            }
            
            logger.debug(f"Audio loaded successfully: {metadata}")
            
            return audio_data, sample_rate, metadata
            
        except PermissionError as e:
            raise FileError(
                f"Permission denied when accessing file: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path, "error": str(e)}
            ) from e
        except OSError as e:
            raise FileError(
                f"OS error when accessing file: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path, "error": str(e)}
            ) from e
        except Exception as e:
            raise AudioError(
                f"Error loading audio file: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path, "error": str(e)}
            ) from e
    
    @staticmethod
    def save_audio(file_path: str, audio_data: np.ndarray, sample_rate: int, 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save audio data to a file.
        
        Args:
            file_path: Path to save the audio file
            audio_data: Audio data as a numpy array
            sample_rate: Sample rate
            metadata: Optional metadata to include
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            FileError: If there's an error writing to the file
            AudioError: If there's an error processing the audio data
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Validate audio data
            if audio_data is None or len(audio_data) == 0:
                raise AudioError(
                    "No audio data to save",
                    ErrorSeverity.ERROR,
                    {"file_path": file_path}
                )
            
            # Check if format is supported for writing
            _, extension = os.path.splitext(file_path.lower())
            if extension not in AudioFileHandler.SUPPORTED_FORMATS:
                raise AudioError(
                    f"Unsupported output format: {extension}",
                    ErrorSeverity.ERROR,
                    {"file_path": file_path, "extension": extension}
                )
            
            # Save audio file
            sf.write(file_path, audio_data, sample_rate)
            
            logger.info(f"Audio saved successfully: {file_path}")
            return True
            
        except PermissionError as e:
            raise FileError(
                f"Permission denied when writing to file: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path, "error": str(e)}
            ) from e
        except OSError as e:
            raise FileError(
                f"OS error when writing to file: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path, "error": str(e)}
            ) from e
        except Exception as e:
            raise AudioError(
                f"Error saving audio file: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path, "error": str(e)}
            ) from e
    
    @staticmethod
    def get_audio_info(file_path: str) -> Dict[str, Any]:
        """Get information about an audio file without loading the entire file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict[str, Any]: Audio file information
            
        Raises:
            FileError: If the file does not exist or cannot be accessed
            AudioError: If there's an error processing the audio file
        """
        if not os.path.exists(file_path):
            raise FileError(
                f"File not found: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path}
            )
        
        try:
            info = {}
            
            # Get basic file info
            info['filename'] = os.path.basename(file_path)
            info['path'] = file_path
            info['format'] = os.path.splitext(file_path)[1].lower()[1:]
            info['size'] = os.path.getsize(file_path)
            
            # Get audio-specific info
            with sf.SoundFile(file_path) as f:
                info['channels'] = f.channels
                info['sample_rate'] = f.samplerate
                info['duration'] = f.frames / f.samplerate
                info['frames'] = f.frames
            
            return info
            
        except PermissionError as e:
            raise FileError(
                f"Permission denied when accessing file: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path, "error": str(e)}
            ) from e
        except OSError as e:
            raise FileError(
                f"OS error when accessing file: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path, "error": str(e)}
            ) from e
        except Exception as e:
            raise AudioError(
                f"Error getting audio info: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path, "error": str(e)}
            ) from e
            
    @staticmethod
    def split_audio(file_path: str, output_dir: str, segment_length: float, 
                   overlap: float = 0.0, format_type: str = 'wav') -> List[str]:
        """Split an audio file into segments of specified length.
        
        Args:
            file_path: Path to the audio file
            output_dir: Directory to save the segments
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments in seconds
            format_type: Format of the output files
            
        Returns:
            List[str]: List of paths to the saved segments
            
        Raises:
            FileError: If there's an error with file operations
            AudioError: If there's an error processing the audio
        """
        try:
            # Check output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load audio file
            audio_data, sample_rate, metadata = AudioFileHandler.load_audio(file_path)
            
            # Convert stereo to mono if needed
            if audio_data.ndim > 1 and audio_data.shape[0] > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            # Calculate segment parameters
            segment_samples = int(segment_length * sample_rate)
            overlap_samples = int(overlap * sample_rate)
            step_size = segment_samples - overlap_samples
            
            if segment_samples <= 0:
                raise AudioError(
                    "Invalid segment length",
                    ErrorSeverity.ERROR,
                    {"segment_length": segment_length, "sample_rate": sample_rate}
                )
                
            if step_size <= 0:
                raise AudioError(
                    "Overlap is greater than or equal to segment length",
                    ErrorSeverity.ERROR,
                    {"segment_length": segment_length, "overlap": overlap}
                )
            
            # Generate segments
            segment_paths = []
            for i, start_sample in enumerate(range(0, len(audio_data), step_size)):
                end_sample = min(start_sample + segment_samples, len(audio_data))
                if end_sample - start_sample < segment_samples / 2:  # Skip too short segments
                    continue
                    
                segment_data = audio_data[start_sample:end_sample]
                
                # Generate output filename
                base_filename = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_dir, f"{base_filename}_segment_{i+1:03d}.{format_type}")
                
                # Save segment
                AudioFileHandler.save_audio(output_path, segment_data, sample_rate)
                segment_paths.append(output_path)
            
            if not segment_paths:
                logger.warning(f"No segments created from {file_path}")
            else:
                logger.info(f"Created {len(segment_paths)} segments from {file_path}")
                
            return segment_paths
            
        except (FileError, AudioError):
            # Re-raise specific errors
            raise
        except Exception as e:
            raise AudioError(
                f"Error splitting audio file: {file_path}",
                ErrorSeverity.ERROR,
                {"file_path": file_path, "output_dir": output_dir, "error": str(e)}
            ) from e 