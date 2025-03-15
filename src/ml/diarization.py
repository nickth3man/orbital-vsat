"""
Speaker diarization module for VSAT.

This module provides functionality for speaker diarization using
the pyannote.audio library.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
import soundfile as sf
import tempfile

from src.utils.error_handler import DiarizationError, ErrorSeverity

logger = logging.getLogger(__name__)

class Diarizer:
    """Class for speaker diarization using pyannote.audio."""
    
    def __init__(self, auth_token: Optional[str] = None, device: str = "cpu",
                download_root: Optional[str] = None):
        """Initialize the diarizer.
        
        Args:
            auth_token: HuggingFace authentication token
            device: Device to use for inference ("cpu" or "cuda")
            download_root: Directory to download models to
        """
        self.device = device
        self.auth_token = auth_token
        self.download_root = download_root
        self.pipeline = None
        
        # Initialize the pipeline
        self._initialize_pipeline()
        
        logger.info(f"Diarizer initialized with device={device}")
    
    def _initialize_pipeline(self):
        """Initialize the diarization pipeline."""
        try:
            # Check if CUDA is available when device is set to cuda
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Initialize the pipeline
            logger.info("Initializing diarization pipeline")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token,
                cache_dir=self.download_root
            )
            
            # Move to specified device
            self.pipeline.to(torch.device(self.device))
            
            logger.info("Diarization pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {str(e)}")
            raise DiarizationError(
                f"Failed to initialize diarization pipeline: {str(e)}",
                ErrorSeverity.CRITICAL,
                {"device": self.device}
            )
    
    def diarize(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Perform speaker diarization on audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dict[str, Any]: Diarization result
            
        Raises:
            DiarizationError: If diarization fails
        """
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Write audio to temporary file
            sf.write(temp_path, audio_data, sample_rate)
            
            # Perform diarization
            logger.info("Performing diarization")
            diarization = self.pipeline(temp_path)
            
            # Convert to our format
            result = self._convert_diarization(diarization)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            logger.info(f"Diarization completed with {len(result['segments'])} segments")
            return result
            
        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            raise DiarizationError(
                f"Failed to perform diarization: {str(e)}",
                ErrorSeverity.ERROR,
                {"sample_rate": sample_rate, "audio_shape": audio_data.shape}
            )
    
    def _convert_diarization(self, diarization: Annotation) -> Dict[str, Any]:
        """Convert pyannote.audio diarization result to our format.
        
        Args:
            diarization: Diarization result from pyannote.audio
            
        Returns:
            Dict[str, Any]: Diarization result in our format
        """
        segments = []
        speakers = set()
        
        # Process each segment
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            # Add segment
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })
            
            # Add speaker to set
            speakers.add(speaker)
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])
        
        return {
            "segments": segments,
            "speakers": list(speakers)
        }
    
    def merge_with_transcription(self, diarization: Dict[str, Any], 
                               transcription: Dict[str, Any]) -> Dict[str, Any]:
        """Merge diarization and transcription results.
        
        Args:
            diarization: Diarization result
            transcription: Transcription result
            
        Returns:
            Dict[str, Any]: Merged result
        """
        # Create a copy of the diarization result
        result = diarization.copy()
        
        # Get transcription segments
        trans_segments = transcription.get("segments", [])
        
        # Assign transcription to diarization segments
        for segment in result["segments"]:
            # Find overlapping transcription segments
            segment_text = []
            
            for trans in trans_segments:
                # Check if transcription segment overlaps with diarization segment
                if (trans["start"] < segment["end"] and 
                    trans["end"] > segment["start"]):
                    # Calculate overlap
                    overlap_start = max(segment["start"], trans["start"])
                    overlap_end = min(segment["end"], trans["end"])
                    overlap_duration = overlap_end - overlap_start
                    
                    # Calculate portion of transcription segment that overlaps
                    trans_duration = trans["end"] - trans["start"]
                    overlap_ratio = overlap_duration / trans_duration
                    
                    # If significant overlap, add text
                    if overlap_ratio > 0.5:
                        segment_text.append(trans["text"])
            
            # Join text and add to segment
            segment["text"] = " ".join(segment_text)
        
        # Add transcription to result
        result["transcription"] = {
            "text": transcription.get("text", ""),
            "language": transcription.get("language", "")
        }
        
        return result
    
    def diarize_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Perform speaker diarization on an audio file.
        
        Args:
            file_path: Path to audio file
            **kwargs: Additional parameters for diarization
            
        Returns:
            Dictionary containing diarization results
            
        Raises:
            DiarizationError: If diarization fails
            FileNotFoundError: If the audio file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        return self.diarize(file_path, **kwargs)
    
    def diarize_segment(self, audio_data: np.ndarray, sample_rate: int, **kwargs) -> Dict[str, Any]:
        """Perform speaker diarization on an audio segment.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            **kwargs: Additional parameters for diarization
            
        Returns:
            Dictionary containing diarization results
            
        Raises:
            DiarizationError: If diarization fails
            ValueError: If audio_data is invalid
        """
        if audio_data.ndim != 1 and not (audio_data.ndim == 2 and audio_data.shape[0] == 1):
            raise ValueError("Audio data must be mono (1D array or 2D array with shape [1, samples])")
        
        # Ensure audio is 1D
        if audio_data.ndim == 2:
            audio_data = audio_data[0]
        
        return self.diarize(audio_data, sample_rate=sample_rate, **kwargs) 