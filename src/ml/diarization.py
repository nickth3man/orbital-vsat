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
from pyannote.core import Segment as PyannoteSegment, Annotation
import soundfile as sf
import tempfile
import traceback
import time
from dataclasses import dataclass, field

from ..utils.error_handler import VSATError, ErrorSeverity
from .error_handling import DiarizationError, ResourceExhaustionError

logger = logging.getLogger(__name__)

# Define common diarization errors and their recovery strategies
DIARIZATION_ERROR_TYPES = {
    "model_load": "Failed to load diarization model",
    "cuda_out_of_memory": "CUDA out of memory",
    "audio_too_short": "Audio too short for diarization",
    "invalid_audio": "Invalid audio format or content",
    "processing_timeout": "Diarization processing timeout",
    "inference_failure": "Model inference failed",
    "file_access": "Failed to access audio file",
}

@dataclass
class Speaker:
    """Class representing a speaker in a diarization result."""
    id: str
    name: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Segment:
    """Class representing a segment in a diarization result."""
    start: float
    end: float
    speaker: str
    text: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiarizationConfig:
    """Configuration for diarization processing."""
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    min_segment_duration: float = 0.5
    collar: float = 0.15  # Collar for merging segments
    device: str = "cpu"
    auth_token: Optional[str] = None
    download_root: Optional[str] = None
    timeout: int = 300

@dataclass
class DiarizationResult:
    """Class representing the result of a diarization process."""
    segments: List[Segment] = field(default_factory=list)
    speakers: List[Speaker] = field(default_factory=list)
    audio_duration: float = 0.0
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": segment.speaker,
                    "text": segment.text,
                    "confidence": segment.confidence,
                    "metadata": segment.metadata
                }
                for segment in self.segments
            ],
            "speakers": [
                {
                    "id": speaker.id,
                    "name": speaker.name,
                    "confidence": speaker.confidence,
                    "metadata": speaker.metadata
                }
                for speaker in self.speakers
            ],
            "audio_duration": self.audio_duration,
            "language": self.language,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiarizationResult':
        """Create a DiarizationResult from a dictionary."""
        segments = [
            Segment(
                start=segment["start"],
                end=segment["end"],
                speaker=segment["speaker"],
                text=segment.get("text"),
                confidence=segment.get("confidence", 1.0),
                metadata=segment.get("metadata", {})
            )
            for segment in data.get("segments", [])
        ]
        
        speakers = [
            Speaker(
                id=speaker["id"],
                name=speaker.get("name"),
                confidence=speaker.get("confidence", 1.0),
                metadata=speaker.get("metadata", {})
            )
            for speaker in data.get("speakers", [])
        ]
        
        return cls(
            segments=segments,
            speakers=speakers,
            audio_duration=data.get("audio_duration", 0.0),
            language=data.get("language"),
            metadata=data.get("metadata", {})
        )

class Diarizer:
    """Class for speaker diarization using pyannote.audio."""
    
    def __init__(self, auth_token: Optional[str] = None, device: str = "cpu",
                download_root: Optional[str] = None, timeout: int = 300):
        """Initialize the diarizer.
        
        Args:
            auth_token: HuggingFace authentication token
            device: Device to use for inference ("cpu" or "cuda")
            download_root: Directory to download models to
            timeout: Maximum time in seconds to allow for diarization
        """
        self.device = device
        self.auth_token = auth_token
        self.download_root = download_root
        self.pipeline = None
        self.timeout = timeout
        self.max_retries = 3
        
        # Initialize the pipeline
        self._initialize_pipeline()
        
        logger.info(f"Diarizer initialized with device={device}")
    
    def _initialize_pipeline(self):
        """Initialize the diarization pipeline."""
        retries = 0
        while retries < self.max_retries:
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
                return
                
            except torch.cuda.OutOfMemoryError as e:
                retries += 1
                logger.warning(f"CUDA out of memory on attempt {retries}/{self.max_retries}. Falling back to CPU.")
                self.device = "cpu"
                if retries >= self.max_retries:
                    raise ResourceExhaustionError(
                        "CUDA out of memory when initializing diarization pipeline",
                        "GPU memory",
                        {"device": self.device, "original_error": str(e), "stack_trace": traceback.format_exc()}
                    )
                
            except Exception as e:
                logger.error(f"Failed to initialize diarization pipeline: {str(e)}")
                error_type = type(e).__name__
                error_context = {
                    "device": self.device,
                    "error_type": error_type,
                    "original_error": str(e),
                    "stack_trace": traceback.format_exc()
                }
                
                # Handle specific error cases
                if "auth" in str(e).lower() or "token" in str(e).lower():
                    error_message = f"Authentication failed for diarization model: {str(e)}"
                    error_context["suggestion"] = "Check your HuggingFace authentication token"
                    raise DiarizationError(
                        error_message, 
                        {"error_type": "authentication", **error_context}
                    )
                elif "download" in str(e).lower() or "network" in str(e).lower():
                    error_message = f"Network error while downloading diarization model: {str(e)}"
                    error_context["suggestion"] = "Check your internet connection or download models manually"
                    raise DiarizationError(
                        error_message, 
                        {"error_type": "network", **error_context}
                    )
                else:
                    error_message = f"Failed to initialize diarization pipeline: {str(e)}"
                    raise DiarizationError(
                        error_message, 
                        {"error_type": "initialization", **error_context}
                    )
    
    def diarize(self, audio_data: np.ndarray, sample_rate: int, 
                min_speakers: Optional[int] = None, 
                max_speakers: Optional[int] = None) -> DiarizationResult:
        """Perform speaker diarization on audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            
        Returns:
            DiarizationResult: Diarization result
            
        Raises:
            DiarizationError: If diarization fails
            ResourceExhaustionError: If system resources are exhausted
        """
        # Check for empty or invalid audio
        if len(audio_data) == 0:
            raise DiarizationError(
                "Empty audio data provided for diarization",
                {"error_type": "invalid_audio", "audio_shape": audio_data.shape, "sample_rate": sample_rate}
            )
        
        # Check if audio is too short
        if len(audio_data) / sample_rate < 0.5:  # Less than 0.5 seconds
            logger.warning("Audio too short for reliable diarization")
            # Return a simple result with a single speaker
            return DiarizationResult(
                segments=[Segment(start=0, end=len(audio_data) / sample_rate, speaker="SPEAKER_0")],
                speakers=[Speaker(id="SPEAKER_0")],
                audio_duration=len(audio_data) / sample_rate
            )
        
        temp_path = None
        start_time = time.time()
        
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Write audio to temporary file
            sf.write(temp_path, audio_data, sample_rate)
            
            # Check if processing time exceeds timeout
            if time.time() - start_time > self.timeout:
                raise DiarizationError(
                    f"Diarization processing timeout after {self.timeout} seconds",
                    {"error_type": "processing_timeout", "timeout": self.timeout}
                )
            
            # Prepare parameters
            params = {}
            if min_speakers is not None:
                params["min_speakers"] = min_speakers
            if max_speakers is not None:
                params["max_speakers"] = max_speakers
            
            # Perform diarization
            logger.info("Performing diarization")
            diarization = self.pipeline(temp_path, **params)
            
            # Convert to our format
            result = self._convert_diarization(diarization)
            
            logger.info(f"Diarization completed with {len(result.segments)} segments")
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory during diarization: {str(e)}")
            raise ResourceExhaustionError(
                "CUDA out of memory during diarization. Try using a CPU device or reducing audio length.",
                "GPU memory",
                {"device": self.device, "audio_duration": len(audio_data) / sample_rate, 
                 "original_error": str(e), "stack_trace": traceback.format_exc()}
            )
            
        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            error_type = type(e).__name__
            error_detail = str(e).lower()
            error_context = {
                "sample_rate": sample_rate, 
                "audio_shape": audio_data.shape,
                "error_type": error_type,
                "original_error": str(e),
                "stack_trace": traceback.format_exc(),
                "processing_time": time.time() - start_time
            }
            
            # Handle specific error types
            if "memory" in error_detail:
                error_message = "Out of memory during diarization. Try reducing audio length or using CPU."
                error_context["error_type"] = "resource_exhaustion"
                raise ResourceExhaustionError(
                    error_message, 
                    "memory", 
                    error_context
                )
            elif "cuda" in error_detail:
                error_message = "CUDA error during diarization. Try falling back to CPU."
                error_context["error_type"] = "gpu_error"
                error_context["suggestion"] = "Use CPU device instead"
                raise DiarizationError(error_message, error_context)
            elif time.time() - start_time > self.timeout:
                error_message = f"Diarization processing timeout after {self.timeout} seconds"
                error_context["error_type"] = "processing_timeout"
                raise DiarizationError(error_message, error_context)
            else:
                error_message = f"Failed to perform diarization: {str(e)}"
                raise DiarizationError(error_message, error_context)
                
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_path}: {str(e)}")

    def _convert_diarization(self, diarization: Annotation) -> DiarizationResult:
        """Convert pyannote.audio diarization result to our format.
        
        Args:
            diarization: Diarization result from pyannote.audio
            
        Returns:
            DiarizationResult: Diarization result in our format
        """
        segments = []
        speakers = set()
        
        # Process each segment
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            # Add segment
            segments.append(
                Segment(
                    start=segment.start,
                    end=segment.end,
                    speaker=speaker
                )
            )
            
            # Add speaker to set
            speakers.add(speaker)
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.start)
        
        return DiarizationResult(
            segments=segments,
            speakers=[Speaker(id=speaker) for speaker in speakers]
        )
    
    def merge_with_transcription(self, diarization: DiarizationResult, 
                               transcription: Dict[str, Any]) -> DiarizationResult:
        """Merge diarization and transcription results.
        
        Args:
            diarization: Diarization result
            transcription: Transcription result
            
        Returns:
            DiarizationResult: Merged result
        """
        # Create a copy of the diarization result
        result = DiarizationResult(
            segments=[Segment(**segment) for segment in diarization.segments],
            speakers=[Speaker(**speaker) for speaker in diarization.speakers],
            audio_duration=diarization.audio_duration,
            language=diarization.language,
            metadata=diarization.metadata
        )
        
        # Get transcription segments
        trans_segments = transcription.get("segments", [])
        
        # Assign transcription to diarization segments
        for segment in result.segments:
            # Find overlapping transcription segments
            segment_text = []
            
            for trans in trans_segments:
                # Check if transcription segment overlaps with diarization segment
                if (trans["start"] < segment.end and 
                    trans["end"] > segment.start):
                    # Calculate overlap
                    overlap_start = max(segment.start, trans["start"])
                    overlap_end = min(segment.end, trans["end"])
                    overlap_duration = overlap_end - overlap_start
                    
                    # Calculate portion of transcription segment that overlaps
                    trans_duration = trans["end"] - trans["start"]
                    overlap_ratio = overlap_duration / trans_duration
                    
                    # If significant overlap, add text
                    if overlap_ratio > 0.5:
                        segment_text.append(trans["text"])
            
            # Join text and add to segment
            segment.text = " ".join(segment_text)
        
        # Add transcription to result
        result.metadata["transcription"] = {
            "text": transcription.get("text", ""),
            "language": transcription.get("language", "")
        }
        
        return result
    
    def diarize_file(self, file_path: str, **kwargs) -> DiarizationResult:
        """Perform speaker diarization on an audio file.
        
        Args:
            file_path: Path to audio file
            **kwargs: Additional parameters for diarization
            
        Returns:
            DiarizationResult: Diarization result
            
        Raises:
            DiarizationError: If diarization fails
            FileNotFoundError: If the audio file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            return self.pipeline(file_path, **kwargs)
        except Exception as e:
            error_context = {
                "file_path": file_path,
                "error_type": type(e).__name__,
                "original_error": str(e),
                "stack_trace": traceback.format_exc()
            }
            raise DiarizationError(
                f"Failed to diarize file {os.path.basename(file_path)}: {str(e)}",
                error_context
            )
    
    def diarize_segment(self, audio_data: np.ndarray, sample_rate: int, **kwargs) -> DiarizationResult:
        """Perform speaker diarization on an audio segment.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            **kwargs: Additional parameters for diarization
            
        Returns:
            DiarizationResult: Diarization result
            
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