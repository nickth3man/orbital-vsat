"""
Whisper transcription module for VSAT.

This module provides functionality for transcribing audio using faster-whisper.
"""

import os
import logging
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
from faster_whisper import WhisperModel

from ..transcription.word_aligner import WordAligner
from ..utils.error_handler import VSATError, ErrorSeverity
from ..ml.error_handling import ModelLoadError, InferenceError, ResourceExhaustionError

logger = logging.getLogger(__name__)

# Define common transcription errors and their recovery strategies
TRANSCRIPTION_ERROR_TYPES = {
    "model_load": "Failed to load transcription model",
    "cuda_out_of_memory": "CUDA out of memory",
    "audio_too_short": "Audio too short for transcription",
    "invalid_audio": "Invalid audio format or content",
    "processing_timeout": "Transcription processing timeout",
    "inference_failure": "Model inference failed",
    "unsupported_language": "Unsupported language",
    "audio_quality": "Poor audio quality affecting transcription"
}

class TranscriptionError(VSATError):
    """Error raised when transcription fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, ErrorSeverity.ERROR, details)

class WhisperTranscriber:
    """Class for transcribing audio using faster-whisper."""
    
    # Available model sizes
    MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
    
    def __init__(self, model_size: str = "medium", device: str = "cpu", 
                compute_type: str = "float32", download_root: Optional[str] = None,
                use_word_aligner: bool = True, timeout: int = 300):
        """Initialize the transcriber.
        
        Args:
            model_size: Size of the Whisper model to use
            device: Device to use for inference ("cpu" or "cuda")
            compute_type: Compute type for inference
            download_root: Directory to download models to
            use_word_aligner: Whether to use the word aligner for improved timestamps
            timeout: Maximum time in seconds to allow for transcription
        
        Raises:
            ValueError: If the model size is invalid
            ModelLoadError: If the model fails to load
        """
        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size: {model_size}. Must be one of {self.MODEL_SIZES}")
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.use_word_aligner = use_word_aligner
        self.timeout = timeout
        self.max_retries = 3
        
        # Set default download root if not provided
        if download_root is None:
            download_root = str(Path.home() / '.vsat' / 'models' / 'whisper')
        
        # Create download directory if it doesn't exist
        os.makedirs(download_root, exist_ok=True)
        
        logger.info(f"Initializing Whisper model: {model_size} on {device}")
        
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                # Check if CUDA is available when device is set to cuda
                if self.device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    self.device = "cpu"
                
                # Initialize the model
                self.model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                    download_root=download_root
                )
                
                # Initialize word aligner if enabled
                if use_word_aligner:
                    self.word_aligner = WordAligner(device=device)
                else:
                    self.word_aligner = None
                
                logger.info("Whisper model initialized successfully")
                return
                
            except torch.cuda.OutOfMemoryError as e:
                retries += 1
                last_error = e
                logger.warning(f"CUDA out of memory on attempt {retries}/{self.max_retries}. Falling back to CPU.")
                self.device = "cpu"
                if retries >= self.max_retries:
                    raise ResourceExhaustionError(
                        "CUDA out of memory when initializing Whisper model",
                        "GPU memory",
                        {"model_size": model_size, "device": self.device, "original_error": str(e), 
                         "stack_trace": traceback.format_exc()}
                    )
                    
            except Exception as e:
                retries += 1
                last_error = e
                logger.error(f"Error initializing Whisper model: {e}")
                
                # Try with a smaller model if current one is too large
                if retries >= self.max_retries:
                    break
                    
                # If model may be too large for the device, try a smaller one
                if "memory" in str(e).lower() or "cuda" in str(e).lower():
                    if model_size in ["large-v3", "large-v2", "large-v1"]:
                        self.model_size = "medium"
                        logger.warning(f"Retrying with medium model instead of {model_size}")
                    elif model_size == "medium":
                        self.model_size = "small"
                        logger.warning(f"Retrying with small model instead of {model_size}")
                    elif model_size == "small":
                        self.model_size = "base"
                        logger.warning(f"Retrying with base model instead of {model_size}")
                    
        # If we get here, all retries failed
        if last_error is not None:
            error_context = {
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
                "original_error": str(last_error),
                "stack_trace": traceback.format_exc()
            }
            
            if "download" in str(last_error).lower() or "network" in str(last_error).lower():
                raise ModelLoadError(
                    f"Network error downloading Whisper model: {str(last_error)}",
                    f"whisper-{model_size}",
                    {"error_type": "network", **error_context}
                )
            elif "memory" in str(last_error).lower():
                raise ResourceExhaustionError(
                    f"Not enough memory to load Whisper model: {str(last_error)}",
                    "memory",
                    {"error_type": "resource_exhaustion", **error_context}
                )
            else:
                raise ModelLoadError(
                    f"Failed to initialize Whisper model: {str(last_error)}",
                    f"whisper-{model_size}",
                    {"error_type": "initialization", **error_context}
                )
    
    def transcribe(self, audio: Union[str, np.ndarray], 
                  sample_rate: Optional[int] = None,
                  word_timestamps: bool = True,
                  language: Optional[str] = "en",
                  task: str = "transcribe",
                  refine_word_timestamps: bool = True) -> Dict[str, Any]:
        """Transcribe audio using Whisper.
        
        Args:
            audio: Path to audio file or audio data as numpy array
            sample_rate: Sample rate of the audio (required if audio is numpy array)
            word_timestamps: Whether to include word-level timestamps
            language: Language code (e.g., "en" for English)
            task: Task to perform ("transcribe" or "translate")
            refine_word_timestamps: Whether to refine word timestamps using the aligner
            
        Returns:
            Dict[str, Any]: Transcription results
            
        Raises:
            ValueError: If sample_rate is not provided for numpy array input
            TranscriptionError: If transcription fails
            ResourceExhaustionError: If system resources are exhausted
        """
        start_time = time.time()
        
        try:
            # Check if input is a file path or audio data
            if isinstance(audio, str):
                # Check if file exists
                if not os.path.exists(audio):
                    raise FileNotFoundError(f"Audio file not found: {audio}")
                
                # Log file information
                logger.info(f"Transcribing file: {audio}")
                
                # Check if processing time exceeds timeout
                if time.time() - start_time > self.timeout:
                    raise TranscriptionError(
                        f"Transcription processing timeout after {self.timeout} seconds",
                        {"error_type": "processing_timeout", "timeout": self.timeout}
                    )
                
                # Transcribe file
                segments, info = self.model.transcribe(
                    audio, 
                    word_timestamps=word_timestamps,
                    language=language,
                    task=task
                )
                
            else:
                # Check if audio is numpy array
                if not isinstance(audio, np.ndarray):
                    try:
                        audio = np.array(audio)
                    except:
                        raise ValueError(f"Audio must be a file path, numpy array, or convertible to numpy array, got {type(audio)}")
                
                # Check if sample rate is provided
                if sample_rate is None:
                    raise ValueError("Sample rate must be provided when audio is numpy array")
                
                # Check if audio is empty
                if len(audio) == 0:
                    raise ValueError("Empty audio data provided for transcription")
                
                # Check if audio is too short
                if len(audio) / sample_rate < 0.5:  # Less than 0.5 seconds
                    logger.warning("Audio too short for reliable transcription")
                    return {
                        "text": "",
                        "segments": [],
                        "language": language,
                        "warning": "Audio too short for reliable transcription"
                    }
                
                # Log audio information
                logger.info(f"Transcribing audio array: shape={audio.shape}, sample_rate={sample_rate}")
                
                # Check if processing time exceeds timeout
                if time.time() - start_time > self.timeout:
                    raise TranscriptionError(
                        f"Transcription processing timeout after {self.timeout} seconds",
                        {"error_type": "processing_timeout", "timeout": self.timeout}
                    )
                
                # Ensure audio is 1D
                if audio.ndim == 2 and audio.shape[0] == 1:
                    audio = audio[0]
                elif audio.ndim == 2 and audio.shape[1] == 1:
                    audio = audio[:, 0]
                elif audio.ndim > 2:
                    raise ValueError(f"Audio must be 1D or 2D array, got {audio.ndim}D")
                
                # Check if audio is mono
                if audio.ndim == 2 and audio.shape[0] > 1 and audio.shape[1] > 1:
                    logger.warning("Multi-channel audio detected. Using first channel only.")
                    audio = audio[0]
                
                # Normalize audio if needed
                if np.abs(audio).max() > 1.0:
                    logger.info("Normalizing audio")
                    audio = audio / np.abs(audio).max()
                
                # Transcribe audio
                segments, info = self.model.transcribe(
                    audio, 
                    word_timestamps=word_timestamps,
                    language=language,
                    task=task,
                    sample_rate=sample_rate
                )
            
            # Convert results to our format
            result = self._convert_transcription(segments, info)
            
            # Refine word timestamps if requested
            if refine_word_timestamps and word_timestamps and self.word_aligner is not None:
                try:
                    self._refine_word_timestamps(result, audio, sample_rate)
                except Exception as e:
                    logger.warning(f"Failed to refine word timestamps: {e}")
                    # Add a warning to the result but don't fail the transcription
                    result["warning"] = f"Word timestamp refinement failed: {str(e)}"
            
            logger.info(f"Transcription completed with {len(result.get('segments', []))} segments")
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory during transcription: {str(e)}")
            raise ResourceExhaustionError(
                "CUDA out of memory during transcription. Try using a CPU device or reducing audio length.",
                "GPU memory",
                {"device": self.device, "model_size": self.model_size, 
                 "original_error": str(e), "stack_trace": traceback.format_exc()}
            )
            
        except (FileNotFoundError, ValueError) as e:
            # Re-raise these as they are input validation errors
            raise
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            error_type = type(e).__name__
            error_detail = str(e).lower()
            
            error_context = {
                "model_size": self.model_size,
                "device": self.device,
                "language": language,
                "task": task,
                "word_timestamps": word_timestamps,
                "error_type": error_type,
                "original_error": str(e),
                "stack_trace": traceback.format_exc(),
                "processing_time": time.time() - start_time
            }
            
            # Add audio info if available
            if isinstance(audio, np.ndarray) and sample_rate is not None:
                error_context["audio_shape"] = audio.shape
                error_context["sample_rate"] = sample_rate
                error_context["audio_duration"] = len(audio) / sample_rate
            elif isinstance(audio, str):
                error_context["audio_file"] = audio
            
            # Handle specific error types
            if "memory" in error_detail:
                error_message = "Out of memory during transcription. Try reducing audio length or using a smaller model."
                error_context["error_type"] = "resource_exhaustion"
                raise ResourceExhaustionError(
                    error_message, 
                    "memory", 
                    error_context
                )
            elif "cuda" in error_detail:
                error_message = "CUDA error during transcription. Try falling back to CPU."
                error_context["error_type"] = "gpu_error"
                error_context["suggestion"] = "Use CPU device instead"
                raise TranscriptionError(error_message, error_context)
            elif time.time() - start_time > self.timeout:
                error_message = f"Transcription processing timeout after {self.timeout} seconds"
                error_context["error_type"] = "processing_timeout"
                raise TranscriptionError(error_message, error_context)
            elif "language" in error_detail:
                error_message = f"Unsupported language for transcription: {language}"
                error_context["error_type"] = "unsupported_language"
                error_context["suggestion"] = "Try using 'en' (English) or check documentation for supported languages"
                raise TranscriptionError(error_message, error_context)
            else:
                error_message = f"Failed to perform transcription: {str(e)}"
                raise TranscriptionError(error_message, error_context)
    
    def _convert_transcription(self, segments, info) -> Dict[str, Any]:
        """Convert faster-whisper transcription result to our format.
        
        Args:
            segments: Segments from faster-whisper
            info: Information from faster-whisper
            
        Returns:
            Dict[str, Any]: Transcription result in our format
        """
        result = {
            "text": "",
            "segments": [],
            "language": info.language,
            "language_probability": info.language_probability
        }
        
        # Process each segment
        all_text = []
        for segment in segments:
            # Add segment text to overall text
            all_text.append(segment.text)
            
            # Convert segment to our format
            segment_dict = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            }
            
            # Add words if available
            if hasattr(segment, "words") and segment.words:
                segment_dict["words"] = []
                for word in segment.words:
                    segment_dict["words"].append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    })
            
            # Add segment to result
            result["segments"].append(segment_dict)
        
        # Set overall text
        result["text"] = " ".join(all_text)
        
        return result
    
    def _refine_word_timestamps(self, transcription: Dict[str, Any], 
                              audio: Union[str, np.ndarray],
                              sample_rate: Optional[int] = None) -> None:
        """Refine word timestamps using the word aligner.
        
        Args:
            transcription: Transcription result
            audio: Audio data or file path
            sample_rate: Sample rate of the audio (required if audio is numpy array)
        """
        if self.word_aligner is None:
            logger.warning("Word aligner not available, skipping timestamp refinement")
            return
        
        # Log refinement start
        logger.info("Refining word timestamps")
        
        try:
            # Process each segment
            for segment in transcription.get("segments", []):
                # Skip if no words
                if "words" not in segment or not segment["words"]:
                    continue
                
                # Get segment text and time range
                text = segment["text"]
                start_time = segment["start"]
                end_time = segment["end"]
                
                # Align words for this segment
                refined_words = self.word_aligner.align_words(
                    audio=audio,
                    sample_rate=sample_rate,
                    text=text,
                    segment_start=start_time,
                    segment_end=end_time
                )
                
                # Update words if alignment was successful
                if refined_words:
                    segment["words"] = refined_words
        
        except Exception as e:
            logger.error(f"Error refining word timestamps: {e}")
            raise
    
    def transcribe_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio file.
        
        Args:
            file_path: Path to audio file
            **kwargs: Additional parameters for transcription
            
        Returns:
            Dict[str, Any]: Transcription results
            
        Raises:
            FileNotFoundError: If the audio file does not exist
            TranscriptionError: If transcription fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        return self.transcribe(file_path, **kwargs)
    
    def transcribe_segment(self, audio_data: np.ndarray, sample_rate: int, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio segment.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            **kwargs: Additional parameters for transcription
            
        Returns:
            Dict[str, Any]: Transcription results
            
        Raises:
            ValueError: If audio_data is invalid
            TranscriptionError: If transcription fails
        """
        # Validate audio data
        if not isinstance(audio_data, np.ndarray):
            raise ValueError(f"Audio data must be a numpy array, got {type(audio_data)}")
        
        return self.transcribe(audio_data, sample_rate=sample_rate, **kwargs)
    
    def align_transcript(self, audio_data: np.ndarray, sample_rate: int, 
                        transcript: str, segment_start: float = 0.0,
                        segment_end: Optional[float] = None) -> List[Dict[str, Any]]:
        """Align a transcript with audio to get precise word timestamps.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            transcript: Transcript text to align
            segment_start: Start time of the segment in seconds
            segment_end: End time of the segment in seconds
            
        Returns:
            List[Dict[str, Any]]: List of words with timestamps
            
        Raises:
            ValueError: If audio_data or transcript is invalid
        """
        if self.word_aligner is None:
            raise ValueError("Word aligner not available")
        
        # Validate audio data
        if not isinstance(audio_data, np.ndarray):
            raise ValueError(f"Audio data must be a numpy array, got {type(audio_data)}")
        
        # Validate transcript
        if not transcript or not isinstance(transcript, str):
            raise ValueError(f"Transcript must be a non-empty string, got {type(transcript)}")
        
        return self.word_aligner.align_words(
            audio=audio_data,
            sample_rate=sample_rate,
            text=transcript,
            segment_start=segment_start,
            segment_end=segment_end
        ) 