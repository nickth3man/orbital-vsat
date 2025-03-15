"""
Whisper transcription module for VSAT.

This module provides functionality for transcribing audio using faster-whisper.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from faster_whisper import WhisperModel

from src.transcription.word_aligner import WordAligner

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Class for transcribing audio using faster-whisper."""
    
    # Available model sizes
    MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
    
    def __init__(self, model_size: str = "medium", device: str = "cpu", 
                compute_type: str = "float32", download_root: Optional[str] = None,
                use_word_aligner: bool = True):
        """Initialize the transcriber.
        
        Args:
            model_size: Size of the Whisper model to use
            device: Device to use for inference ("cpu" or "cuda")
            compute_type: Compute type for inference
            download_root: Directory to download models to
            use_word_aligner: Whether to use the word aligner for improved timestamps
        
        Raises:
            ValueError: If the model size is invalid
        """
        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size: {model_size}. Must be one of {self.MODEL_SIZES}")
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.use_word_aligner = use_word_aligner
        
        # Set default download root if not provided
        if download_root is None:
            download_root = str(Path.home() / '.vsat' / 'models' / 'whisper')
        
        # Create download directory if it doesn't exist
        os.makedirs(download_root, exist_ok=True)
        
        logger.info(f"Initializing Whisper model: {model_size} on {device}")
        
        try:
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
            
        except Exception as e:
            logger.error(f"Error initializing Whisper model: {e}")
            raise
    
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
        """
        try:
            # Check if audio is a file path or numpy array
            if isinstance(audio, str):
                audio_path = audio
                audio_data = None
            else:
                if sample_rate is None:
                    raise ValueError("Sample rate must be provided when audio is a numpy array")
                audio_path = None
                audio_data = audio
            
            logger.info(f"Transcribing {'file' if audio_path else 'data'} with Whisper")
            
            # Set transcription parameters
            beam_size = 5
            
            # Perform transcription
            if audio_path:
                segments, info = self.model.transcribe(
                    audio_path,
                    beam_size=beam_size,
                    word_timestamps=word_timestamps,
                    language=language,
                    task=task
                )
                
                # Load audio data for timestamp refinement if needed
                if refine_word_timestamps and self.word_aligner and word_timestamps:
                    import soundfile as sf
                    audio_data, sample_rate = sf.read(audio_path)
            else:
                segments, info = self.model.transcribe(
                    audio_data,
                    beam_size=beam_size,
                    word_timestamps=word_timestamps,
                    language=language,
                    task=task,
                    sr=sample_rate
                )
            
            # Process the results
            result = {
                'segments': [],
                'language': info.language,
                'language_probability': info.language_probability
            }
            
            # Process segments
            for segment in segments:
                seg_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'words': []
                }
                
                # Process words if available
                if word_timestamps and segment.words:
                    for word in segment.words:
                        word_data = {
                            'start': word.start,
                            'end': word.end,
                            'text': word.word,
                            'probability': word.probability
                        }
                        seg_data['words'].append(word_data)
                    
                    # Refine word timestamps if enabled
                    if refine_word_timestamps and self.word_aligner and audio_data is not None:
                        refined_words = self.word_aligner.refine_word_timestamps(
                            audio_data, sample_rate, seg_data['words'], segment_start=segment.start
                        )
                        seg_data['words'] = refined_words
                
                result['segments'].append(seg_data)
            
            logger.info(f"Transcription completed: {len(result['segments'])} segments")
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    def transcribe_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio file.
        
        Args:
            file_path: Path to the audio file
            **kwargs: Additional arguments to pass to transcribe()
            
        Returns:
            Dict[str, Any]: Transcription results
        """
        return self.transcribe(file_path, **kwargs)
    
    def transcribe_segment(self, audio_data: np.ndarray, sample_rate: int, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio segment.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            **kwargs: Additional arguments to pass to transcribe()
            
        Returns:
            Dict[str, Any]: Transcription results
        """
        return self.transcribe(audio_data, sample_rate=sample_rate, **kwargs)
    
    def align_transcript(self, audio_data: np.ndarray, sample_rate: int, 
                        transcript: str, segment_start: float = 0.0,
                        segment_end: Optional[float] = None) -> List[Dict[str, Any]]:
        """Align a transcript with audio using forced alignment.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            transcript: Text transcript to align
            segment_start: Start time of the segment in the original audio
            segment_end: End time of the segment in the original audio
            
        Returns:
            List[Dict[str, Any]]: Word dictionaries with timestamps
        """
        if not self.word_aligner:
            logger.warning("Word aligner is not enabled, initializing it now")
            self.word_aligner = WordAligner(device=self.device)
        
        return self.word_aligner.align_transcript_with_audio(
            audio_data, sample_rate, transcript, segment_start, segment_end
        ) 