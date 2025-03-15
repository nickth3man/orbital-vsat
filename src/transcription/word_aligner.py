"""
Word alignment module for VSAT.

This module provides functionality for improving word-level timestamp accuracy
through forced alignment techniques.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from src.transcription.boundary_refiner import BoundaryRefiner

logger = logging.getLogger(__name__)

class WordAligner:
    """Class for improving word-level timestamp accuracy through forced alignment."""
    
    def __init__(self, device: str = "cpu", download_root: Optional[str] = None):
        """Initialize the word aligner.
        
        Args:
            device: Device to use for inference ("cpu" or "cuda")
            download_root: Directory to download models to
        """
        self.device = device
        
        # Set default download root if not provided
        if download_root is None:
            download_root = str(Path.home() / '.vsat' / 'models' / 'aligner')
        
        # Create download directory if it doesn't exist
        os.makedirs(download_root, exist_ok=True)
        
        logger.info(f"Initializing word aligner on {device}")
        
        try:
            # Import here to avoid dependency issues if not needed
            import torch
            from pyannote.core import Segment
            
            # Initialize the aligner model
            # For now, we'll use a simple energy-based approach
            # In the future, this could be replaced with a more sophisticated model
            self.use_torch = True
            self.torch_device = torch.device(device)
            
            # Initialize boundary refiner
            self.boundary_refiner = BoundaryRefiner(device)
            
            logger.info("Word aligner initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Could not import required packages for word alignment: {e}")
            self.use_torch = False
            
        except Exception as e:
            logger.error(f"Error initializing word aligner: {e}")
            raise
    
    def refine_word_timestamps(self, audio_data: np.ndarray, 
                              sample_rate: int,
                              words: List[Dict[str, Any]],
                              segment_start: float = 0.0) -> List[Dict[str, Any]]:
        """Refine word-level timestamps using audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            words: List of word dictionaries with 'text', 'start', and 'end' keys
            segment_start: Start time of the segment in the original audio
            
        Returns:
            List[Dict[str, Any]]: Words with refined timestamps
        """
        if not words:
            logger.warning("No words provided for timestamp refinement")
            return words
        
        if not self.use_torch:
            logger.warning("Word alignment requires PyTorch and pyannote.core")
            return words
        
        try:
            # Create a copy of the words to avoid modifying the original
            refined_words = []
            
            # Process each word
            for i, word in enumerate(words):
                # Skip words without text
                if not word.get('text', '').strip():
                    refined_words.append(word.copy())
                    continue
                
                # Get word start and end times relative to the audio data
                start_time = word['start'] - segment_start
                end_time = word['end'] - segment_start
                
                # Convert to samples
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # Ensure valid range
                if start_sample < 0:
                    start_sample = 0
                if end_sample > len(audio_data):
                    end_sample = len(audio_data)
                
                # Skip words with invalid ranges
                if start_sample >= end_sample or start_sample >= len(audio_data):
                    refined_words.append(word.copy())
                    continue
                
                # Extract audio for this word
                word_audio = audio_data[start_sample:end_sample]
                
                # Refine word boundaries
                refined_start, refined_end = self.boundary_refiner.refine_boundaries(
                    word_audio, sample_rate
                )
                
                # Create refined word
                refined_word = word.copy()
                refined_word['start'] = start_time + refined_start
                refined_word['end'] = start_time + refined_end
                
                # Add to refined words
                refined_words.append(refined_word)
            
            # Fix any overlapping words
            refined_words = self._fix_overlapping_words(refined_words)
            
            logger.info(f"Refined timestamps for {len(refined_words)} words")
            return refined_words
            
        except Exception as e:
            logger.error(f"Error refining word timestamps: {e}")
            return words
    
    def _fix_overlapping_words(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix overlapping word timestamps.
        
        Args:
            words: List of word dictionaries with 'start' and 'end' keys
            
        Returns:
            List[Dict[str, Any]]: Words with fixed timestamps
        """
        if not words:
            return words
        
        # Sort words by start time
        sorted_words = sorted(words, key=lambda x: x['start'])
        
        # Fix overlaps
        for i in range(1, len(sorted_words)):
            prev_word = sorted_words[i-1]
            curr_word = sorted_words[i]
            
            # Check for overlap
            if prev_word['end'] > curr_word['start']:
                # Calculate midpoint
                midpoint = (prev_word['end'] + curr_word['start']) / 2
                
                # Adjust timestamps
                prev_word['end'] = midpoint
                curr_word['start'] = midpoint
        
        return sorted_words
    
    def align_transcript_with_audio(self, audio_data: np.ndarray,
                                   sample_rate: int,
                                   transcript: str,
                                   segment_start: float = 0.0,
                                   segment_end: float = None) -> List[Dict[str, Any]]:
        """Align a transcript with audio data to generate word-level timestamps.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            transcript: Transcript text
            segment_start: Start time of the segment in the original audio
            segment_end: End time of the segment in the original audio
            
        Returns:
            List[Dict[str, Any]]: Words with timestamps
        """
        if not transcript.strip():
            logger.warning("Empty transcript provided for alignment")
            return []
        
        if not self.use_torch:
            logger.warning("Word alignment requires PyTorch and pyannote.core")
            return []
        
        try:
            # Set default segment end if not provided
            if segment_end is None:
                segment_end = segment_start + (len(audio_data) / sample_rate)
            
            # Split transcript into words
            words = transcript.strip().split()
            
            # Calculate average word duration
            total_duration = segment_end - segment_start
            avg_word_duration = total_duration / len(words)
            
            # Create word dictionaries with initial timestamps
            word_dicts = []
            current_time = segment_start
            
            for word_text in words:
                word_dict = {
                    'text': word_text,
                    'start': current_time,
                    'end': current_time + avg_word_duration
                }
                
                word_dicts.append(word_dict)
                current_time += avg_word_duration
            
            # Refine timestamps
            refined_words = self.refine_word_timestamps(
                audio_data, sample_rate, word_dicts, segment_start
            )
            
            logger.info(f"Aligned {len(refined_words)} words with audio")
            return refined_words
            
        except Exception as e:
            logger.error(f"Error aligning transcript with audio: {e}")
            return [] 