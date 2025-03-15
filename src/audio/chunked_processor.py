"""
Chunked audio processor for VSAT.

This module provides functionality for processing large audio files in chunks,
which helps to reduce memory usage and improve performance.
"""

import os
import logging
import tempfile
import numpy as np
import soundfile as sf
from typing import List, Dict, Any, Callable, Optional, Tuple, Union

from src.utils.error_handler import ErrorSeverity

logger = logging.getLogger(__name__)

class ChunkingError(Exception):
    """Exception raised when an error occurs during chunked processing."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional details about the error
        """
        super().__init__(message)
        self.details = details or {}
        self.severity = ErrorSeverity.ERROR

class ChunkedProcessor:
    """Process audio files in chunks to reduce memory usage."""
    
    def __init__(self, 
                chunk_size: float = 30.0, 
                overlap: float = 5.0,
                processor_func: Callable = None,
                temp_dir: Optional[str] = None):
        """Initialize the chunked processor.
        
        Args:
            chunk_size: Size of each chunk in seconds
            overlap: Overlap between chunks in seconds
            processor_func: Function to process each chunk
            temp_dir: Directory to store temporary files
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.processor_func = processor_func
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        logger.info(f"ChunkedProcessor initialized with chunk_size={chunk_size}s, "
                  f"overlap={overlap}s, temp_dir={self.temp_dir}")
    
    def process_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Process an audio file in chunks.
        
        Args:
            file_path: Path to the audio file
            **kwargs: Additional arguments for the processor function
            
        Returns:
            Dict[str, Any]: Processing results
            
        Raises:
            FileNotFoundError: If the file does not exist
            ChunkingError: If processing fails
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load audio file
            logger.info(f"Loading audio file: {file_path}")
            audio_data, sample_rate = sf.read(file_path)
            
            # Process audio data
            result = self.process_data(audio_data, sample_rate, **kwargs)
            
            # Add file information
            result['file_info'] = {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'sample_rate': sample_rate,
                'duration': len(audio_data) / sample_rate
            }
            
            logger.info(f"Completed processing file: {file_path}")
            return result
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing file {file_path} in chunks: {str(e)}")
            raise ChunkingError(f"Failed to process audio file in chunks: {str(e)}", 
                              {'file_path': file_path})
    
    def process_data(self, audio_data: np.ndarray, sample_rate: int, **kwargs) -> Dict[str, Any]:
        """Process audio data in chunks.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            **kwargs: Additional arguments for the processor function
            
        Returns:
            Dict[str, Any]: Processing results
            
        Raises:
            ChunkingError: If processing fails
        """
        try:
            # Save chunks to disk
            logger.info("Saving audio chunks to disk")
            chunk_files = self._save_chunks_to_disk(audio_data, sample_rate, self.temp_dir)
            
            # Process each chunk
            logger.info(f"Processing {len(chunk_files)} chunks")
            chunk_results = []
            
            for i, chunk_file in enumerate(chunk_files):
                logger.info(f"Processing chunk {i+1}/{len(chunk_files)}")
                
                # Load chunk
                chunk_data, chunk_sr = sf.read(chunk_file)
                
                # Process chunk
                chunk_result = self.processor_func(chunk_data, chunk_sr, **kwargs)
                
                # Add chunk information
                chunk_result['chunk_info'] = {
                    'index': i,
                    'file': chunk_file,
                    'total_chunks': len(chunk_files)
                }
                
                chunk_results.append(chunk_result)
            
            # Merge results
            logger.info("Merging chunk results")
            merged_result = self._merge_results(chunk_results, sample_rate, len(audio_data) / sample_rate)
            
            # Clean up chunks
            logger.info("Cleaning up temporary chunk files")
            self._cleanup_chunks(chunk_files)
            
            return merged_result
            
        except Exception as e:
            logger.error(f"Error processing audio data in chunks: {str(e)}")
            raise ChunkingError(f"Failed to process audio data in chunks: {str(e)}", 
                              {'sample_rate': sample_rate, 'audio_shape': audio_data.shape})
    
    def _save_chunks_to_disk(self, audio_data: np.ndarray, sample_rate: int, 
                           temp_dir: str) -> List[str]:
        """Save audio data in chunks to disk.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            temp_dir: Directory to save chunks to
            
        Returns:
            List[str]: Paths to chunk files
        """
        # Calculate chunk and overlap sizes in samples
        chunk_size_samples = int(self.chunk_size * sample_rate)
        overlap_samples = int(self.overlap * sample_rate)
        
        # Calculate number of chunks
        total_samples = len(audio_data)
        step_size = chunk_size_samples - overlap_samples
        
        # Ensure at least one chunk
        if step_size <= 0:
            logger.warning("Overlap is greater than or equal to chunk size. Setting overlap to 0.")
            overlap_samples = 0
            step_size = chunk_size_samples
        
        num_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / step_size)))
        
        # Create chunks
        chunk_files = []
        
        for i in range(num_chunks):
            # Calculate start and end samples
            start = i * step_size
            end = min(start + chunk_size_samples, total_samples)
            
            # Extract chunk
            chunk = audio_data[start:end]
            
            # Create chunk file
            chunk_file = os.path.join(temp_dir, f"chunk_{i:04d}.wav")
            sf.write(chunk_file, chunk, sample_rate)
            
            # Add to list
            chunk_files.append(chunk_file)
        
        logger.info(f"Created {len(chunk_files)} chunks from {total_samples} samples")
        return chunk_files
    
    def _cleanup_chunks(self, chunk_files: List[str]):
        """Clean up temporary chunk files.
        
        Args:
            chunk_files: List of chunk file paths
        """
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                except Exception as e:
                    logger.warning(f"Failed to remove chunk file {chunk_file}: {str(e)}")
    
    def _merge_results(self, chunk_results: List[Dict[str, Any]], 
                      sample_rate: int, duration: float) -> Dict[str, Any]:
        """Merge results from individual chunks.
        
        Args:
            chunk_results: List of results from individual chunks
            sample_rate: Sample rate of the audio
            duration: Duration of the audio in seconds
            
        Returns:
            Dict[str, Any]: Merged results
        """
        # Basic merge implementation - can be overridden in subclasses
        return {
            'merged_result': True,
            'num_chunks': len(chunk_results),
            'sample_rate': sample_rate,
            'duration': duration,
            'chunk_results': chunk_results
        } 