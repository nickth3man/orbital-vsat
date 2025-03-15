"""
Boundary refinement module for VSAT.

This module provides functionality for refining word boundaries in audio
using energy-based approaches.
"""

import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

class BoundaryRefiner:
    """Class for refining word boundaries in audio."""
    
    def __init__(self, device: str = "cpu"):
        """Initialize the boundary refiner.
        
        Args:
            device: Device to use for processing ("cpu" or "cuda")
        """
        self.device = device
        logger.debug(f"Boundary refiner initialized on {device}")
    
    def refine_boundaries(self, audio: np.ndarray, 
                         sample_rate: int, 
                         padding_ms: int = 50) -> Tuple[float, float]:
        """Refine word boundaries using energy-based approach.
        
        Args:
            audio: Audio data for the word
            sample_rate: Sample rate of the audio
            padding_ms: Padding in milliseconds to add to the boundaries
            
        Returns:
            Tuple[float, float]: Refined start and end times in seconds
        """
        # Calculate energy (squared amplitude)
        energy = audio ** 2
        
        # Apply smoothing
        window_size = int(sample_rate * 0.01)  # 10ms window
        if window_size > 1 and len(energy) > window_size:
            kernel = np.ones(window_size) / window_size
            energy_smooth = np.convolve(energy, kernel, mode='same')
        else:
            energy_smooth = energy
        
        # Calculate threshold (25% of max energy)
        threshold = 0.25 * np.max(energy_smooth)
        
        # Find regions above threshold
        above_threshold = energy_smooth > threshold
        
        # Find the first and last indices above threshold
        if np.any(above_threshold):
            indices = np.where(above_threshold)[0]
            start_idx = indices[0]
            end_idx = indices[-1]
        else:
            # If no clear energy peak, use the middle 80% of the audio
            start_idx = int(len(audio) * 0.1)
            end_idx = int(len(audio) * 0.9)
        
        # Convert to seconds
        start_sec = start_idx / sample_rate
        end_sec = end_idx / sample_rate
        
        # Add padding (convert ms to seconds)
        padding_sec = padding_ms / 1000.0
        start_sec = max(0, start_sec - padding_sec)
        end_sec = min(len(audio) / sample_rate, end_sec + padding_sec)
        
        # Calculate confidence
        confidence = self.calculate_boundary_confidence(audio, start_sec, end_sec)
        logger.debug(f"Refined boundaries: {start_sec:.3f}s to {end_sec:.3f}s (confidence: {confidence:.2f})")
        
        return start_sec, end_sec
    
    def calculate_boundary_confidence(self, audio: np.ndarray, 
                                    start_sec: float, 
                                    end_sec: float) -> float:
        """Calculate confidence score for the refined boundaries.
        
        Args:
            audio: Audio data for the word
            start_sec: Start time in seconds
            end_sec: End time in seconds
            
        Returns:
            float: Confidence score (0-1)
        """
        # Simple confidence calculation based on energy distribution
        # More sophisticated methods could be implemented in the future
        
        # Calculate energy
        energy = audio ** 2
        
        # Calculate total energy
        total_energy = np.sum(energy)
        
        if total_energy == 0:
            return 0.5  # Default confidence for silent regions
        
        # Calculate energy ratio in the selected region
        start_sample = int(start_sec * len(audio) / (len(audio) / audio.size))
        end_sample = int(end_sec * len(audio) / (len(audio) / audio.size))
        
        # Ensure valid indices
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        if start_sample >= end_sample:
            return 0.5
        
        region_energy = np.sum(energy[start_sample:end_sample])
        energy_ratio = region_energy / total_energy
        
        # Convert to confidence score (0-1)
        confidence = min(1.0, energy_ratio * 1.5)  # Scale up slightly
        
        return confidence 