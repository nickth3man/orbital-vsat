"""
Player utility functions for VSAT.

This module provides utility functions for audio playback.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from src.utils.error_handler import AudioError, FileError, ErrorSeverity

logger = logging.getLogger(__name__)

def validate_file_path(file_path: str) -> Path:
    """Validate that a file exists and return a Path object.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Path: Path object for the file
        
    Raises:
        FileError: If the file does not exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileError(
            f"Audio file not found: {file_path}",
            ErrorSeverity.ERROR,
            {"file_path": file_path}
        )
    return path

def validate_volume(volume: float) -> float:
    """Validate and normalize volume level.
    
    Args:
        volume: Volume level (should be between 0.0 and 1.0)
        
    Returns:
        float: Normalized volume level
    """
    if volume < 0.0:
        return 0.0
    elif volume > 1.0:
        return 1.0
    return volume

def validate_segment_boundaries(start: float, end: float, duration: float = None) -> tuple[float, float]:
    """Validate and normalize segment boundaries.
    
    Args:
        start: Start position in seconds
        end: End position in seconds
        duration: Total duration in seconds (optional)
        
    Returns:
        tuple: (start, end) positions in seconds
        
    Raises:
        AudioError: If the segment boundaries are invalid
    """
    # Ensure start is not negative
    if start < 0:
        start = 0
        logger.warning("Start position adjusted to 0")
        
    # Check if end is after start
    if end <= start:
        raise AudioError(
            "End time must be greater than start time",
            ErrorSeverity.WARNING,
            {"start": start, "end": end}
        )
    
    # If duration is provided, ensure end is not beyond duration
    if duration is not None and end > duration:
        end = duration
        logger.warning(f"End position adjusted to match duration: {duration}")
    
    return start, end

def validate_position(position: float, duration: float) -> float:
    """Validate and normalize playback position.
    
    Args:
        position: Position in seconds
        duration: Total duration in seconds
        
    Returns:
        float: Normalized position in seconds
    """
    if position < 0:
        return 0
    if duration is not None and position > duration:
        return duration
    return position 