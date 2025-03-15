"""
Player state management module for VSAT.

This module provides state tracking for the audio player.
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PlaybackState(Enum):
    """Enum representing possible playback states."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"

class PlayerState:
    """Manages and tracks the state of the audio player."""
    
    def __init__(self):
        """Initialize the player state."""
        # File information
        self.current_file = None
        self.current_duration = 0.0
        
        # Playback state
        self.state = PlaybackState.STOPPED
        self.position = 0.0  # Current position in seconds
        
        # Segment playback state
        self.is_segment_playback = False
        self.segment_start = None
        self.segment_end = None
        
        # Volume state
        self.volume = 0.7  # Default volume
        self.is_muted = False
        
        logger.debug("Player state initialized")
    
    def set_file(self, file_path: str, duration: float):
        """Set the current file.
        
        Args:
            file_path: Path to the audio file
            duration: Duration of the file in seconds
        """
        self.current_file = file_path
        self.current_duration = duration
        self.position = 0.0
        self.reset_segment()
        logger.debug(f"Player state: file set to {file_path} (duration: {duration:.2f}s)")
    
    def clear_file(self):
        """Clear the current file."""
        self.current_file = None
        self.current_duration = 0.0
        self.position = 0.0
        self.state = PlaybackState.STOPPED
        self.reset_segment()
        logger.debug("Player state: file cleared")
    
    def update_position(self, position: float):
        """Update the current playback position.
        
        Args:
            position: Current position in seconds
        """
        self.position = position
    
    def update_state(self, state: PlaybackState):
        """Update the current playback state.
        
        Args:
            state: New playback state
        """
        self.state = state
        logger.debug(f"Player state: changed to {state.value}")
    
    def start_segment_playback(self, start: float, end: float):
        """Start segment playback.
        
        Args:
            start: Start position in seconds
            end: End position in seconds
        """
        self.is_segment_playback = True
        self.segment_start = start
        self.segment_end = end
        self.position = start
        logger.debug(f"Player state: segment playback started ({start:.2f}s to {end:.2f}s)")
    
    def end_segment_playback(self):
        """End segment playback."""
        self.reset_segment()
        logger.debug("Player state: segment playback ended")
    
    def reset_segment(self):
        """Reset segment playback state."""
        self.is_segment_playback = False
        self.segment_start = None
        self.segment_end = None
    
    def update_volume(self, volume: float):
        """Update the volume level.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.volume = volume
        logger.debug(f"Player state: volume set to {volume:.2f}")
    
    def set_muted(self, muted: bool):
        """Set the muted state.
        
        Args:
            muted: Whether audio is muted
        """
        self.is_muted = muted
        logger.debug(f"Player state: muted set to {muted}")
    
    def has_file(self) -> bool:
        """Check if a file is loaded.
        
        Returns:
            bool: True if a file is loaded, False otherwise
        """
        return self.current_file is not None
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing.
        
        Returns:
            bool: True if playing, False otherwise
        """
        return self.state == PlaybackState.PLAYING 