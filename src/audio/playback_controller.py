"""
Playback controller module for VSAT.

This module provides playback control logic for the audio player.
"""

import logging
from typing import Optional
from pathlib import Path

from PyQt6.QtCore import QObject, QUrl
from PyQt6.QtMultimedia import QMediaPlayer

from src.utils.error_handler import AudioError, FileError, ErrorSeverity
from src.audio.player_utils import (
    validate_file_path,
    validate_position,
)
from src.audio.player_state import PlaybackState

logger = logging.getLogger(__name__)

class PlaybackController(QObject):
    """Controls playback operations for the audio player."""
    
    def __init__(self, audio_player, parent=None):
        """Initialize the playback controller.
        
        Args:
            audio_player: The AudioPlayer instance that owns this controller
            parent: Parent QObject
        """
        super().__init__(parent)
        self.audio_player = audio_player
        logger.debug("Playback controller initialized")
    
    def load_file(self, file_path: str) -> bool:
        """Load an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate file exists
            path = validate_file_path(file_path)
            
            # Stop any current playback
            self.audio_player.stop()
            
            # Set the source
            url = QUrl.fromLocalFile(str(path.absolute()))
            self.audio_player.player.setSource(url)
            
            # Check if the format is supported
            if self.audio_player.player.mediaStatus() == QMediaPlayer.MediaStatus.InvalidMedia:
                raise AudioError(
                    f"Unsupported audio format: {file_path}",
                    ErrorSeverity.ERROR,
                    {"file_path": file_path}
                )
            
            # Get duration in seconds
            duration = self.audio_player.player.duration() / 1000.0
            
            # Update state
            self.audio_player.state.set_file(file_path, duration)
            
            logger.info(f"Loaded audio file: {file_path} (duration: {duration:.2f}s)")
            return True
            
        except (FileError, AudioError) as e:
            # Re-raise VSAT errors
            logger.error(f"Error loading audio file: {e}")
            self.audio_player.errorOccurred.emit(str(e))
            raise
            
        except Exception as e:
            # Wrap other exceptions
            error_msg = f"Error loading audio file: {str(e)}"
            logger.error(error_msg)
            self.audio_player.errorOccurred.emit(error_msg)
            return False
    
    def play(self, file_path: Optional[str] = None, start: Optional[float] = None, end: Optional[float] = None):
        """Play audio from the specified file or the current file.
        
        Args:
            file_path: Path to the audio file (optional, uses current file if None)
            start: Start position in seconds (optional)
            end: End position in seconds (optional)
        """
        try:
            # Load file if specified
            if file_path:
                if not self.load_file(file_path):
                    return
            
            # Check if we have a file loaded
            if not self.audio_player.state.has_file():
                raise AudioError(
                    "No audio file loaded",
                    ErrorSeverity.WARNING,
                    {}
                )
            
            # If start and end are specified, play segment
            if start is not None and end is not None:
                self.audio_player.play_segment(start, end)
                return
            
            # If only start is specified, seek to that position
            if start is not None:
                self.set_position(start)
            
            # Start playback
            self.audio_player.player.play()
            
            # Update state
            self.audio_player.state.update_state(PlaybackState.PLAYING)
            
            logger.info(f"Playing audio: {self.audio_player.state.current_file}")
            
        except AudioError as e:
            # Handle VSAT errors
            logger.error(f"Error playing audio: {e}")
            self.audio_player.errorOccurred.emit(str(e))
            
        except Exception as e:
            # Wrap other exceptions
            error_msg = f"Error playing audio: {str(e)}"
            logger.error(error_msg)
            self.audio_player.errorOccurred.emit(error_msg)
    
    def set_position(self, position: float):
        """Set the playback position.
        
        Args:
            position: Position in seconds
        """
        try:
            # Check if we have a file loaded
            if not self.audio_player.state.has_file():
                raise AudioError(
                    "No audio file loaded",
                    ErrorSeverity.WARNING,
                    {}
                )
            
            # Validate and normalize position
            position = validate_position(position, self.audio_player.state.current_duration)
            
            # Update state
            self.audio_player.state.update_position(position)
            
            # Convert to milliseconds
            position_ms = int(position * 1000)
            
            # Set position
            self.audio_player.player.setPosition(position_ms)
            
            logger.debug(f"Set position to {position:.2f}s")
            
        except AudioError as e:
            # Handle VSAT errors
            logger.error(f"Error setting position: {e}")
            self.audio_player.errorOccurred.emit(str(e))
            
        except Exception as e:
            # Wrap other exceptions
            error_msg = f"Error setting position: {str(e)}"
            logger.error(error_msg)
            self.audio_player.errorOccurred.emit(error_msg) 