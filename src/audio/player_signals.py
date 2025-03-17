"""
Player signal handling module for VSAT.

This module provides functionality for handling audio player signals.
"""

import logging
from PyQt6.QtCore import QObject, pyqtSlot
from PyQt6.QtMultimedia import QMediaPlayer

from src.audio.player_state import PlaybackState


logger = logging.getLogger(__name__)


class PlayerSignalHandler(QObject):
    """Handles signal connections and processing for the audio player."""

    def __init__(self, player_obj, parent=None):
        """Initialize the signal handler.

        Args:
            player_obj: The AudioPlayer instance that owns this handler
            parent: Parent QObject
        """
        super().__init__(parent)
        self.player_obj = player_obj
        logger.debug("Player signal handler initialized")


    @pyqtSlot(int)
    def on_position_changed(self, position_ms: int):
        """Handle position changed signal from QMediaPlayer.

        Args:
            position_ms: Position in milliseconds
        """
        # Convert to seconds
        position_seconds = position_ms / 1000.0

        # Update state
        self.player_obj.state.update_position(position_seconds)

        # Emit the position signal
        self.player_obj.positionChanged.emit(position_seconds)


    @pyqtSlot(QMediaPlayer.PlaybackState)
    def on_state_changed(self, state: QMediaPlayer.PlaybackState):
        """Handle playback state changed signal from QMediaPlayer.

        Args:
            state: New playback state
        """
        # Map QMediaPlayer state to string and our PlaybackState enum
        if state == QMediaPlayer.PlaybackState.PlayingState:
            state_str = "playing"
            playback_state = PlaybackState.PLAYING
        elif state == QMediaPlayer.PlaybackState.PausedState:
            state_str = "paused"
            playback_state = PlaybackState.PAUSED
        else:  # StoppedState
            state_str = "stopped"
            playback_state = PlaybackState.STOPPED

        # Update state
        self.player_obj.state.update_state(playback_state)

        # Emit state changed signal
        self.player_obj.stateChanged.emit(state_str)

        # If stopped, emit playback finished signal
        if state == QMediaPlayer.PlaybackState.StoppedState:
            # Only emit if we're not in segment playback mode
            # (segment playback has its own handling)
            if not self.player_obj.state.is_segment_playback:
                self.player_obj.playbackFinished.emit()


    @pyqtSlot(QMediaPlayer.Error, str)
    def on_error(self, error: QMediaPlayer.Error, error_string: str):
        """Handle error signal from QMediaPlayer.

        Args:
            error: Error code
            error_string: Error message
        """
        # Log the error
        logger.error(f"Media player error: {error_string} (code: {error})")

        # Emit error signal
        error_msg = f"Audio playback error: {error_string}"
        self.player_obj.errorOccurred.emit(error_msg)