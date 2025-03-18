"""
Audio playback module for VSAT.

This module provides functionality for playing audio files and segments.
"""

import logging
from typing import Optional, Dict, Any, Callable

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from src.audio.segment_player import SegmentPlayer
from src.audio.player_signals import PlayerSignalHandler
from src.audio.playback_controller import PlaybackController
from src.audio.volume_controller import VolumeController
from src.audio.player_state import PlayerState, PlaybackState
from src.audio.player_events import PlayerEvents, EventType
from src.audio.player_config import PlayerConfig


logger = logging.getLogger(__name__)


class AudioPlayer(QObject):
    """Class for handling audio playback."""

    # Signal emitted when playback position changes
    positionChanged = pyqtSignal(float)  # Position in seconds

    # Signal emitted when playback state changes
    stateChanged = pyqtSignal(str)  # State name: 'playing', 'paused', 'stopped'

    # Signal emitted when playback reaches the end
    playbackFinished = pyqtSignal()

    # Signal emitted when an error occurs
    errorOccurred = pyqtSignal(str)

    def __init__(self, config_path: Optional[str] = None, parent=None):
        """Initialize the audio player.
        
        Args:
            config_path: Path to the configuration file (optional)
            parent: Parent QObject
        """
        super().__init__(parent)

        # Initialize configuration
        self.config = PlayerConfig(config_path)

        # Initialize state manager
        self.state = PlayerState()

        # Initialize event system
        self.events = PlayerEvents()

        # Initialize media player
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

        # Set initial volume from configuration
        initial_volume = self.config.get("volume", 0.7)
        self.audio_output.setVolume(initial_volume)
        self.state.update_volume(initial_volume)

        # Set initial muted state from configuration
        initial_muted = self.config.get("muted", False)
        self.audio_output.setMuted(initial_muted)
        self.state.set_muted(initial_muted)

        # Initialize signal handler
        self.signal_handler = PlayerSignalHandler(self)

        # Connect signals
        self.player.positionChanged.connect(self.signal_handler.on_position_changed)
        self.player.playbackStateChanged.connect(self.signal_handler.on_state_changed)
        self.player.errorOccurred.connect(self.signal_handler.on_error)

        # Connect internal signals to events
        self.positionChanged.connect(self._on_position_changed)
        self.stateChanged.connect(self._on_state_changed)
        self.playbackFinished.connect(self._on_playback_finished)
        self.errorOccurred.connect(self._on_error_occurred)

        # Initialize segment player with configuration
        self.segment_player = SegmentPlayer(
            self.player,
            self.audio_output,
            word_padding=self.config.get("word_padding", 0.05),
            crossfade=self.config.get("segment_crossfade", 0.01)
        )
        self.segment_player.segmentFinished.connect(self.playbackFinished)

        # Initialize controllers
        self.playback_controller = PlaybackController(self)
        self.volume_controller = VolumeController(self, self.audio_output)

        # Apply hardware acceleration setting
        if self.config.get("use_hardware_acceleration", True):
            self._enable_hardware_acceleration()

        # Set playback rate
        self.set_playback_rate(self.config.get("playback_rate", 1.0))

        logger.debug("Audio player initialized")

    def load_file(self, file_path: str) -> bool:
        """Load an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        result = self.playback_controller.load_file(file_path)
        if result:
            # Add to recent files
            self.config.add_recent_file(file_path)
            
            # Auto-play if configured
            if self.config.get("auto_play_on_load", False):
                self.play()
            
            # Emit event
            self.events.emit(EventType.FILE_LOADED, {
                "file_path": file_path,
                "duration": self.state.current_duration
            })
        return result

    def play(self, file_path: Optional[str] = None, start: Optional[float] = None,
             end: Optional[float] = None):
        """Play audio from the specified file or the current file.
        
        Args:
            file_path: Path to the audio file (optional, uses current file if None)
            start: Start position in seconds (optional)
            end: End position in seconds (optional)
        """
        self.playback_controller.play(file_path, start, end)

    def pause(self):
        """Pause playback."""
        self.player.pause()
        self.state.update_state(PlaybackState.PAUSED)
        logger.debug("Playback paused")

    def stop(self):
        """Stop playback."""
        # Stop segment playback if active
        if self.state.is_segment_playback:
            self.segment_player.stop_segment()
            self.state.end_segment_playback()
            self.events.emit(EventType.SEGMENT_FINISHED, {})
        
        # Stop player
        self.player.stop()
        self.state.update_state(PlaybackState.STOPPED)
        logger.debug("Playback stopped")

    def set_position(self, position: float):
        """Set the playback position.
        
        Args:
            position: Position in seconds
        """
        self.playback_controller.set_position(position)

    def get_position(self) -> float:
        """Get the current playback position.
        
        Returns:
            float: Position in seconds
        """
        return self.player.position() / 1000.0  # Convert to seconds

    def set_volume(self, volume: float):
        """Set the playback volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.volume_controller.set_volume(volume)
        self.state.update_volume(volume)
        
        # Update configuration if remember_volume is enabled
        if self.config.get("remember_volume", True):
            self.config.set("volume", volume)
        
        self.events.emit(EventType.VOLUME_CHANGED, {"volume": volume})

    def get_volume(self) -> float:
        """Get the current volume level.
        
        Returns:
            float: Volume level (0.0 to 1.0)
        """
        return self.volume_controller.get_volume()

    def mute(self):
        """Mute audio playback."""
        self.volume_controller.mute()
        self.state.set_muted(True)
        
        # Update configuration if remember_volume is enabled
        if self.config.get("remember_volume", True):
            self.config.set("muted", True)
        
        self.events.emit(EventType.MUTE_CHANGED, {"muted": True})

    def unmute(self):
        """Unmute audio playback."""
        self.volume_controller.unmute()
        self.state.set_muted(False)
        
        # Update configuration if remember_volume is enabled
        if self.config.get("remember_volume", True):
            self.config.set("muted", False)
        
        self.events.emit(EventType.MUTE_CHANGED, {"muted": False})

    def is_muted(self) -> bool:
        """Check if audio is muted.
        
        Returns:
            bool: True if muted, False otherwise
        """
        return self.state.is_muted

    def is_playing(self) -> bool:
        """Check if audio is currently playing.
        
        Returns:
            bool: True if playing, False otherwise
        """
        return self.state.is_playing()

    def play_segment(self, start: float, end: float):
        """Play a segment of the current audio file.
        
        Args:
            start: Start position in seconds
            end: End position in seconds
        """
        self.segment_player.play_segment(start, end)
        self.state.start_segment_playback(start, end)
        self.events.emit(EventType.SEGMENT_STARTED, {
            "start": start,
            "end": end
        })

    def play_word(self, word: Dict[str, Any]):
        """Play audio for a specific word.
        
        Args:
            word: Word data with 'start' and 'end' keys
        """
        self.segment_player.play_word(word)
        start = word.get('start', 0)
        end = word.get('end', 0)
        self.state.start_segment_playback(start, end)
        self.events.emit(EventType.SEGMENT_STARTED, {
            "start": start,
            "end": end,
            "word": word
        })

    def get_duration(self) -> float:
        """Get the duration of the current audio file.
        
        Returns:
            float: Duration in seconds
        """
        return self.state.current_duration

    def set_playback_rate(self, rate: float) -> None:
        """Set the playback rate.
        
        Args:
            rate: Playback rate (0.5 to 2.0)
        """
        # Validate rate
        if rate < 0.5:
            rate = 0.5
        elif rate > 2.0:
            rate = 2.0
        
        # Set rate
        self.player.setPlaybackRate(rate)
        
        # Update configuration
        self.config.set("playback_rate", rate)
        
        logger.debug(f"Set playback rate to {rate}")

    def get_playback_rate(self) -> float:
        """Get the current playback rate.
        
        Returns:
            float: Playback rate
        """
        return self.player.playbackRate()

    def get_recent_files(self) -> list[str]:
        """Get the list of recent files.
        
        Returns:
            list: List of recent file paths
        """
        return self.config.get_recent_files()

    def save_config(self) -> bool:
        """Save the current configuration.
        
        Returns:
            bool: True if successful, False otherwise
        """
        return self.config.save_config()

    def add_event_listener(self, event_type: EventType, callback: Callable) -> None:
        """Add a listener for a specific event type.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when the event occurs
        """
        self.events.add_listener(event_type, callback)

    def remove_event_listener(self, event_type: EventType, callback: Callable) -> None:
        """Remove a listener for a specific event type.
        
        Args:
            event_type: Type of event to remove listener from
            callback: Function to remove
        """
        self.events.remove_listener(event_type, callback)

    def _enable_hardware_acceleration(self) -> None:
        """Enable hardware acceleration for the media player."""
        # Check for hardware support
        available_backends = QMediaPlayer.supportedVideoSinks()
        
        if "directshow" in available_backends:
            self.player.setVideoOutput(None)
            logger.debug("DirectShow hardware acceleration enabled")

    def _on_position_changed(self, position: float):
        """Handle position changed signal.
        
        Args:
            position: Position in seconds
        """
        self.events.emit(EventType.POSITION_CHANGED, {"position": position})

    def _on_state_changed(self, state: str):
        """Handle state changed signal.
        
        Args:
            state: New state as string
        """
        if state == "playing":
            self.events.emit(EventType.PLAYBACK_STARTED, {})
        elif state == "paused":
            self.events.emit(EventType.PLAYBACK_PAUSED, {})
        elif state == "stopped":
            self.events.emit(EventType.PLAYBACK_STOPPED, {})

    def _on_playback_finished(self):
        """Handle playback finished signal."""
        self.events.emit(EventType.PLAYBACK_FINISHED, {})
        
        if self.state.is_segment_playback:
            self.events.emit(EventType.SEGMENT_FINISHED, {
                "start": self.state.segment_start,
                "end": self.state.segment_end
            })
            self.state.end_segment_playback()

    def _on_error_occurred(self, error_message: str):
        """Handle error occurred signal.
        
        Args:
            error_message: Error message
        """
        self.events.emit(EventType.ERROR_OCCURRED, {"message": error_message})
        logger.error(f"Audio playback error: {error_message}")