"""
Volume controller module for VSAT.

This module provides volume control logic for the audio player.
"""

import logging
from PyQt6.QtCore import QObject
from PyQt6.QtMultimedia import QAudioOutput

from src.audio.player_utils import validate_volume

logger = logging.getLogger(__name__)

class VolumeController(QObject):
    """Controls volume operations for the audio player."""
    
    def __init__(self, audio_player, audio_output: QAudioOutput, parent=None):
        """Initialize the volume controller.
        
        Args:
            audio_player: The AudioPlayer instance that owns this controller
            audio_output: QAudioOutput instance for the player
            parent: Parent QObject
        """
        super().__init__(parent)
        self.audio_player = audio_player
        self.audio_output = audio_output
        logger.debug("Volume controller initialized")
    
    def set_volume(self, volume: float):
        """Set the playback volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        try:
            # Validate and normalize volume
            volume = validate_volume(volume)
            
            # Set volume
            self.audio_output.setVolume(volume)
            
            logger.debug(f"Set volume to {volume:.2f}")
            
        except Exception as e:
            # Wrap exceptions
            error_msg = f"Error setting volume: {str(e)}"
            logger.error(error_msg)
            self.audio_player.errorOccurred.emit(error_msg)
    
    def get_volume(self) -> float:
        """Get the current volume level.
        
        Returns:
            float: Volume level (0.0 to 1.0)
        """
        return self.audio_output.volume()
        
    def mute(self):
        """Mute the audio output."""
        try:
            self.audio_output.setMuted(True)
            logger.debug("Audio muted")
        except Exception as e:
            error_msg = f"Error muting audio: {str(e)}"
            logger.error(error_msg)
            self.audio_player.errorOccurred.emit(error_msg)
    
    def unmute(self):
        """Unmute the audio output."""
        try:
            self.audio_output.setMuted(False)
            logger.debug("Audio unmuted")
        except Exception as e:
            error_msg = f"Error unmuting audio: {str(e)}"
            logger.error(error_msg)
            self.audio_player.errorOccurred.emit(error_msg)
    
    def is_muted(self) -> bool:
        """Check if audio is muted.
        
        Returns:
            bool: True if muted, False otherwise
        """
        return self.audio_output.isMuted() 