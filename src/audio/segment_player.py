"""
Segment playback module for VSAT.

This module provides functionality for playing audio segments.
"""

import logging
from typing import Dict, Any, Optional
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from src.utils.error_handler import AudioError, ErrorSeverity
from src.audio.player_state import PlaybackState

logger = logging.getLogger(__name__)

class SegmentPlayer(QObject):
    """Class for handling audio segment playback."""
    
    # Signal emitted when segment playback finishes
    segmentFinished = pyqtSignal()
    
    # Signal emitted when an error occurs
    errorOccurred = pyqtSignal(str)
    
    def __init__(self, player: QMediaPlayer, audio_output: QAudioOutput, 
                 word_padding: float = 0.05, crossfade: float = 0.01, parent=None):
        """Initialize the segment player.
        
        Args:
            player: QMediaPlayer instance to use for playback
            audio_output: QAudioOutput instance for the player
            word_padding: Padding in seconds to add around words (default: 0.05)
            crossfade: Crossfade duration in seconds for segment transitions (default: 0.01)
            parent: Parent QObject
        """
        super().__init__(parent)
        
        # Store references to player and audio output
        self.player = player
        self.audio_output = audio_output
        
        # Store configuration
        self.word_padding = word_padding
        self.crossfade = crossfade
        
        # Create timer for segment playback
        self.segment_timer = QTimer(self)
        self.segment_timer.setSingleShot(True)
        self.segment_timer.timeout.connect(self._on_segment_end)
        
        logger.debug(f"Segment player initialized (word_padding={word_padding}s, crossfade={crossfade}s)")
    
    def play_segment(self, start: float, end: float):
        """Play a specific segment of the audio.
        
        Args:
            start: Start position in seconds
            end: End position in seconds
        """
        try:
            # Check if the segment is valid
            if start < 0:
                logger.warning("Start position adjusted to 0")
                start = 0
                
            if end <= start:
                raise AudioError(
                    "End time must be greater than start time",
                    ErrorSeverity.WARNING,
                    {"start": start, "end": end}
                )
            
            # Apply crossfade if enabled
            if self.crossfade > 0:
                # Adjust start position to account for crossfade
                # This ensures the audio starts at the correct position
                # but allows for a smooth transition
                start_with_crossfade = max(0, start - self.crossfade / 2)
                
                # Set position to adjusted start
                self.player.setPosition(int(start_with_crossfade * 1000))
                
                # Calculate duration including crossfade
                duration = end - start + self.crossfade
            else:
                # Set position to start
                self.player.setPosition(int(start * 1000))
                
                # Calculate duration
                duration = end - start
            
            # Calculate duration in milliseconds
            duration_ms = int(duration * 1000)
            
            # Start timer to stop at end position
            self.segment_timer.start(duration_ms)
            
            # Start playback
            self.player.play()
            
            logger.debug(f"Playing segment from {start:.2f}s to {end:.2f}s")
            
        except Exception as e:
            error_msg = f"Error playing segment: {str(e)}"
            logger.error(error_msg)
            self.errorOccurred.emit(error_msg)
    
    def play_word(self, word: Dict[str, Any]):
        """Play a specific word from the transcript.
        
        Args:
            word: Word data with 'start' and 'end' times
        """
        try:
            # Check if word has required fields
            if 'start' not in word or 'end' not in word:
                raise AudioError(
                    "Word data missing required timing information",
                    ErrorSeverity.WARNING,
                    {"word": word}
                )
            
            # Add padding around the word
            start = max(0, word['start'] - self.word_padding)
            end = word['end'] + self.word_padding
            
            # Play segment for the word
            self.play_segment(start, end)
            logger.debug(f"Playing word: {word.get('text', '')} ({start:.2f}s to {end:.2f}s)")
            
        except Exception as e:
            error_msg = f"Error playing word: {str(e)}"
            logger.error(error_msg)
            self.errorOccurred.emit(error_msg)
    
    def stop_segment(self):
        """Stop segment playback."""
        # Stop segment timer if active
        if self.segment_timer.isActive():
            self.segment_timer.stop()
        
        logger.debug("Segment playback stopped")
    
    def set_word_padding(self, padding: float):
        """Set the word padding value.
        
        Args:
            padding: Padding in seconds to add around words
        """
        self.word_padding = max(0, padding)
        logger.debug(f"Word padding set to {self.word_padding}s")
    
    def set_crossfade(self, crossfade: float):
        """Set the crossfade duration.
        
        Args:
            crossfade: Crossfade duration in seconds
        """
        self.crossfade = max(0, crossfade)
        logger.debug(f"Crossfade set to {self.crossfade}s")
    
    @pyqtSlot()
    def _on_segment_end(self):
        """Handle segment playback end."""
        # Stop playback
        self.player.stop()
        
        # Emit playback finished signal
        self.segmentFinished.emit()
        
        logger.debug("Segment playback finished") 