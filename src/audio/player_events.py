"""
Player events module for VSAT.

This module provides an event system for the audio player.
"""

import logging
from enum import Enum
from typing import Dict, Any, Callable, List, Optional
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Enum representing possible event types."""
    PLAYBACK_STARTED = "playback_started"
    PLAYBACK_PAUSED = "playback_paused"
    PLAYBACK_STOPPED = "playback_stopped"
    PLAYBACK_FINISHED = "playback_finished"
    POSITION_CHANGED = "position_changed"
    VOLUME_CHANGED = "volume_changed"
    MUTE_CHANGED = "mute_changed"
    FILE_LOADED = "file_loaded"
    FILE_CLOSED = "file_closed"
    SEGMENT_STARTED = "segment_started"
    SEGMENT_FINISHED = "segment_finished"
    ERROR_OCCURRED = "error_occurred"

class PlayerEvent(Enum):
    """Enum for audio player events."""
    PLAYING = 0
    PAUSED = 1
    STOPPED = 2
    LOADING = 3
    LOADED = 4
    ERROR = 5
    POSITION_CHANGED = 6
    DURATION_CHANGED = 7
    VOLUME_CHANGED = 8
    SEGMENT_STARTED = 9
    SEGMENT_ENDED = 10

class PlayerSignals(QObject):
    """Signal handler for audio player events."""
    
    # Playback state signals
    playback_state_changed = pyqtSignal(PlayerEvent)
    
    # Position and duration signals
    position_changed = pyqtSignal(int)  # Position in milliseconds
    duration_changed = pyqtSignal(int)  # Duration in milliseconds
    
    # Volume signal
    volume_changed = pyqtSignal(int)  # Volume as percentage (0-100)
    
    # Segment signals
    segment_started = pyqtSignal(dict)  # Segment info
    segment_ended = pyqtSignal(dict)    # Segment info
    
    # Error signal
    error_occurred = pyqtSignal(str)    # Error message
    
    def on_playback_state_changed(self, state: PlayerEvent):
        """Handle playback state changes.
        
        Args:
            state: New playback state
        """
        self.playback_state_changed.emit(state)
    
    def on_position_changed(self, position: int):
        """Handle position changes.
        
        Args:
            position: New position in milliseconds
        """
        self.position_changed.emit(position)
    
    def on_duration_changed(self, duration: int):
        """Handle duration changes.
        
        Args:
            duration: New duration in milliseconds
        """
        self.duration_changed.emit(duration)
    
    def on_volume_changed(self, volume: int):
        """Handle volume changes.
        
        Args:
            volume: New volume as percentage (0-100)
        """
        self.volume_changed.emit(volume)
    
    def on_segment_started(self, segment: dict):
        """Handle segment start events.
        
        Args:
            segment: Segment information
        """
        self.segment_started.emit(segment)
    
    def on_segment_ended(self, segment: dict):
        """Handle segment end events.
        
        Args:
            segment: Segment information
        """
        self.segment_ended.emit(segment)
    
    def on_error(self, error_message: str):
        """Handle error events.
        
        Args:
            error_message: Error message
        """
        self.error_occurred.emit(error_message)

class PlayerEvents:
    """Manages events for the audio player."""
    
    def __init__(self):
        """Initialize the event system."""
        # Dictionary to store event listeners
        self.listeners: Dict[EventType, List[Callable]] = {}
        
        # Initialize listeners for each event type
        for event_type in EventType:
            self.listeners[event_type] = []
        
        logger.debug("Player events system initialized")
    
    def add_listener(self, event_type: EventType, callback: Callable) -> None:
        """Add a listener for a specific event type.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when the event occurs
        """
        if event_type not in self.listeners:
            logger.warning(f"Unknown event type: {event_type}")
            return
        
        self.listeners[event_type].append(callback)
        logger.debug(f"Added listener for {event_type.value}")
    
    def remove_listener(self, event_type: EventType, callback: Callable) -> None:
        """Remove a listener for a specific event type.
        
        Args:
            event_type: Type of event to remove listener from
            callback: Function to remove
        """
        if event_type not in self.listeners:
            logger.warning(f"Unknown event type: {event_type}")
            return
        
        if callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)
            logger.debug(f"Removed listener for {event_type.value}")
    
    def emit(self, event_type: EventType, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event to all registered listeners.
        
        Args:
            event_type: Type of event to emit
            data: Data to pass to listeners (optional)
        """
        if event_type not in self.listeners:
            logger.warning(f"Unknown event type: {event_type}")
            return
        
        if data is None:
            data = {}
        
        for callback in self.listeners[event_type]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in event listener for {event_type.value}: {str(e)}")
        
        logger.debug(f"Emitted {event_type.value} event")
    
    def clear_listeners(self, event_type: Optional[EventType] = None) -> None:
        """Clear all listeners for a specific event type or all events.
        
        Args:
            event_type: Type of event to clear listeners for (optional, clears all if None)
        """
        if event_type is None:
            # Clear all listeners
            for event_type in EventType:
                self.listeners[event_type] = []
            logger.debug("Cleared all event listeners")
        elif event_type in self.listeners:
            # Clear listeners for specific event type
            self.listeners[event_type] = []
            logger.debug(f"Cleared listeners for {event_type.value}")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def has_listeners(self, event_type: EventType) -> bool:
        """Check if an event type has any listeners.
        
        Args:
            event_type: Type of event to check
            
        Returns:
            bool: True if the event has listeners, False otherwise
        """
        if event_type not in self.listeners:
            return False
        
        return len(self.listeners[event_type]) > 0 