"""
Waveform interaction module for VSAT.

This module provides the WaveformInteraction class that handles user interaction
with the waveform display including mouse events, zooming, and selection.
"""

import logging
from typing import Optional, Callable, Tuple

from PyQt6.QtCore import Qt, QPointF, QEvent, QObject
from PyQt6.QtGui import QMouseEvent, QWheelEvent

logger = logging.getLogger(__name__)

class WaveformInteraction(QObject):
    """Handles user interaction with the waveform widget."""
    
    def __init__(self, widget, parent=None):
        """Initialize the interaction handler.
        
        Args:
            widget: The WaveformWidget instance this handler is for
            parent: Parent QObject
        """
        super().__init__(parent)
        self.widget = widget
        
        # Mouse state
        self.is_dragging = False
        self.is_selecting = False
        self.last_mouse_pos = None
        self.drag_start_time = None
        
        # Callbacks
        self.on_position_clicked = None
        self.on_range_selected = None
        self.on_zoom_changed = None
        self.on_scroll_changed = None
        
    def set_callbacks(self, position_clicked: Callable[[float], None],
                     range_selected: Callable[[float, float], None],
                     zoom_changed: Callable[[float], None],
                     scroll_changed: Callable[[float], None]):
        """Set the callback functions for various interactions.
        
        Args:
            position_clicked: Function to call when position is clicked
            range_selected: Function to call when range is selected
            zoom_changed: Function to call when zoom level changes
            scroll_changed: Function to call when scroll position changes
        """
        self.on_position_clicked = position_clicked
        self.on_range_selected = range_selected
        self.on_zoom_changed = zoom_changed
        self.on_scroll_changed = scroll_changed
        
    def get_time_at_position(self, x: float) -> float:
        """Convert screen X position to time in seconds.
        
        Args:
            x: X position in widget coordinates
            
        Returns:
            float: Time in seconds
        """
        # Get the visible time range
        visible_start = self.widget.scroll_position * self.widget.duration
        visible_duration = self.widget.duration / self.widget.zoom_level
        visible_end = visible_start + visible_duration
        
        # Convert position to time
        time_width = visible_end - visible_start
        width = self.widget.width()
        
        # Protect against division by zero
        if width == 0:
            return visible_start
            
        rel_x = (x - 0) / width
        return visible_start + rel_x * time_width
        
    def handle_mouse_press(self, event: QMouseEvent) -> bool:
        """Handle mouse press events.
        
        Args:
            event: QMouseEvent
            
        Returns:
            bool: True if event was handled, False otherwise
        """
        if event.button() == Qt.MouseButton.LeftButton:
            # Store mouse position
            self.last_mouse_pos = event.position()
            
            # Convert to time
            click_time = self.get_time_at_position(event.position().x())
            
            # If Ctrl is pressed, start selection
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.is_selecting = True
                self.widget.selection_start = click_time
                self.widget.selection_end = click_time
            else:
                # Regular click sets position
                self.is_dragging = True
                self.drag_start_time = click_time
                
                # Emit signal
                if self.on_position_clicked:
                    self.on_position_clicked(click_time)
                
            return True
            
        return False
        
    def handle_mouse_move(self, event: QMouseEvent) -> bool:
        """Handle mouse move events.
        
        Args:
            event: QMouseEvent
            
        Returns:
            bool: True if event was handled, False otherwise
        """
        if self.is_selecting and event.buttons() & Qt.MouseButton.LeftButton:
            # Update selection end point
            self.widget.selection_end = self.get_time_at_position(event.position().x())
            self.widget.update()
            return True
            
        elif self.is_dragging and event.buttons() & Qt.MouseButton.LeftButton:
            # Calculate deltas
            delta_x = event.position().x() - self.last_mouse_pos.x()
            
            # Update scroll position based on drag
            visible_duration = self.widget.duration / self.widget.zoom_level
            delta_time = -delta_x / self.widget.width() * visible_duration
            
            new_scroll = self.widget.scroll_position + delta_time / self.widget.duration
            
            # Constrain scroll position
            max_scroll = 1.0 - (1.0 / self.widget.zoom_level)
            new_scroll = max(0.0, min(max_scroll, new_scroll))
            
            # Update scroll position
            if new_scroll != self.widget.scroll_position:
                self.widget.scroll_position = new_scroll
                if self.on_scroll_changed:
                    self.on_scroll_changed(new_scroll)
                self.widget.update()
            
            # Store new position
            self.last_mouse_pos = event.position()
            return True
            
        return False
        
    def handle_mouse_release(self, event: QMouseEvent) -> bool:
        """Handle mouse release events.
        
        Args:
            event: QMouseEvent
            
        Returns:
            bool: True if event was handled, False otherwise
        """
        if event.button() == Qt.MouseButton.LeftButton:
            if self.is_selecting:
                # Emit selection completed signal
                if self.widget.selection_start is not None and self.widget.selection_end is not None:
                    start = min(self.widget.selection_start, self.widget.selection_end)
                    end = max(self.widget.selection_start, self.widget.selection_end)
                    
                    # Only emit if selection is large enough (more than 10ms)
                    if end - start > 0.01 and self.on_range_selected:
                        self.on_range_selected(start, end)
                
                self.is_selecting = False
                return True
                
            elif self.is_dragging:
                self.is_dragging = False
                return True
                
        return False
        
    def handle_wheel(self, event: QWheelEvent) -> bool:
        """Handle mouse wheel events for zooming.
        
        Args:
            event: QWheelEvent
            
        Returns:
            bool: True if event was handled, False otherwise
        """
        # Get wheel delta
        delta = event.angleDelta().y()
        
        if delta != 0:
            # Find the time position under the cursor
            cursor_pos_x = event.position().x()
            time_at_cursor = self.get_time_at_position(cursor_pos_x)
            
            # Calculate relative position (0.0 to 1.0)
            visible_start = self.widget.scroll_position * self.widget.duration
            visible_duration = self.widget.duration / self.widget.zoom_level
            rel_pos = (time_at_cursor - visible_start) / visible_duration
            
            # Calculate new zoom level
            zoom_factor = 1.2 if delta > 0 else 1 / 1.2
            new_zoom = self.widget.zoom_level * zoom_factor
            
            # Constrain zoom level (1.0 = full view, 100.0 = max zoom)
            new_zoom = max(1.0, min(100.0, new_zoom))
            
            # Calculate new scroll position to keep time point under cursor
            center_pos = time_at_cursor / self.widget.duration
            new_scroll = center_pos - rel_pos / new_zoom
            
            # Clamp scroll position to valid range
            max_scroll = 1.0 - (1.0 / new_zoom)
            new_scroll = max(0.0, min(max_scroll, new_scroll))
            
            # Only update if changed
            if new_zoom != self.widget.zoom_level:
                # Update parameters
                self.widget.zoom_level = new_zoom
                self.widget.scroll_position = new_scroll
                
                # Notify change
                if self.on_zoom_changed:
                    self.on_zoom_changed(new_zoom)
                if self.on_scroll_changed:
                    self.on_scroll_changed(new_scroll)
                
                # Redraw
                self.widget.update()
                
            return True
            
        return False 