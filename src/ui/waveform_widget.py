"""
Waveform rendering widget for VSAT.

This module provides the WaveformWidget class that is responsible for rendering the waveform visualization.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QFontMetrics
from PyQt6.QtWidgets import QWidget

from src.ui.waveform_renderer import WaveformRenderer
from src.ui.waveform_interaction import WaveformInteraction

logger = logging.getLogger(__name__)

class WaveformWidget(QWidget):
    """Widget for rendering the waveform visualization."""
    
    # Signal emitted when the user clicks on a position in the waveform
    positionClicked = pyqtSignal(float)  # Time in seconds
    
    # Signal emitted when the user selects a range in the waveform
    rangeSelected = pyqtSignal(float, float)  # Start and end time in seconds
    
    def __init__(self, parent=None):
        """Initialize the waveform widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up widget properties
        self.setMinimumHeight(100)
        self.setMinimumWidth(300)
        self.setMouseTracking(True)
        
        # Initialize data
        self.audio_data = None
        self.sample_rate = None
        self.duration = 0.0
        self.segments = []
        self.speaker_colors = {}
        
        # Initialize view parameters
        self.zoom_level = 1.0
        self.scroll_position = 0.0
        self.selection_start = None
        self.selection_end = None
        self.current_position = None
        
        # Create renderer
        self.renderer = WaveformRenderer()
        
        # Create interaction handler
        self.interaction = WaveformInteraction(self)
        self.interaction.set_callbacks(
            position_clicked=self._on_position_clicked,
            range_selected=self._on_range_selected,
            zoom_changed=self._on_zoom_changed,
            scroll_changed=self._on_scroll_changed
        )
        
    def set_audio_data(self, audio_data: np.ndarray, sample_rate: int):
        """Set the audio data to visualize.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
        """
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        
        if audio_data is not None and sample_rate > 0:
            self.duration = len(audio_data) / sample_rate
        else:
            self.duration = 0.0
            
        # Reset view parameters
        self.zoom_level = 1.0
        self.scroll_position = 0.0
        self.selection_start = None
        self.selection_end = None
        
        self.update()
        logger.debug(f"Set audio data: {self.duration:.2f}s @ {sample_rate}Hz")
        
    def set_segments(self, segments: List[Dict[str, Any]], speaker_colors: Dict[Any, QColor]):
        """Set the speaker segments to display.
        
        Args:
            segments: List of segments with 'start', 'end', and 'speaker' keys
            speaker_colors: Dict mapping speaker IDs to colors
        """
        self.segments = segments
        self.speaker_colors = speaker_colors
        self.update()
        logger.debug(f"Set {len(segments)} segments")
        
    def set_current_position(self, position: float):
        """Set the current playback position.
        
        Args:
            position: Position in seconds
        """
        self.current_position = position
        self.update()
        
    def set_zoom_level(self, zoom_level: float):
        """Set the zoom level.
        
        Args:
            zoom_level: Zoom level (1.0 = whole file visible)
        """
        self.zoom_level = max(1.0, min(100.0, zoom_level))
        self.update()
        
    def set_scroll_position(self, position: float):
        """Set the scroll position.
        
        Args:
            position: Scroll position (0.0 to 1.0)
        """
        max_scroll = 1.0 - (1.0 / self.zoom_level)
        self.scroll_position = max(0.0, min(max_scroll, position))
        self.update()
        
    def paintEvent(self, event):
        """Handle paint events.
        
        Args:
            event: QPaintEvent
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), self.renderer.bg_color)
        
        # Skip if no audio data
        if self.audio_data is None or self.sample_rate is None:
            return
            
        # Calculate visible range in seconds
        visible_start = self.scroll_position * self.duration
        visible_duration = self.duration / self.zoom_level
        visible_end = visible_start + visible_duration
        
        # Calculate visible range in samples
        visible_start_sample = int(visible_start * self.sample_rate)
        visible_end_sample = int(visible_end * self.sample_rate)
        
        # Create drawing rect (leave space at bottom for time axis)
        main_rect = QRectF(0, 0, self.width(), self.height() - 25)
        time_rect = QRectF(0, self.height() - 25, self.width(), 25)
        
        # Draw segments first (as background)
        self.renderer.draw_segments(
            painter, main_rect, self.segments, self.speaker_colors,
            visible_start, visible_end, self.current_position
        )
        
        # Draw waveform
        self.renderer.draw_waveform(
            painter, main_rect, self.audio_data,
            visible_start_sample, visible_end_sample, self.sample_rate
        )
        
        # Draw selection
        self.renderer.draw_selection(
            painter, main_rect, self.selection_start, self.selection_end,
            visible_start, visible_end
        )
        
        # Draw position marker
        self.renderer.draw_position_marker(
            painter, main_rect, self.current_position,
            visible_start, visible_end
        )
        
        # Draw time axis
        self.renderer.draw_time_axis(
            painter, time_rect, visible_start, visible_end
        )
        
    def mousePressEvent(self, event):
        """Handle mouse press events.
        
        Args:
            event: QMouseEvent
        """
        if self.interaction.handle_mouse_press(event):
            event.accept()
        else:
            super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        """Handle mouse move events.
        
        Args:
            event: QMouseEvent
        """
        if self.interaction.handle_mouse_move(event):
            event.accept()
        else:
            super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release events.
        
        Args:
            event: QMouseEvent
        """
        if self.interaction.handle_mouse_release(event):
            event.accept()
        else:
            super().mouseReleaseEvent(event)
            
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming.
        
        Args:
            event: QWheelEvent
        """
        if self.interaction.handle_wheel(event):
            event.accept()
        else:
            super().wheelEvent(event)
            
    def _on_position_clicked(self, position: float):
        """Handle internal position clicked event.
        
        Args:
            position: Position in seconds
        """
        self.positionClicked.emit(position)
        
    def _on_range_selected(self, start: float, end: float):
        """Handle internal range selected event.
        
        Args:
            start: Start position in seconds
            end: End position in seconds
        """
        self.rangeSelected.emit(start, end)
        
    def _on_zoom_changed(self, zoom_level: float):
        """Handle zoom level change.
        
        Args:
            zoom_level: New zoom level
        """
        # Notify parent to update scrollbar
        parent = self.parent()
        if parent and hasattr(parent, 'update_scrollbar'):
            parent.update_scrollbar()
            
    def _on_scroll_changed(self, scroll_position: float):
        """Handle scroll position change.
        
        Args:
            scroll_position: New scroll position
        """
        # Notify parent to update scrollbar
        parent = self.parent()
        if parent and hasattr(parent, 'update_scrollbar'):
            parent.update_scrollbar() 