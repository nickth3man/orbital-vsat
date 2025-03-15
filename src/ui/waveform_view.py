"""
Waveform visualization widget for VSAT.

This module provides a PyQt widget for visualizing audio waveforms with speaker coloring.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QFontMetrics
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollBar

from src.ui.waveform_widget import WaveformWidget

logger = logging.getLogger(__name__)

class WaveformView(QWidget):
    """Widget for visualizing audio waveforms with speaker coloring."""
    
    # Signal emitted when the user clicks on a position in the waveform
    positionClicked = pyqtSignal(float)  # Time in seconds
    
    # Signal emitted when the user selects a range in the waveform
    rangeSelected = pyqtSignal(float, float)  # Start and end time in seconds
    
    def __init__(self, parent=None):
        """Initialize the waveform view widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up widget properties
        self.setMinimumHeight(100)
        self.setMinimumWidth(300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
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
        
        # Initialize UI components
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create waveform widget
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.positionClicked.connect(self.on_position_clicked)
        self.waveform_widget.rangeSelected.connect(self.on_range_selected)
        layout.addWidget(self.waveform_widget)
        
        # Create horizontal scrollbar
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.scrollbar.setRange(0, 1000)
        self.scrollbar.setPageStep(100)
        self.scrollbar.valueChanged.connect(self.on_scroll_changed)
        layout.addWidget(self.scrollbar)
    
    def set_audio_data(self, audio_data: np.ndarray, sample_rate: int):
        """Set the audio data to display.
        
        Args:
            audio_data: Audio data as a numpy array
            sample_rate: Sample rate of the audio data
        """
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.duration = len(audio_data) / sample_rate
        
        # Reset view parameters
        self.zoom_level = 1.0
        self.scroll_position = 0.0
        self.selection_start = None
        self.selection_end = None
        
        # Update waveform widget
        self.waveform_widget.set_audio_data(audio_data, sample_rate)
        
        # Update scrollbar
        self.update_scrollbar()
    
    def set_segments(self, segments: List[Dict[str, Any]]):
        """Set the segments to display.
        
        Args:
            segments: List of segments with speaker information
        """
        self.segments = segments
        
        # Generate colors for speakers
        speaker_ids = set(segment['speaker'] for segment in segments)
        colors = generate_speaker_colors(len(speaker_ids))
        self.speaker_colors = dict(zip(speaker_ids, colors))
        
        # Update waveform widget
        self.waveform_widget.set_segments(segments, self.speaker_colors)
    
    def set_current_position(self, position: float):
        """Set the current playback position.
        
        Args:
            position: Position in seconds
        """
        # Update waveform widget
        self.waveform_widget.set_current_position(position)
        
        # Ensure position is visible
        self.ensure_position_visible(position)
    
    def set_zoom_level(self, zoom_level: float):
        """Set the zoom level.
        
        Args:
            zoom_level: Zoom level (1.0 = 100%)
        """
        # Clamp zoom level
        zoom_level = max(0.1, min(10.0, zoom_level))
        
        if zoom_level != self.zoom_level:
            self.zoom_level = zoom_level
            
            # Update waveform widget
            self.waveform_widget.set_zoom_level(zoom_level)
            
            # Update scrollbar
            self.update_scrollbar()
    
    def zoom_in(self):
        """Increase the zoom level."""
        self.set_zoom_level(self.zoom_level * 1.2)
    
    def zoom_out(self):
        """Decrease the zoom level."""
        self.set_zoom_level(self.zoom_level / 1.2)
    
    def ensure_position_visible(self, position: float):
        """Ensure that a position is visible in the view.
        
        Args:
            position: Position in seconds
        """
        if not self.duration:
            return
        
        # Normalize position to 0-1 range
        normalized_pos = position / self.duration
        
        # Calculate visible range
        visible_width = 1.0 / self.zoom_level
        visible_start = self.scroll_position
        visible_end = visible_start + visible_width
        
        # Check if position is outside visible range
        if normalized_pos < visible_start or normalized_pos > visible_end:
            # Center position in view
            new_scroll_position = normalized_pos - (visible_width / 2)
            
            # Clamp scroll position
            new_scroll_position = max(0.0, min(1.0 - visible_width, new_scroll_position))
            
            # Update scroll position
            self.scroll_position = new_scroll_position
            self.update_scrollbar()
            
            # Update waveform widget
            self.waveform_widget.set_scroll_position(new_scroll_position)
    
    def update_scrollbar(self):
        """Update the scrollbar based on current zoom and scroll position."""
        if not self.duration:
            return
        
        # Calculate scrollbar parameters
        visible_width = 1.0 / self.zoom_level
        max_scroll = 1.0 - visible_width
        
        # Update scrollbar range
        self.scrollbar.blockSignals(True)
        self.scrollbar.setMaximum(int(max_scroll * 1000))
        self.scrollbar.setValue(int(self.scroll_position * 1000))
        self.scrollbar.setPageStep(int(visible_width * 1000))
        self.scrollbar.blockSignals(False)
    
    @pyqtSlot(int)
    def on_scroll_changed(self, value: int):
        """Handle scrollbar value changes.
        
        Args:
            value: Scrollbar value
        """
        # Calculate scroll position
        self.scroll_position = value / 1000.0
        
        # Update waveform widget
        self.waveform_widget.set_scroll_position(self.scroll_position)
    
    @pyqtSlot(float)
    def on_position_clicked(self, position: float):
        """Handle position clicks in the waveform widget.
        
        Args:
            position: Position in seconds
        """
        # Emit position clicked signal
        self.positionClicked.emit(position)
    
    @pyqtSlot(float, float)
    def on_range_selected(self, start: float, end: float):
        """Handle range selection in the waveform widget.
        
        Args:
            start: Start position in seconds
            end: End position in seconds
        """
        # Store selection range
        self.selection_start = start
        self.selection_end = end
        
        # Emit range selected signal
        self.rangeSelected.emit(start, end)
    
    def has_selection(self) -> bool:
        """Check if there is a selection.
        
        Returns:
            True if there is a selection, False otherwise
        """
        return self.selection_start is not None and self.selection_end is not None
    
    def get_selection_range(self) -> Tuple[Optional[float], Optional[float]]:
        """Get the current selection range.
        
        Returns:
            Tuple of (start, end) positions in seconds, or (None, None) if no selection
        """
        return self.selection_start, self.selection_end
    
    def set_selection(self, start: float, end: float):
        """Set the selection range.
        
        Args:
            start: Start position in seconds
            end: End position in seconds
        """
        self.selection_start = start
        self.selection_end = end
        
        # Update waveform widget
        self.waveform_widget.update()
        
        # Ensure selection is visible
        self.ensure_position_visible(start)

def generate_speaker_colors(num_speakers: int) -> List[QColor]:
    """Generate a list of distinct colors for speakers.
    
    Args:
        num_speakers: Number of speakers
        
    Returns:
        List of QColor objects
    """
    colors = []
    
    # Use HSV color space for better control over color generation
    hue_step = 360 / num_speakers
    
    for i in range(num_speakers):
        hue = (i * hue_step) % 360
        color = QColor()
        color.setHsv(int(hue), 200, 230)  # Moderately saturated and bright
        colors.append(color)
    
    return colors 