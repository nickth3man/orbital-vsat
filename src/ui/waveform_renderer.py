"""
Waveform rendering module for VSAT.

This module provides the WaveformRenderer class that is responsible for rendering
waveform visualizations. It handles the drawing of waveforms, segments, markers, etc.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QFontMetrics

logger = logging.getLogger(__name__)

class WaveformRenderer:
    """Class for rendering waveform visualizations."""
    
    def __init__(self):
        """Initialize the waveform renderer."""
        # Default colors and styles
        self.waveform_color = QColor(0, 120, 215)
        self.bg_color = QColor(240, 240, 240)
        self.text_color = QColor(30, 30, 30)
        self.marker_color = QColor(255, 0, 0)
        self.selection_color = QColor(100, 100, 255, 100)
        self.time_marker_color = QColor(150, 150, 150)
        self.segment_opacity = 0.7
        
    def draw_waveform(self, painter: QPainter, rect: QRectF, audio_data: np.ndarray, 
                     visible_start: int, visible_end: int, sample_rate: int):
        """Draw the waveform visualization.
        
        Args:
            painter: QPainter instance
            rect: Rectangle to draw in
            audio_data: Audio data as numpy array
            visible_start: Start sample
            visible_end: End sample
            sample_rate: Sample rate of the audio
        """
        if audio_data is None or len(audio_data) == 0:
            return
        
        # Compute step size based on zoom level
        num_visible_samples = visible_end - visible_start
        samples_per_pixel = max(1, num_visible_samples // int(rect.width()))
        
        # Get height and center point for drawing
        height = rect.height()
        center_y = rect.center().y()
        
        # Set up pen for drawing
        painter.setPen(QPen(self.waveform_color, 1))
        
        # Draw each pixel of the waveform
        x = rect.x()
        for i in range(visible_start, visible_end, samples_per_pixel):
            if i >= len(audio_data):
                break
                
            # Get the range of samples for this pixel
            end_idx = min(i + samples_per_pixel, len(audio_data))
            samples = audio_data[i:end_idx]
            
            # Compute min and max values for the sample range
            if len(samples) > 0:
                min_val = np.min(samples)
                max_val = np.max(samples)
                
                # Scale to the view height
                y1 = center_y - (max_val * height / 2)
                y2 = center_y - (min_val * height / 2)
                
                # Draw the line
                painter.drawLine(QPointF(x, y1), QPointF(x, y2))
            
            x += 1
            if x > rect.right():
                break
                
    def draw_segments(self, painter: QPainter, rect: QRectF, segments: List[Dict[str, Any]], 
                     speaker_colors: Dict[Any, QColor], visible_start: float, visible_end: float,
                     current_position: float = None):
        """Draw speaker segments on the waveform.
        
        Args:
            painter: QPainter instance
            rect: Rectangle to draw in
            segments: List of segments with 'start', 'end', and 'speaker' keys
            speaker_colors: Dict mapping speaker IDs to colors
            visible_start: Start time in seconds
            visible_end: End time in seconds
            current_position: Current playback position in seconds (optional)
        """
        if not segments:
            return
            
        # Set up font for speaker labels
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        
        # Compute scaling
        time_width = visible_end - visible_start
        width = rect.width()
        scale = width / time_width
        
        # Draw each segment
        for segment in segments:
            # Skip segments outside the visible range
            if segment['end'] < visible_start or segment['start'] > visible_end:
                continue
                
            # Get the color for this speaker
            speaker = segment.get('speaker', 'unknown')
            color = speaker_colors.get(speaker, QColor(150, 150, 150))
            
            # Compute segment position and width
            start_x = rect.x() + max(0, (segment['start'] - visible_start) * scale)
            end_x = rect.x() + min(width, (segment['end'] - visible_start) * scale)
            
            # Draw segment background
            segment_rect = QRectF(start_x, rect.y(), end_x - start_x, rect.height())
            alpha_color = QColor(color)
            alpha_color.setAlphaF(self.segment_opacity)
            painter.fillRect(segment_rect, alpha_color)
            
            # Draw segment borders
            painter.setPen(QPen(color.darker(120), 1))
            painter.drawRect(segment_rect)
            
            # Draw speaker label
            text_rect = QRectF(segment_rect)
            text_rect.setHeight(20)
            metrics = QFontMetrics(font)
            text = str(speaker)
            text_width = metrics.horizontalAdvance(text)
            
            if text_width + 10 < segment_rect.width():
                painter.setPen(QPen(self.text_color, 1))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)
                
    def draw_selection(self, painter: QPainter, rect: QRectF, 
                      selection_start: float, selection_end: float,
                      visible_start: float, visible_end: float):
        """Draw the selection overlay.
        
        Args:
            painter: QPainter instance
            rect: Rectangle to draw in
            selection_start: Start time of selection in seconds
            selection_end: End time of selection in seconds
            visible_start: Start time in seconds of visible range
            visible_end: End time in seconds of visible range
        """
        if selection_start is None or selection_end is None:
            return
            
        # Ensure selection bounds are in order
        start = min(selection_start, selection_end)
        end = max(selection_start, selection_end)
        
        # Skip if selection is outside visible range
        if end < visible_start or start > visible_end:
            return
            
        # Clip to visible range
        start = max(start, visible_start)
        end = min(end, visible_end)
        
        # Compute position and width
        time_width = visible_end - visible_start
        width = rect.width()
        scale = width / time_width
        
        start_x = rect.x() + (start - visible_start) * scale
        end_x = rect.x() + (end - visible_start) * scale
        
        # Draw selection rectangle
        selection_rect = QRectF(start_x, rect.y(), end_x - start_x, rect.height())
        painter.fillRect(selection_rect, self.selection_color)
        painter.setPen(QPen(self.selection_color.darker(120), 1))
        painter.drawRect(selection_rect)
        
    def draw_position_marker(self, painter: QPainter, rect: QRectF, 
                           position: float, visible_start: float, visible_end: float):
        """Draw the current position marker.
        
        Args:
            painter: QPainter instance
            rect: Rectangle to draw in
            position: Current position in seconds
            visible_start: Start time in seconds of visible range
            visible_end: End time in seconds of visible range
        """
        if position is None:
            return
            
        # Skip if position is outside visible range
        if position < visible_start or position > visible_end:
            return
            
        # Compute position
        time_width = visible_end - visible_start
        width = rect.width()
        scale = width / time_width
        
        pos_x = rect.x() + (position - visible_start) * scale
        
        # Draw position line
        painter.setPen(QPen(self.marker_color, 2))
        painter.drawLine(QPointF(pos_x, rect.top()), QPointF(pos_x, rect.bottom()))
        
    def draw_time_axis(self, painter: QPainter, rect: QRectF, 
                      visible_start: float, visible_end: float):
        """Draw time markers on the x-axis.
        
        Args:
            painter: QPainter instance
            rect: Rectangle to draw in
            visible_start: Start time in seconds of visible range
            visible_end: End time in seconds of visible range
        """
        # Determine appropriate time step based on visible duration
        duration = visible_end - visible_start
        
        if duration <= 5:
            step = 0.5  # 0.5 seconds
            minor_step = 0.1
            format_str = "{:.1f}"
        elif duration <= 30:
            step = 1.0  # 1 second
            minor_step = 0.5
            format_str = "{:.1f}"
        elif duration <= 60:
            step = 5.0  # 5 seconds
            minor_step = 1.0
            format_str = "{:.0f}"
        elif duration <= 300:
            step = 10.0  # 10 seconds
            minor_step = 5.0
            format_str = "{:.0f}"
        else:
            step = 30.0  # 30 seconds
            minor_step = 10.0
            format_str = "{:.0f}"
        
        # Set up font
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        
        # Compute scaling
        width = rect.width()
        scale = width / duration
        
        # Draw minor time markers
        start_time = int(visible_start / minor_step) * minor_step
        painter.setPen(QPen(self.time_marker_color.lighter(120), 0.5))
        
        for t in np.arange(start_time, visible_end, minor_step):
            if t < visible_start:
                continue
                
            x = rect.x() + (t - visible_start) * scale
            painter.drawLine(QPointF(x, rect.bottom() - 3), QPointF(x, rect.bottom()))
        
        # Draw major time markers and labels
        start_time = int(visible_start / step) * step
        painter.setPen(QPen(self.time_marker_color, 1))
        
        for t in np.arange(start_time, visible_end, step):
            if t < visible_start:
                continue
                
            x = rect.x() + (t - visible_start) * scale
            painter.drawLine(QPointF(x, rect.bottom() - 5), QPointF(x, rect.bottom()))
            
            # Draw time label
            text = format_str.format(t)
            metrics = QFontMetrics(font)
            text_width = metrics.horizontalAdvance(text)
            
            text_rect = QRectF(x - text_width / 2, rect.bottom() - 20, text_width, 15)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text) 