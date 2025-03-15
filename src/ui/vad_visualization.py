"""
Voice Activity Detection visualization for VSAT.

This module provides UI components for visualizing speech segments detected by the VAD module.
"""

import logging
from typing import List, Dict, Optional, Callable

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QProgressBar, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush

from src.audio.processor import AudioProcessor

logger = logging.getLogger(__name__)

class SpeechSegmentWidget(QWidget):
    """Widget for displaying speech segments on a timeline."""
    
    segmentClicked = pyqtSignal(dict)  # Signal emitted when a segment is clicked
    
    def __init__(self, parent=None):
        """Initialize the speech segment widget."""
        super().__init__(parent)
        self.segments = []
        self.total_duration = 0
        self.setMinimumHeight(100)
        self.setMinimumWidth(600)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.GlobalColor.white)
        self.setPalette(palette)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
        # Current hover segment
        self.hover_segment_index = -1
    
    def set_segments(self, segments: List[Dict], total_duration: float):
        """Set the speech segments to display.
        
        Args:
            segments: List of speech segments with start/end times
            total_duration: Total duration of the audio in seconds
        """
        self.segments = segments
        self.total_duration = total_duration
        self.update()
    
    def paintEvent(self, event):
        """Paint the speech segments on the timeline."""
        if not self.segments or self.total_duration <= 0:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw timeline
        width = self.width()
        height = self.height()
        
        # Draw background
        painter.fillRect(0, 0, width, height, QColor(240, 240, 240))
        
        # Draw segments
        for i, segment in enumerate(self.segments):
            start_pos = int((segment["start"] / self.total_duration) * width)
            end_pos = int((segment["end"] / self.total_duration) * width)
            
            # Calculate color based on confidence
            confidence = segment.get("confidence", 0.5)
            green = int(200 * confidence)
            
            # Highlight hovered segment
            if i == self.hover_segment_index:
                painter.setBrush(QBrush(QColor(0, green, 100, 200)))
                painter.setPen(QPen(QColor(0, 100, 0), 2))
            else:
                painter.setBrush(QBrush(QColor(0, green, 0, 150)))
                painter.setPen(QPen(QColor(0, 100, 0), 1))
            
            # Draw segment rectangle
            painter.drawRect(QRectF(start_pos, 10, max(2, end_pos - start_pos), height - 20))
            
            # Draw confidence text if segment is wide enough
            if end_pos - start_pos > 40:
                painter.setPen(QPen(QColor(0, 0, 0)))
                painter.drawText(
                    QRectF(start_pos + 5, 10, end_pos - start_pos - 10, height - 20),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{confidence:.2f}"
                )
        
        # Draw time markers
        painter.setPen(QPen(QColor(100, 100, 100)))
        marker_interval = max(1, int(self.total_duration / 10))  # At most 10 markers
        
        for t in range(0, int(self.total_duration) + 1, marker_interval):
            x_pos = int((t / self.total_duration) * width)
            painter.drawLine(x_pos, height - 5, x_pos, height)
            painter.drawText(x_pos - 15, height - 8, 30, 20, Qt.AlignmentFlag.AlignCenter, f"{t}s")
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for hover effects."""
        x = event.position().x()
        width = self.width()
        
        # Find segment under cursor
        hover_index = -1
        for i, segment in enumerate(self.segments):
            start_pos = int((segment["start"] / self.total_duration) * width)
            end_pos = int((segment["end"] / self.total_duration) * width)
            
            if start_pos <= x <= end_pos:
                hover_index = i
                break
        
        # Update if changed
        if hover_index != self.hover_segment_index:
            self.hover_segment_index = hover_index
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse click events to emit segment clicked signal."""
        if self.hover_segment_index >= 0:
            self.segmentClicked.emit(self.segments[self.hover_segment_index])

class VADVisualizationWidget(QWidget):
    """Widget for visualizing Voice Activity Detection results."""
    
    def __init__(self, audio_processor: AudioProcessor, parent=None):
        """Initialize the VAD visualization widget.
        
        Args:
            audio_processor: AudioProcessor instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.audio_processor = audio_processor
        self.current_file = None
        self.segments = []
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        
        # File info section
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File:"))
        self.file_label = QLabel("No file loaded")
        file_layout.addWidget(self.file_label, 1)
        layout.addLayout(file_layout)
        
        # Sensitivity preset section
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("Sensitivity:"))
        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItems(self.audio_processor.get_vad_sensitivity_presets())
        self.sensitivity_combo.setCurrentText("medium")
        sensitivity_layout.addWidget(self.sensitivity_combo)
        
        self.detect_button = QPushButton("Detect Speech")
        self.detect_button.clicked.connect(self.detect_speech)
        sensitivity_layout.addWidget(self.detect_button)
        layout.addLayout(sensitivity_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Speech segments visualization
        self.segment_widget = SpeechSegmentWidget()
        self.segment_widget.segmentClicked.connect(self.on_segment_clicked)
        
        # Add segment widget to a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.segment_widget)
        layout.addWidget(scroll_area, 1)
        
        # Statistics section
        stats_frame = QFrame()
        stats_frame.setFrameShape(QFrame.Shape.StyledPanel)
        stats_layout = QVBoxLayout(stats_frame)
        
        self.stats_label = QLabel("No statistics available")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_frame)
        
        # Selected segment info
        segment_info_frame = QFrame()
        segment_info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        segment_info_layout = QVBoxLayout(segment_info_frame)
        
        self.segment_info_label = QLabel("Click on a segment to see details")
        segment_info_layout.addWidget(self.segment_info_label)
        
        layout.addWidget(segment_info_frame)
        
        self.setLayout(layout)
    
    def set_file(self, file_path: str):
        """Set the audio file to visualize.
        
        Args:
            file_path: Path to the audio file
        """
        self.current_file = file_path
        self.file_label.setText(file_path)
        self.segments = []
        self.segment_widget.set_segments([], 0)
        self.stats_label.setText("No statistics available")
        self.segment_info_label.setText("Click on a segment to see details")
    
    def detect_speech(self):
        """Detect speech segments in the current file."""
        if not self.current_file:
            logger.warning("No file loaded")
            return
        
        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        try:
            # Get selected sensitivity preset
            sensitivity = self.sensitivity_combo.currentText()
            
            # Detect speech segments
            self.segments = self.audio_processor.detect_speech_segments(
                self.current_file,
                sensitivity_preset=sensitivity,
                progress_callback=self.update_progress
            )
            
            # Get audio duration
            audio_data, sample_rate = self.audio_processor.file_handler.load_file(self.current_file)
            total_duration = len(audio_data) / sample_rate
            
            # Update visualization
            self.segment_widget.set_segments(self.segments, total_duration)
            
            # Update statistics
            stats = self.audio_processor.get_speech_statistics(self.current_file)
            stats_text = (
                f"Speech segments: {stats['speech_count']}\n"
                f"Speech percentage: {stats['speech_percentage']:.1f}%\n"
                f"Total speech duration: {stats['total_speech_duration']:.2f}s\n"
                f"Average speech duration: {stats['avg_speech_duration']:.2f}s\n"
                f"Average confidence: {stats['avg_confidence']:.2f}"
            )
            self.stats_label.setText(stats_text)
            
        except Exception as e:
            logger.error(f"Error detecting speech: {e}")
            self.stats_label.setText(f"Error: {str(e)}")
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
    
    def update_progress(self, progress: float):
        """Update progress bar.
        
        Args:
            progress: Progress value (0-1)
        """
        self.progress_bar.setValue(int(progress * 100))
    
    def on_segment_clicked(self, segment: Dict):
        """Handle segment click event.
        
        Args:
            segment: Speech segment that was clicked
        """
        # Update segment info
        info_text = (
            f"Start: {segment['start']:.2f}s\n"
            f"End: {segment['end']:.2f}s\n"
            f"Duration: {segment['duration']:.2f}s\n"
            f"Confidence: {segment['confidence']:.2f}"
        )
        self.segment_info_label.setText(info_text) 