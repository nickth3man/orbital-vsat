"""
Search result components for VSAT.

This module provides UI components for displaying search results.
"""

import logging
from typing import Dict, Any, Optional, List

from PyQt6.QtWidgets import (
    QWidget, 
    QLabel, 
    QHBoxLayout, 
    QVBoxLayout, 
    QPushButton,
    QFrame,
    QToolButton,
    QGridLayout,
    QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette

logger = logging.getLogger(__name__)

class SearchResultItem(QWidget):
    """Widget for displaying a single search result item."""
    
    def __init__(self, text: str, time: str, parent=None):
        """Initialize the search result item.
        
        Args:
            text: Text to display
            time: Time to display
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Create text label
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        layout.addWidget(self.text_label, 1)
        
        # Create time label
        self.time_label = QLabel(time)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.time_label)
        
        # Set accessibility properties
        self.setAccessibleName(f"Search Result: {text}")
        self.text_label.setAccessibleName("Result Text")
        self.time_label.setAccessibleName("Result Time")

class SearchResult(QWidget):
    """Widget for displaying a search result."""
    
    # Signal emitted when the result is clicked
    result_clicked = pyqtSignal(dict)
    
    def __init__(self, result: Dict[str, Any], index: int, parent=None):
        """Initialize the search result.
        
        Args:
            result: Search result data
            index: Result index
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Store result data
        self.result = result
        self.index = index
        
        # Create main frame with border
        self.main_frame = QFrame(self)
        self.main_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.main_frame.setFrameShadow(QFrame.Shadow.Raised)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 5)
        main_layout.addWidget(self.main_frame)
        
        # Create layout for the frame
        layout = QVBoxLayout(self.main_frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Create header grid layout for better organization
        header_grid = QGridLayout()
        header_grid.setColumnStretch(2, 1)  # Make the third column stretch
        
        # Result number with speaker color indicator
        speaker = result.get("speaker", "Unknown")
        speaker_color = self._get_speaker_color(speaker)
        
        # Result number
        self.number_label = QLabel(f"Result #{index}")
        font = QFont()
        font.setBold(True)
        self.number_label.setFont(font)
        
        # Create colored speaker indicator
        self.speaker_indicator = QFrame()
        self.speaker_indicator.setFixedSize(16, 16)
        self.speaker_indicator.setStyleSheet(f"background-color: {speaker_color}; border-radius: 8px;")
        
        # Add number and speaker indicator to a horizontal layout
        number_layout = QHBoxLayout()
        number_layout.setSpacing(8)
        number_layout.addWidget(self.number_label)
        number_layout.addWidget(self.speaker_indicator)
        number_layout.addStretch()
        
        header_grid.addLayout(number_layout, 0, 0)
        
        # Create speaker label
        self.speaker_label = QLabel(f"Speaker: {speaker}")
        header_grid.addWidget(self.speaker_label, 0, 1)
        
        # Create time label with more detailed formatting
        start = result.get("start", 0)
        end = result.get("end", 0)
        duration = end - start
        time_str = f"{self._format_time(start)} - {self._format_time(end)} ({duration:.2f}s)"
        self.time_label = QLabel(time_str)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header_grid.addWidget(self.time_label, 0, 2)
        
        layout.addLayout(header_grid)
        
        # Add a separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        
        # Get the surrounding text with more context
        context = result.get("context", "")
        # Replace markdown-style highlight with HTML for better visibility
        context_html = context.replace("**", "<span style='background-color: #FFFF00; color: #000000;'>", 1)
        context_html = context_html.replace("**", "</span>", 1)
        
        self.context_label = QLabel(f"<p style='line-height: 130%;'>{context_html}</p>")
        self.context_label.setWordWrap(True)
        self.context_label.setTextFormat(Qt.TextFormat.RichText)
        self.context_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout.addWidget(self.context_label)
        
        # Create button layout
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 5, 0, 0)
        
        # Create play button
        self.play_button = QPushButton("Play")
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_button.clicked.connect(self._on_play_clicked)
        button_layout.addWidget(self.play_button)
        
        # Create jump to button
        self.jump_button = QPushButton("Jump to")
        self.jump_button.setIcon(QIcon.fromTheme("go-jump"))
        self.jump_button.clicked.connect(self._on_jump_clicked)
        button_layout.addWidget(self.jump_button)
        
        # Create copy text button
        self.copy_button = QPushButton("Copy Text")
        self.copy_button.setIcon(QIcon.fromTheme("edit-copy"))
        self.copy_button.clicked.connect(self._on_copy_clicked)
        button_layout.addWidget(self.copy_button)
        
        # Add stretch to push buttons to the left
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Set styling
        self.setStyleSheet("""
            QFrame.StyledPanel {
                background-color: #f8f8f8;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #e8e8e8;
                border: 1px solid #d0d0d0;
                padding: 5px 12px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d8d8d8;
                border: 1px solid #c0c0c0;
            }
            QPushButton:pressed {
                background-color: #c8c8c8;
            }
            QLabel {
                color: #303030;
            }
        """)
        
        # Set accessibility properties
        self.setAccessibleName(f"Search Result {index}")
        self.number_label.setAccessibleName("Result Number")
        self.speaker_label.setAccessibleName("Speaker")
        self.time_label.setAccessibleName("Time Range")
        self.context_label.setAccessibleName("Context")
        self.play_button.setAccessibleName("Play Button")
        self.jump_button.setAccessibleName("Jump To Button")
        self.copy_button.setAccessibleName("Copy Text Button")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS.MS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 100)
        return f"{minutes:02d}:{secs:02d}.{ms:02d}"
    
    def _get_speaker_color(self, speaker: str) -> str:
        """Get color for speaker.
        
        Args:
            speaker: Speaker name
            
        Returns:
            Hex color string
        """
        # Define colors for speakers (can be expanded)
        speaker_colors = {
            "Speaker 1": "#FF6B6B",
            "Speaker 2": "#4ECDC4",
            "Speaker 3": "#FFD166",
            "Speaker 4": "#6B5B95",
            "Speaker 5": "#88D8B0",
            "Unknown": "#CCCCCC"
        }
        
        # Use speaker-specific color or a default
        if speaker in speaker_colors:
            return speaker_colors[speaker]
        
        # Generate a color based on the speaker name if not in predefined list
        import hashlib
        hash_val = int(hashlib.md5(speaker.encode()).hexdigest(), 16)
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        
        # Ensure colors are not too light
        r = min(max(r, 50), 220)
        g = min(max(g, 50), 220)
        b = min(max(b, 50), 220)
        
        return f"#{r:02X}{g:02X}{b:02X}"
    
    def _on_play_clicked(self):
        """Handle play button click."""
        self.result_clicked.emit(self.result)
        logger.info(f"Play clicked for result {self.index}")
    
    def _on_jump_clicked(self):
        """Handle jump to button click."""
        self.result_clicked.emit(self.result)
        logger.info(f"Jump to clicked for result {self.index}")
    
    def _on_copy_clicked(self):
        """Handle copy text button click."""
        from PyQt6.QtGui import QGuiApplication
        
        # Get the matched text
        text = self.result.get("text", "")
        QGuiApplication.clipboard().setText(text)
        
        logger.info(f"Copied text: {text}") 