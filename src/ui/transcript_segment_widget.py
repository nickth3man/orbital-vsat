"""
Transcript segment widget for VSAT.

This module provides a PyQt widget for displaying and interacting with a transcript segment.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Callable

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QSizePolicy
)

from src.ui.flow_layout import FlowLayout

logger = logging.getLogger(__name__)

class TranscriptSegmentWidget(QWidget):
    """Widget for displaying a transcript segment."""
    
    # Signal emitted when a word is clicked
    wordClicked = pyqtSignal(dict)  # Word data
    
    # Signal emitted when words are selected
    wordsSelected = pyqtSignal(list)  # List of word data
    
    def __init__(self, segment: Dict[str, Any], speaker_color: QColor = None, parent=None):
        """Initialize the transcript segment widget.
        
        Args:
            segment: Segment data
            speaker_color: Color for the speaker
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Store segment data
        self.segment = segment
        self.speaker_color = speaker_color
        self.words = segment.get('words', [])
        self.word_buttons = []
        self.is_expanded = False
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components."""
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # Create header
        header_layout = QHBoxLayout()
        
        # Speaker label
        speaker_text = ""
        if 'speaker' in self.segment and self.segment['speaker']:
            speaker_name = self.segment.get('speaker_name', f"Speaker {self.segment['speaker']}")
            speaker_text = f"{speaker_name}: "
        
        self.speaker_label = QLabel(speaker_text)
        if self.speaker_color:
            self.speaker_label.setStyleSheet(f"color: {self.speaker_color.name()}")
        header_layout.addWidget(self.speaker_label)
        
        # Time label
        if 'start' in self.segment and 'end' in self.segment:
            start = self.format_time(self.segment['start'])
            end = self.format_time(self.segment['end'])
            time_text = f"[{start} - {end}]"
            
            time_label = QLabel(time_text)
            time_label.setStyleSheet("color: gray")
            header_layout.addWidget(time_label)
        
        # Add stretch to push expand button to the right
        header_layout.addStretch()
        
        # Expand button
        self.expand_button = QPushButton("▼")
        self.expand_button.setFixedSize(24, 24)
        self.expand_button.clicked.connect(self.toggle_expand)
        header_layout.addWidget(self.expand_button)
        
        self.layout.addLayout(header_layout)
        
        # Create word layout
        self.word_layout = QHBoxLayout()
        self.word_layout.setContentsMargins(10, 0, 10, 0)
        
        # Add text label (will be replaced by word buttons when expanded)
        self.text_label = QLabel(self.segment.get('text', ''))
        self.text_label.setWordWrap(True)
        self.word_layout.addWidget(self.text_label)
        
        self.layout.addLayout(self.word_layout)
        
        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(line)
    
    def format_time(self, seconds: float) -> str:
        """Format time in seconds to a string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted time string (MM:SS.ms)
        """
        minutes = int(seconds // 60)
        seconds_remainder = seconds % 60
        return f"{minutes:02d}:{seconds_remainder:06.3f}"
    
    def toggle_expand(self):
        """Toggle the expansion state of the segment."""
        self.is_expanded = not self.is_expanded
        
        # Update expand button text
        self.expand_button.setText("▲" if self.is_expanded else "▼")
        
        # Update word display
        self.update_word_display()
    
    def update_word_display(self):
        """Update the word display based on the expansion state."""
        # Clear word layout
        while self.word_layout.count():
            item = self.word_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.word_buttons = []
        
        if self.is_expanded:
            # Add word buttons
            if 'words' in self.segment and self.segment['words']:
                flow_layout = FlowLayout()
                flow_layout.setContentsMargins(10, 5, 10, 5)
                
                for i, word in enumerate(self.segment['words']):
                    word_button = QPushButton(word['text'])
                    word_button.setFlat(True)
                    word_button.setCursor(Qt.CursorShape.PointingHandCursor)
                    word_button.setCheckable(True)
                    word_button.setProperty('selected', False)
                    word_button.setProperty('current', False)
                    
                    # Connect signals
                    word_button.clicked.connect(lambda checked, idx=i: self.on_word_clicked(idx))
                    
                    # Add to layout
                    flow_layout.addWidget(word_button)
                    self.word_buttons.append(word_button)
                
                # Create a container widget for the flow layout
                container = QWidget()
                container.setLayout(flow_layout)
                self.word_layout.addWidget(container)
            else:
                # If no word-level data, just add the segment text
                self.text_label = QLabel(self.segment.get('text', ''))
                self.text_label.setWordWrap(True)
                self.word_layout.addWidget(self.text_label)
        else:
            # Show text label
            self.text_label = QLabel(self.segment.get('text', ''))
            self.text_label.setWordWrap(True)
            self.word_layout.addWidget(self.text_label)
    
    def highlight_word_at_position(self, position: float):
        """Highlight the word at the given position.
        
        Args:
            position: Position in seconds
        """
        if not self.is_expanded or position is None:
            return
        
        # Find the word closest to the position
        closest_word = None
        min_distance = float('inf')
        
        for i, word in enumerate(self.segment['words']):
            # Calculate distance to word midpoint
            word_mid = (word['start'] + word['end']) / 2
            distance = abs(position - word_mid)
            
            if distance < min_distance:
                min_distance = distance
                closest_word = i
        
        # Highlight the word
        if closest_word is not None:
            for i, button in enumerate(self.word_buttons):
                is_current = (i == closest_word)
                button.setProperty('current', is_current)
                button.setStyleSheet("background-color: #e0e0e0;" if is_current else "")
    
    def get_selected_words(self) -> List[Dict[str, Any]]:
        """Get the currently selected words.
        
        Returns:
            List of selected word data
        """
        if not self.is_expanded:
            return []
        
        selected_words = []
        
        # Check which word buttons are selected
        for i, button in enumerate(self.word_buttons):
            if button.isChecked():
                selected_words.append(self.segment['words'][i])
        
        return selected_words
    
    def on_word_clicked(self, index: int):
        """Handle word button click.
        
        Args:
            index: Index of the word button
        """
        # Emit word clicked signal
        self.wordClicked.emit(self.segment['words'][index])
        
        # Update selected words
        selected_words = self.get_selected_words()
        if selected_words:
            self.wordsSelected.emit(selected_words)
    
    def select_all(self):
        """Select all words in the segment."""
        if not self.is_expanded:
            self.toggle_expand()
        
        # Select all word buttons
        for button in self.word_buttons:
            button.setChecked(True)
        
        # Emit signal for selected words
        selected_words = self.get_selected_words()
        if selected_words:
            self.wordsSelected.emit(selected_words) 