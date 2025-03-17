"""
Transcript visualization widget for VSAT.

This module provides a PyQt widget for displaying and navigating transcripts with word-level precision.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Callable

from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QFontMetrics, QTextCursor, QTextCharFormat, QAction
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollBar, 
    QTextEdit, QSplitter, QFrame, QPushButton, QToolBar, QToolButton,
    QMenu, QLayout, QSizePolicy, QScrollArea
)

from src.ui.transcript_segment_widget import TranscriptSegmentWidget

logger = logging.getLogger(__name__)

class TranscriptView(QScrollArea):
    """Widget for displaying and navigating transcripts with word-level precision."""
    
    # Signal emitted when a word is clicked
    wordClicked = pyqtSignal(dict)  # Word data
    
    # Signal emitted when words are selected
    wordsSelected = pyqtSignal(list)  # List of word data
    
    # Signal emitted when the position changes
    positionChanged = pyqtSignal(float)  # Position in seconds
    
    def __init__(self, parent=None):
        """Initialize the transcript view.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up scroll area
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create container widget
        self.container = QWidget()
        self.setWidget(self.container)
        
        # Create layout
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # Initialize variables
        self.segments = []
        self.speaker_colors = {}
        self.segment_widgets = []
    
    def set_segments(self, segments: List[Dict[str, Any]], speaker_colors: Dict[Any, QColor] = None):
        """Set the transcript segments.
        
        Args:
            segments: List of transcript segments
            speaker_colors: Optional mapping of speaker IDs to colors
        """
        # Clear existing segments
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget:
                self.layout.removeWidget(widget)
                widget.deleteLater()
        
        self.segments = segments
        self.segment_widgets = []
        
        # Generate colors if not provided
        if not speaker_colors:
            self.speaker_colors = self._generate_speaker_colors(segments)
        else:
            self.speaker_colors = speaker_colors
        
        # Create segment widgets
        for segment in segments:
            widget = TranscriptSegmentWidget(segment, self.speaker_colors.get(segment.get('speaker')))
            widget.wordClicked.connect(self.on_word_clicked)
            widget.wordsSelected.connect(self.on_words_selected)
            self.layout.addWidget(widget)
            self.segment_widgets.append(widget)
        
        # Add stretch at the end
        self.layout.addStretch()
    
    def on_word_clicked(self, word: Dict[str, Any]):
        """Handle word click events.
        
        Args:
            word: Word data
        """
        # Emit signal
        self.wordClicked.emit(word)
        
        # Emit position changed signal
        self.positionChanged.emit(word['start'])
    
    def on_words_selected(self, words: List[Dict[str, Any]]):
        """Handle words selection events.
        
        Args:
            words: List of word data
        """
        # Emit signal
        self.wordsSelected.emit(words)
    
    def set_current_position(self, position: float):
        """Set the current playback position.
        
        Args:
            position: Current position in seconds
        """
        # Find segment containing the position
        segment_index = 0
        for i, segment in enumerate(self.segments):
            if segment['start'] <= position <= segment['end']:
                segment_index = i
                break
            elif segment['start'] > position:
                segment_index = max(0, i - 1)
                break
        
        # Highlight the word at the position
        for i, widget in enumerate(self.segment_widgets):
            highlighted = i == segment_index
            widget.highlight_word_at_position(position if highlighted else None)
        
        # Ensure the segment is visible
        if segment_index < len(self.segment_widgets):
            widget = self.segment_widgets[segment_index]
            self.ensureWidgetVisible(widget)
    
    def get_selected_words(self) -> List[Dict[str, Any]]:
        """Get the currently selected words.
        
        Returns:
            List of selected word data
        """
        selected_words = []
        
        # Collect selected words from all segment widgets
        for widget in self.segment_widgets:
            words = widget.get_selected_words()
            if words:
                selected_words.extend(words)
        
        return selected_words
    
    def contextMenuEvent(self, event):
        """Handle context menu events.
        
        Args:
            event: Context menu event
        """
        # Create context menu
        menu = QMenu(self)
        
        # Add actions
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(self.copy_selection)
        
        select_all_action = menu.addAction("Select All")
        select_all_action.triggered.connect(self.select_all)
        
        menu.addSeparator()
        
        export_action = menu.addAction("Export Selection...")
        export_action.triggered.connect(self.export_selection)
        
        # Show menu
        menu.exec(event.globalPos())
    
    def copy_selection(self):
        """Copy selected text to clipboard."""
        # Get selected words
        words = self.get_selected_words()
        
        if words:
            # Extract text
            text = " ".join(word['text'] for word in words)
            
            # Copy to clipboard
            QApplication.clipboard().setText(text)
    
    def select_all(self):
        """Select all text."""
        # Select all words in all segments
        for widget in self.segment_widgets:
            widget.select_all()
    
    def export_selection(self):
        """Export selected text and audio."""
        # Get selected words
        words = self.get_selected_words()
        
        if words:
            # Pass to export handler in main window
            main_window = self.window()
            if hasattr(main_window, "export_handlers") and hasattr(main_window.export_handlers, "export_selection"):
                main_window.export_handlers.export_selection()
    
    def _generate_speaker_colors(self, segments: List[Dict[str, Any]]) -> Dict[Any, QColor]:
        """Generate colors for speakers.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            Mapping of speaker IDs to colors
        """
        speakers = set()
        for segment in segments:
            if 'speaker' in segment and segment['speaker'] is not None:
                speakers.add(segment['speaker'])
        
        # Assign colors to speakers
        colors = [
            QColor(31, 119, 180),   # Blue
            QColor(255, 127, 14),   # Orange
            QColor(44, 160, 44),    # Green
            QColor(214, 39, 40),    # Red
            QColor(148, 103, 189),  # Purple
            QColor(140, 86, 75),    # Brown
            QColor(227, 119, 194),  # Pink
            QColor(127, 127, 127),  # Gray
            QColor(188, 189, 34),   # Olive
            QColor(23, 190, 207)    # Teal
        ]
        
        speaker_colors = {}
        for i, speaker in enumerate(speakers):
            speaker_colors[speaker] = colors[i % len(colors)]
        
        return speaker_colors 