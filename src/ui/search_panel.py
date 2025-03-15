"""
Search panel for VSAT.

This module provides a search panel for searching through transcripts.
"""

import logging
import re
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QLineEdit, 
    QPushButton, 
    QLabel,
    QScrollArea,
    QFrame,
    QCheckBox,
    QToolButton,
    QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon

from src.ui.search_result import SearchResult, SearchResultItem

logger = logging.getLogger(__name__)

class SearchPanel(QWidget):
    """Search panel for searching through transcripts."""
    
    # Signals
    searchRequested = pyqtSignal(str, dict)  # Query, options
    resultSelected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """Initialize the search panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Initialize UI
        self.init_ui()
        
        # Initialize variables
        self.search_results = []
        self.current_transcript = None
        self.current_page = 0
        self.results_per_page = 10
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Search title
        title_label = QLabel("Search Transcript")
        title_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(title_label)
        
        # Search bar layout
        search_layout = QHBoxLayout()
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search term...")
        self.search_input.returnPressed.connect(self.search)
        search_layout.addWidget(self.search_input, 1)
        
        # Search button
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search)
        search_layout.addWidget(self.search_button)
        
        main_layout.addLayout(search_layout)
        
        # Search options layout
        options_layout = QGridLayout()
        options_layout.setContentsMargins(0, 5, 0, 5)
        
        # Case sensitivity option
        self.case_sensitive_check = QCheckBox("Case sensitive")
        self.case_sensitive_check.setChecked(False)
        options_layout.addWidget(self.case_sensitive_check, 0, 0)
        
        # Whole word option
        self.whole_word_check = QCheckBox("Whole word")
        self.whole_word_check.setChecked(False)
        options_layout.addWidget(self.whole_word_check, 0, 1)
        
        # Regex option
        self.regex_check = QCheckBox("Regular expression")
        self.regex_check.setChecked(False)
        options_layout.addWidget(self.regex_check, 1, 0)
        
        # Add options layout
        main_layout.addLayout(options_layout)
        
        # Results count label
        self.results_label = QLabel("No results")
        main_layout.addWidget(self.results_label)
        
        # Results area
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        # Container for results
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(5)
        self.results_layout.addStretch(1)
        
        self.results_scroll.setWidget(self.results_container)
        main_layout.addWidget(self.results_scroll, 1)
        
        # Pagination controls
        pagination_layout = QHBoxLayout()
        pagination_layout.setContentsMargins(0, 5, 0, 0)
        
        # Previous page button
        self.prev_button = QToolButton()
        self.prev_button.setText("Previous")
        self.prev_button.setArrowType(Qt.ArrowType.LeftArrow)
        self.prev_button.clicked.connect(self.previous_page)
        self.prev_button.setEnabled(False)
        pagination_layout.addWidget(self.prev_button)
        
        # Page indicator
        self.page_label = QLabel("Page 0 of 0")
        pagination_layout.addWidget(self.page_label, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Next page button
        self.next_button = QToolButton()
        self.next_button.setText("Next")
        self.next_button.setArrowType(Qt.ArrowType.RightArrow)
        self.next_button.clicked.connect(self.next_page)
        self.next_button.setEnabled(False)
        pagination_layout.addWidget(self.next_button)
        
        main_layout.addLayout(pagination_layout)
        
        # Set accessibility properties
        self.setAccessibleName("Search Panel")
        self.search_input.setAccessibleName("Search Input")
        self.search_button.setAccessibleName("Search Button")
        self.case_sensitive_check.setAccessibleName("Case Sensitive Option")
        self.whole_word_check.setAccessibleName("Whole Word Option")
        self.regex_check.setAccessibleName("Regular Expression Option")
        self.results_scroll.setAccessibleName("Search Results")
        self.prev_button.setAccessibleName("Previous Results Page")
        self.next_button.setAccessibleName("Next Results Page")
    
    def set_segments(self, segments: List[Dict[str, Any]]):
        """Set the transcript segments to search through.
        
        Args:
            segments: List of transcript segments
        """
        self.current_transcript = segments
        # Clear search when new transcript is loaded
        self.search_input.clear()
        self.clear_results()
    
    def search(self):
        """Perform search and emit searchRequested signal."""
        query = self.search_input.text().strip()
        if not query or not self.current_transcript:
            self.clear_results()
            return
        
        # Get search options
        options = {
            'case_sensitive': self.case_sensitive_check.isChecked(),
            'whole_word': self.whole_word_check.isChecked(),
            'regex': self.regex_check.isChecked()
        }
        
        # Reset pagination
        self.current_page = 0
        
        # Emit signal for search
        self.searchRequested.emit(query, options)
        
    def display_results(self, results: List[Dict[str, Any]]):
        """Display search results.
        
        Args:
            results: List of search results
        """
        self.search_results = results
        self.update_results_display()
        self.update_pagination()
    
    def update_results_display(self):
        """Update the display of search results."""
        # Clear previous results
        self.clear_results_display()
        
        # Update results count label
        count = len(self.search_results)
        self.results_label.setText(f"{count} result{'s' if count != 1 else ''} found")
        
        # Calculate slice for current page
        start_idx = self.current_page * self.results_per_page
        end_idx = start_idx + self.results_per_page
        page_results = self.search_results[start_idx:end_idx]
        
        # Add results to the display
        for i, result in enumerate(page_results):
            result_widget = SearchResult(result, start_idx + i + 1, self)
            result_widget.result_clicked.connect(self.on_result_selected)
            self.results_layout.insertWidget(self.results_layout.count() - 1, result_widget)
    
    def update_pagination(self):
        """Update pagination controls."""
        total_results = len(self.search_results)
        total_pages = (total_results + self.results_per_page - 1) // self.results_per_page
        
        if total_pages <= 1:
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.page_label.setText("Page 1 of 1" if total_results > 0 else "No results")
        else:
            self.prev_button.setEnabled(self.current_page > 0)
            self.next_button.setEnabled(self.current_page < total_pages - 1)
            self.page_label.setText(f"Page {self.current_page + 1} of {total_pages}")
    
    def next_page(self):
        """Go to the next page of results."""
        total_pages = (len(self.search_results) + self.results_per_page - 1) // self.results_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.update_results_display()
            self.update_pagination()
    
    def previous_page(self):
        """Go to the previous page of results."""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_results_display()
            self.update_pagination()
    
    def clear_results(self):
        """Clear search results."""
        self.search_results = []
        self.current_page = 0
        self.results_label.setText("No results")
        self.clear_results_display()
        self.update_pagination()
    
    def clear_results_display(self):
        """Clear the results display."""
        # Remove all widgets except the stretch at the end
        while self.results_layout.count() > 1:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def on_result_selected(self, result: Dict[str, Any]):
        """Handle selection of a search result.
        
        Args:
            result: The selected search result
        """
        self.resultSelected.emit(result)
        logger.info(f"Search result selected: {result.get('text', '')[:30]}...")
    
    def keyPressEvent(self, event):
        """Handle key press events.
        
        Args:
            event: Key event
        """
        # Handle shortcuts for navigating results
        if event.key() == Qt.Key.Key_N and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.next_page()
        elif event.key() == Qt.Key.Key_P and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.previous_page()
        else:
            super().keyPressEvent(event) 