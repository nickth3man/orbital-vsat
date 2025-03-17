"""
UI components module for VSAT.

This module provides UI component initialization and management functionality.
"""

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QTabWidget, QProgressBar, QStatusBar
)
from PyQt6.QtCore import Qt

from src.ui.waveform_view import WaveformView
from src.ui.transcript_view import TranscriptView
from src.ui.search_panel import SearchPanel
from src.ui.content_analysis_panel import ContentAnalysisPanel

logger = logging.getLogger(__name__)


class UIComponentManager:
    """Manages UI component creation and initialization."""

    def __init__(self, main_window):
        """Initialize the UI component manager.

        Args:
            main_window: The parent MainWindow instance
        """
        self.main_window = main_window
        self.central_widget = None
        self.main_layout = None
        self.main_splitter = None
        self.bottom_splitter = None
        self.waveform_view = None
        self.transcript_view = None
        self.search_panel = None
        self.content_analysis_panel = None
        self.status_bar = None
        self.progress_bar = None

        logger.debug("UI component manager initialized")

    def initialize(self):
        """Initialize the UI components."""
        # Set window properties
        self.main_window.setWindowTitle("Voice Separation & Analysis Tool")
        self.main_window.setMinimumSize(1200, 800)

        # Create and set up the UI components
        self._create_central_widget()
        self._create_status_bar()

        logger.debug("UI components initialized")

    def _create_central_widget(self):
        """Create the central widget and its components."""
        # Create central widget and layout
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create splitter for main components
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Create waveform view
        self.waveform_view = WaveformView()
        self.main_splitter.addWidget(self.waveform_view)

        # Create horizontal splitter for transcript and tools
        self.bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create transcript view
        self.transcript_view = TranscriptView()
        self.bottom_splitter.addWidget(self.transcript_view)

        # Create tab widget for tools
        tools_tab_widget = QTabWidget()

        # Create search panel
        self.search_panel = SearchPanel()
        tools_tab_widget.addTab(self.search_panel, "Search")

        # Create content analysis panel
        self.content_analysis_panel = ContentAnalysisPanel()
        tools_tab_widget.addTab(self.content_analysis_panel, "Content Analysis")

        self.bottom_splitter.addWidget(tools_tab_widget)

        # Set initial sizes
        self.bottom_splitter.setSizes([600, 400])

        self.main_splitter.addWidget(self.bottom_splitter)

        # Set initial sizes
        self.main_splitter.setSizes([300, 500])

        # Add components to main layout
        self.main_layout.addWidget(self.main_splitter)

        # Set central widget
        self.main_window.setCentralWidget(self.central_widget)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.main_window.setStatusBar(self.status_bar)

        # Create progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def get_waveform_view(self):
        """Get the waveform view.

        Returns:
            WaveformView: The waveform view
        """
        return self.waveform_view

    def get_transcript_view(self):
        """Get the transcript view.

        Returns:
            TranscriptView: The transcript view
        """
        return self.transcript_view

    def get_search_panel(self):
        """Get the search panel.

        Returns:
            SearchPanel: The search panel
        """
        return self.search_panel

    def get_content_analysis_panel(self):
        """Get the content analysis panel.

        Returns:
            ContentAnalysisPanel: The content analysis panel
        """
        return self.content_analysis_panel

    def get_progress_bar(self):
        """Get the progress bar.

        Returns:
            QProgressBar: The progress bar
        """
        return self.progress_bar

    def set_progress_bar_visible(self, visible):
        """Set the visibility of the progress bar.

        Args:
            visible: Whether the progress bar should be visible
        """
        self.progress_bar.setVisible(visible)

    def update_progress_bar(self, value):
        """Update the progress bar value.

        Args:
            value: The progress value (0-100)
        """
        self.progress_bar.setValue(int(value * 100))
