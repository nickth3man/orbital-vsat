"""
Main window for the VSAT UI.

This module defines the MainWindow class that provides the main user interface.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import os

from PyQt6.QtWidgets import (
    QMainWindow, QMessageBox, QFileDialog, 
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QToolBar, QStatusBar, QProgressBar, QLabel, QPushButton,
    QMenu, QMenuBar, QSlider, QInputDialog, QTabWidget, QProgressDialog
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QAction, QIcon, QColor

from src.ui.waveform_view import WaveformView
from src.ui.transcript_view import TranscriptView
from src.ui.search_panel import SearchPanel
from src.ui.export_handlers import ExportHandlers
from src.audio.file_handler import AudioFileHandler
from src.audio.processor import AudioProcessor
from src.audio.audio_player import AudioPlayer
from src.utils.error_handler import ErrorHandler, ExportError, FileError, ErrorSeverity
from src.ui.app import ProcessingWorker
from src.ui.accessibility_dialog import AccessibilityDialog
from src.ui.content_analysis_panel import ContentAnalysisPanel
from src.ui.data_management_dialog import DataManagementDialog

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main window for the VSAT application."""
    
    def __init__(self, app):
        """Initialize the main window."""
        super().__init__()
        self.app = app
        self.current_file = None
        self.segments = []
        self.audio_processor = None
        self.processing_worker = None
        self.audio_player = AudioPlayer()
        self.settings = QSettings("VSAT", "Voice Separation & Analysis Tool")
        self.db_manager = None
        self.transcript_view = None
        self.waveform_view = None
        self.search_panel = None
        self.content_analysis_panel = None
        
        # Create export handlers
        self.export_handlers = ExportHandlers(self)
        
        self._init_ui()
        self.restore_geometry()
        
        logger.debug("Main window initialized")
    
    def _init_ui(self):
        """Initialize the UI components."""
        # Set window properties
        self.setWindowTitle("Voice Separation & Analysis Tool")
        self.setMinimumSize(1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for main components
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Create waveform view
        self.waveform_view = WaveformView()
        main_splitter.addWidget(self.waveform_view)
        
        # Create horizontal splitter for transcript and tools
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create transcript view
        self.transcript_view = TranscriptView()
        bottom_splitter.addWidget(self.transcript_view)
        
        # Create tab widget for tools
        tools_tab_widget = QTabWidget()
        
        # Create search panel
        self.search_panel = SearchPanel()
        tools_tab_widget.addTab(self.search_panel, "Search")
        
        # Create content analysis panel
        self.content_analysis_panel = ContentAnalysisPanel()
        tools_tab_widget.addTab(self.content_analysis_panel, "Content Analysis")
        
        bottom_splitter.addWidget(tools_tab_widget)
        
        # Set initial sizes
        bottom_splitter.setSizes([600, 400])
        
        main_splitter.addWidget(bottom_splitter)
        
        # Set initial sizes
        main_splitter.setSizes([300, 500])
        
        # Add components to main layout
        main_layout.addWidget(main_splitter)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Connect signals
        self.waveform_view.positionClicked.connect(self.on_position_clicked)
        self.waveform_view.rangeSelected.connect(self.on_range_selected)
        self.transcript_view.wordClicked.connect(self.on_word_clicked)
        self.transcript_view.wordsSelected.connect(self.on_words_selected)
        self.search_panel.searchRequested.connect(self.on_search_requested)
        self.search_panel.resultSelected.connect(self.on_search_result_selected)
        self.audio_player.position_changed.connect(self.on_playback_position_changed)
        self.audio_player.playback_state_changed.connect(self.on_playback_state_changed)
        self.content_analysis_panel.important_moment_selected.connect(self.on_important_moment_selected)
    
    def create_menu_bar(self):
        """Create the menu bar."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        open_action = file_menu.addAction("&Open...")
        open_action.triggered.connect(self.open_file)
        open_action.setShortcut("Ctrl+O")
        
        file_menu.addSeparator()
        
        # Export submenu
        export_menu = file_menu.addMenu("&Export")
        
        export_transcript_action = export_menu.addAction("Export &Transcript...")
        export_transcript_action.triggered.connect(self.export_handlers.export_transcript)
        
        export_audio_action = export_menu.addAction("Export &Audio Segment...")
        export_audio_action.triggered.connect(self.export_handlers.export_audio_segment)
        
        export_speaker_action = export_menu.addAction("Export &Speaker Audio...")
        export_speaker_action.triggered.connect(self.export_handlers.export_speaker_audio)
        
        export_selection_action = export_menu.addAction("Export &Selection...")
        export_selection_action.triggered.connect(self.export_handlers.export_selection)
        
        file_menu.addSeparator()
        
        # Data management action
        data_management_action = file_menu.addAction("Data Management...")
        data_management_action.triggered.connect(self.show_data_management_dialog)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Alt+F4")
        
        # Playback menu
        playback_menu = menu_bar.addMenu("&Playback")
        
        play_pause_action = playback_menu.addAction("&Play/Pause")
        play_pause_action.triggered.connect(self.toggle_playback)
        play_pause_action.setShortcut("Space")
        
        stop_action = playback_menu.addAction("&Stop")
        stop_action.triggered.connect(self.stop_playback)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        about_action = help_menu.addAction("&About")
        about_action.triggered.connect(self.show_about_dialog)
    
    def create_toolbar(self):
        """Create the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Add playback controls
        play_action = toolbar.addAction("Play")
        play_action.triggered.connect(self.toggle_playback)
        
        stop_action = toolbar.addAction("Stop")
        stop_action.triggered.connect(self.stop_playback)
    
    def restore_geometry(self):
        """Restore window geometry from settings."""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            # Default to center of screen
            screen_geometry = QApplication.primaryScreen().availableGeometry()
            self.setGeometry(
                (screen_geometry.width() - 1024) // 2,
                (screen_geometry.height() - 768) // 2,
                1024, 768
            )
    
    def save_geometry(self):
        """Save window geometry to settings."""
        self.settings.setValue("geometry", self.saveGeometry())
    
    def open_file(self, file_path=None):
        """Open an audio file for processing.
        
        Args:
            file_path: Optional path to the audio file
        """
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Audio File",
                str(Path.home()),
                "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
            )
        
        if file_path:
            logger.info(f"Opening file: {file_path}")
            self.current_file = file_path
            
            # Update window title
            file_name = Path(file_path).name
            self.setWindowTitle(f"{file_name} - Voice Separation & Analysis Tool")
            
            # Show status message
            self.status_bar.showMessage(f"Processing: {file_name}")
            
            # Show progress bar
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            
            # Start processing in a separate thread
            self.processing_worker = ProcessingWorker(file_path)
            self.processing_worker.progressUpdated.connect(self.update_processing_progress)
            self.processing_worker.processingComplete.connect(self.processing_complete)
            self.processing_worker.errorOccurred.connect(self.processing_error)
            self.processing_worker.start()
    
    @pyqtSlot(str, float)
    def update_processing_progress(self, status: str, progress: float):
        """Update processing progress in the UI.
        
        Args:
            status: Status message
            progress: Progress value (0.0 to 1.0)
        """
        # Update status bar
        self.status_bar.showMessage(status)
        
        # Update progress bar
        self.progress_bar.setValue(int(progress * 100))
    
    @pyqtSlot(dict)
    def processing_complete(self, results: Dict[str, Any]):
        """Handle completion of audio processing.
        
        Args:
            results: Processing results
        """
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Update status bar
        file_name = Path(self.current_file).name
        self.status_bar.showMessage(f"Loaded: {file_name}")
        
        # Store results
        self.segments = results.get('segments', [])
        
        # Update UI components
        self.update_ui_with_results()
    
    @pyqtSlot(str)
    def processing_error(self, error_message: str):
        """Handle processing error.
        
        Args:
            error_message: Error message
        """
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Update status bar
        self.status_bar.showMessage("Error processing file")
        
        # Show error message
        QMessageBox.critical(
            self,
            "Processing Error",
            f"An error occurred while processing the file:\n\n{error_message}"
        )
    
    def update_ui_with_results(self):
        """Update UI components with processing results."""
        # Update waveform view
        if self.current_file:
            self.waveform_view.set_audio_data(self.current_file)
        
        # Update waveform segments
        self.waveform_view.set_segments(self.segments)
        
        # Update transcript view
        self.transcript_view.set_segments(self.segments)
        
        # Update search panel
        self.search_panel.set_segments(self.segments)
        
        # Update content analysis panel
        self.content_analysis_panel.set_segments(self.segments)
        
        # Load audio file in player
        if self.current_file:
            self.audio_player.load_file(self.current_file)
    
    def toggle_playback(self):
        """Toggle playback of the current audio file."""
        if not self.current_file:
            return
        
        if self.audio_player.is_playing():
            self.audio_player.pause()
        else:
            # If there's a selection, play that range
            if self.waveform_view.has_selection():
                start, end = self.waveform_view.get_selection_range()
                self.audio_player.play(self.current_file, start, end)
            else:
                # Otherwise play from current position
                self.audio_player.play(self.current_file)
    
    def stop_playback(self):
        """Stop playback of the current audio file."""
        self.audio_player.stop()
    
    def on_playback_position_changed(self, position):
        """Handle playback position changes."""
        self.waveform_view.set_current_position(position)
        self.transcript_view.set_current_position(position)
    
    def on_playback_state_changed(self, is_playing):
        """Handle playback state changes."""
        # Update UI based on playback state
        pass
    
    def show_about_dialog(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About Voice Separation & Analysis Tool",
            "Voice Separation & Analysis Tool (VSAT)\n\n"
            "A tool for analyzing and separating voices in audio recordings.\n\n"
            "Version: 0.1.0"
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Save window geometry
        self.save_geometry()
        
        # TODO: Check for unsaved changes
        
        # Accept the close event
        event.accept()
    
    def on_position_clicked(self, position):
        """Handle click on a position in the waveform."""
        # Update transcript view
        self.transcript_view.set_current_position(position)
        
        # Update audio player position
        self.audio_player.set_position(position)
        
        # Update status bar
        minutes = int(position) // 60
        seconds = int(position) % 60
        milliseconds = int((position - int(position)) * 1000)
        self.status_bar.showMessage(f"Position: {minutes:02d}:{seconds:02d}.{milliseconds:03d}")
    
    def on_range_selected(self, start, end):
        """Handle selection of a range in the waveform."""
        # Play the selected range
        self.audio_player.play(self.current_file, start, end)
        
        # Update status bar
        duration = end - start
        self.status_bar.showMessage(f"Selected range: {start:.2f}s to {end:.2f}s (duration: {duration:.2f}s)")
    
    def on_word_clicked(self, word):
        """Handle click on a word in the transcript."""
        # Update waveform position
        position = word['start']
        self.waveform_view.set_current_position(position)
        
        # Update audio player position
        self.audio_player.set_position(position)
        
        # Update status bar
        self.status_bar.showMessage(f"Word: {word['text']} ({word['start']:.2f}s to {word['end']:.2f}s)")
    
    def on_words_selected(self, words):
        """Handle selection of words in the transcript."""
        if not words:
            return
            
        # Get start and end times
        start = min(word['start'] for word in words)
        end = max(word['end'] for word in words)
        
        # Update waveform selection
        self.waveform_view.set_selection(start, end)
        
        # Update status bar
        word_count = len(words)
        text = " ".join(word['text'] for word in words)
        if len(text) > 50:
            text = text[:47] + "..."
        self.status_bar.showMessage(f"Selected {word_count} words: {text}")
    
    def on_search_requested(self, query, options=None):
        """Handle search request from the search panel.
        
        Args:
            query: Search query string
            options: Optional dictionary with search options
        """
        # Default options if none provided
        if options is None:
            options = {
                'case_sensitive': False,
                'whole_word': False,
                'regex': False
            }
        
        # Update status bar
        self.status_bar.showMessage(f"Searching for: {query}")
        
        # Check if we have segments to search through
        if not self.segments:
            self.search_panel.clear_results()
            self.status_bar.showMessage("No transcript to search")
            return
        
        # Search in segments
        results = []
        
        # Prepare query for different search methods
        if options['regex']:
            try:
                if options['case_sensitive']:
                    pattern = re.compile(query)
                else:
                    pattern = re.compile(query, re.IGNORECASE)
            except re.error:
                self.status_bar.showMessage(f"Invalid regular expression: {query}")
                self.search_panel.clear_results()
                return
        elif options['whole_word']:
            if options['case_sensitive']:
                pattern = re.compile(r'\b' + re.escape(query) + r'\b')
            else:
                pattern = re.compile(r'\b' + re.escape(query) + r'\b', re.IGNORECASE)
        elif not options['case_sensitive']:
            query = query.lower()
        
        for segment in self.segments:
            # Get text from segment
            segment_text = segment.get("text", "")
            if not options['case_sensitive'] and not options['regex'] and not options['whole_word']:
                segment_text = segment_text.lower()
            
            # Check if query is in segment text using the appropriate method
            is_match = False
            if options['regex'] or options['whole_word']:
                is_match = bool(pattern.search(segment_text))
            else:
                is_match = query in segment_text
            
            if is_match:
                # Get words from segment
                words = segment.get("words", [])
                
                # Find matching words/phrases
                for i, word in enumerate(words):
                    word_text = word.get("text", "")
                    if not options['case_sensitive'] and not options['regex'] and not options['whole_word']:
                        word_text = word_text.lower()
                    
                    # Check for match using the appropriate method
                    word_match = False
                    if options['regex'] or options['whole_word']:
                        word_match = bool(pattern.search(word_text))
                    else:
                        word_match = query in word_text
                    
                    if word_match:
                        results.append({
                            "segment": segment,
                            "word": word,
                            "text": word.get("text", ""),
                            "start": word.get("start", 0),
                            "end": word.get("end", 0),
                            "speaker": segment.get("speaker", "Unknown"),
                            "context": self._get_context(segment, i, 5)  # Increased context size
                        })
        
        # Send results to search panel
        self.search_panel.display_results(results)
        
        # Update status bar with count
        count = len(results)
        self.status_bar.showMessage(f"Found {count} result{'s' if count != 1 else ''} for: {query}")
    
    def _get_context(self, segment, word_index, context_size=3):
        """Get context around a word in a segment.
        
        Args:
            segment: Segment containing the word
            word_index: Index of the word in the segment
            context_size: Number of words to include before and after
            
        Returns:
            Context string
        """
        words = segment.get("words", [])
        
        # Get words before and after the matched word
        start_idx = max(0, word_index - context_size)
        end_idx = min(len(words), word_index + context_size + 1)
        
        # Extract context words and highlight the matched word
        context_words = []
        for i in range(start_idx, end_idx):
            word_text = words[i].get("text", "")
            if i == word_index:
                context_words.append(f"**{word_text}**")  # Highlight matched word
            else:
                context_words.append(word_text)
        
        return " ".join(context_words)
    
    def on_search_result_selected(self, result):
        """Handle selection of a search result."""
        if result and 'start' in result:
            # Set playback position
            if self.audio_player:
                self.audio_player.set_position(result['start'])
            
            # Highlight in transcript view
            if self.transcript_view:
                self.transcript_view.scroll_to_position(result['start'])
            
            # Highlight in waveform view
            if self.waveform_view:
                self.waveform_view.set_position(result['start'])
    
    def on_important_moment_selected(self, moment):
        """Handle selection of an important moment."""
        if moment and 'start' in moment:
            # Set playback position
            if self.audio_player:
                self.audio_player.set_position(moment['start'])
            
            # Highlight in transcript view
            if self.transcript_view:
                self.transcript_view.scroll_to_position(moment['start'])
            
            # Highlight in waveform view
            if self.waveform_view:
                self.waveform_view.set_position(moment['start'])
            
            # Play the segment
            if self.audio_player and 'end' in moment:
                self.audio_player.play_segment(moment['start'], moment['end'])
    
    def show_data_management_dialog(self):
        """Show the data management dialog."""
        dialog = DataManagementDialog(self.db_manager, self)
        dialog.exec() 