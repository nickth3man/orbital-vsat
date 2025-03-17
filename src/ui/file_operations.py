"""
File operations module for VSAT.

This module provides file loading and processing functionality.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from src.audio.file_handler import AudioFileHandler
from src.ui.processing_worker import ProcessingWorker
from src.utils.error_handler import ErrorHandler, FileError, ErrorSeverity

logger = logging.getLogger(__name__)


class FileOperations:
    """Handles file operations for the main window."""
    
    def __init__(self, main_window):
        """Initialize the file operations handler.
        
        Args:
            main_window: The parent MainWindow instance
        """
        self.main_window = main_window
        
        logger.debug("File operations handler initialized")
        
    def open_file(self, file_path=None):
        """Open an audio file for processing.
        
        Args:
            file_path: Optional path to the audio file
        """
        # If no file path provided, show file dialog
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Open Audio File",
                str(Path.home()),
                "Audio Files (*.mp3 *.wav *.ogg *.flac);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
        
        try:
            # Check if file exists
            if not Path(file_path).exists():
                raise FileError(f"File not found: {file_path}", ErrorSeverity.ERROR)
                
            # Check if file is readable
            audio_file = AudioFileHandler()
            if not audio_file.is_valid_audio_file(file_path):
                raise FileError(f"Invalid audio file: {file_path}", ErrorSeverity.ERROR)
                
            # Update UI
            self.main_window.current_file = file_path
            self.main_window.setWindowTitle(f"VSAT - {Path(file_path).name}")
            
            # Show progress dialog
            progress_dialog = QProgressDialog("Processing audio file...", "Cancel", 0, 0, self.main_window)
            progress_dialog.setWindowTitle("Processing")
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setMinimumDuration(500)  # ms
            progress_dialog.setAutoClose(True)
            progress_dialog.setAutoReset(True)
            
            # Create processing worker
            self.main_window.processing_worker = ProcessingWorker(file_path, self.main_window.audio_processor)
            
            # Connect worker signals
            self.main_window.processing_worker.progress.connect(
                lambda status, progress: self.update_processing_progress(status, progress, progress_dialog)
            )
            self.main_window.processing_worker.finished.connect(
                lambda results: self.processing_complete(results, progress_dialog)
            )
            self.main_window.processing_worker.error.connect(
                lambda error: self.processing_error(error, progress_dialog)
            )
            
            # Start processing thread
            processing_thread = threading.Thread(
                target=self.main_window.processing_worker.process,
                daemon=True
            )
            processing_thread.start()
            
            logger.info(f"Opened audio file: {file_path}")
            
        except FileError as e:
            # Handle file error
            logger.error(f"File error: {str(e)}")
            ErrorHandler.show_error(
                self.main_window,
                f"Error opening audio file: {str(e)}",
                e.severity
            )
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error opening audio file: {str(e)}", exc_info=True)
            ErrorHandler.show_error(
                self.main_window,
                f"Unexpected error opening audio file: {str(e)}",
                ErrorSeverity.ERROR
            )
            
    def update_processing_progress(self, status: str, progress: float, dialog=None):
        """Update processing progress in the UI.
        
        Args:
            status: Status message
            progress: Progress value (0.0 to 1.0)
            dialog: Optional progress dialog
        """
        # Update status bar progress
        ui = self.main_window.ui_components
        progress_bar = ui.get_progress_bar()
        
        if progress_bar:
            if progress < 0:
                # Indeterminate progress
                progress_bar.setRange(0, 0)
            else:
                # Determinate progress
                progress_bar.setRange(0, 100)
                progress_bar.setValue(int(progress * 100))
                
            progress_bar.setVisible(True)
            
        # Update status bar text
        if status:
            self.main_window.statusBar().showMessage(status)
            
        # Update progress dialog if provided
        if dialog:
            if progress < 0:
                dialog.setRange(0, 0)
            else:
                dialog.setRange(0, 100)
                dialog.setValue(int(progress * 100))
                
            dialog.setLabelText(status)
            
    def processing_complete(self, results: Dict[str, Any], dialog=None):
        """Handle completion of audio processing.
        
        Args:
            results: Processing results
            dialog: Optional progress dialog
        """
        # Hide progress bar
        ui = self.main_window.ui_components
        progress_bar = ui.get_progress_bar()
        
        if progress_bar:
            progress_bar.setVisible(False)
            
        # Update status bar
        self.main_window.statusBar().showMessage("Processing complete")
        
        # Close dialog if provided
        if dialog and dialog.isVisible():
            dialog.close()
            
        # Update segments
        self.main_window.segments = results.get('segments', [])
        
        # Update UI with results
        self.update_ui_with_results()
        
        # Load audio file in player
        if self.main_window.audio_player:
            self.main_window.audio_player.load_file(self.main_window.current_file)
            
        logger.info("Processing complete")
        
    def processing_error(self, error_message: str, dialog=None):
        """Handle processing error.
        
        Args:
            error_message: Error message
            dialog: Optional progress dialog
        """
        # Hide progress bar
        ui = self.main_window.ui_components
        progress_bar = ui.get_progress_bar()
        
        if progress_bar:
            progress_bar.setVisible(False)
            
        # Update status bar
        self.main_window.statusBar().showMessage("Processing failed")
        
        # Close dialog if provided
        if dialog and dialog.isVisible():
            dialog.close()
            
        # Show error message
        ErrorHandler.show_error(
            self.main_window,
            f"Error processing audio file: {error_message}",
            ErrorSeverity.ERROR
        )
        
        logger.error(f"Processing error: {error_message}")
        
    def update_ui_with_results(self):
        """Update UI components with processing results."""
        if not self.main_window.segments:
            return
            
        # Update waveform view
        ui = self.main_window.ui_components
        waveform_view = ui.get_waveform_view()
        
        if waveform_view:
            waveform_view.set_segments(self.main_window.segments)
            waveform_view.update()
            
        # Update transcript view
        transcript_view = ui.get_transcript_view()
        
        if transcript_view:
            transcript_view.set_segments(self.main_window.segments)
            transcript_view.update()
            
        # Update content analysis panel
        content_analysis_panel = ui.get_content_analysis_panel()
        
        if content_analysis_panel:
            content_analysis_panel.analyze_content(self.main_window.segments)
            
        logger.debug("UI updated with processing results")
