"""
Main application class for the VSAT UI.

This module defines the Application class that initializes and manages the UI.
"""

import sys
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from src.ui.main_window import MainWindow
from src.audio.file_handler import AudioFileHandler
from src.audio.processor import AudioProcessor
from src.utils.error_handler import ErrorHandler, ExportError, FileError

logger = logging.getLogger(__name__)

class ProcessingWorker(QThread):
    """Worker thread for processing audio files."""
    
    # Signal emitted when processing progress updates
    progressUpdated = pyqtSignal(str, float)  # status message, progress value
    
    # Signal emitted when processing is complete
    processingComplete = pyqtSignal(dict)  # processing results
    
    # Signal emitted when an error occurs
    errorOccurred = pyqtSignal(str)  # error message
    
    def __init__(self, file_path: str, parent=None):
        """Initialize the worker thread.
        
        Args:
            file_path: Path to the audio file to process
            parent: Parent object
        """
        super().__init__(parent)
        self.file_path = file_path
        self.audio_processor = AudioProcessor()
        self.audio_processor.progress_callback = self.update_progress
    
    def run(self):
        """Run the processing task."""
        try:
            # Process the audio file
            results = self.audio_processor.process_file(self.file_path)
            
            # Emit the result signal
            self.processingComplete.emit(results)
        except Exception as e:
            # Log the error
            logger.error(f"Error processing audio file: {e}", exc_info=True)
            
            # Emit the error signal
            self.errorOccurred.emit(str(e))
    
    def update_progress(self, status: str, progress: float):
        """Update the processing progress.
        
        Args:
            status: Status message
            progress: Progress value (0-1)
        """
        self.progressUpdated.emit(status, progress)


class Application:
    """Main application class for VSAT."""
    
    def __init__(self):
        """Initialize the application."""
        self.app = QApplication(sys.argv)
        self.window = MainWindow(self)
        
        # Set up error handler
        self.error_handler = ErrorHandler()
        self.error_handler.set_parent_window(self.window)
        
        # Set application style
        self.app.setStyle("Fusion")
    
    def open_file(self, file_path):
        """Open an audio file.
        
        Args:
            file_path: Path to the audio file
        """
        self.window.open_file(file_path)
    
    def run(self):
        """Run the application."""
        self.window.show()
        return self.app.exec() 