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
from src.ui.processing_worker import ProcessingWorker
from src.audio.file_handler import AudioFileHandler
from src.audio.processor import AudioProcessor
from src.utils.error_handler import ErrorHandler, ExportError, FileError

logger = logging.getLogger(__name__)

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