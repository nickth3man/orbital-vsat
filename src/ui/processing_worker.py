"""
Worker thread for processing audio files in VSAT.

This module defines the ProcessingWorker class that handles audio processing in a separate thread.
"""

import logging
from typing import Dict, Any

from PyQt6.QtCore import QThread, pyqtSignal

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
        self.is_cancelled = False
        
    def run(self):
        """Run the processing task."""
        try:
            from src.audio.processor import AudioProcessor
            
            self.progressUpdated.emit("Initializing audio processor...", 0.0)
            processor = AudioProcessor()
            
            # Process the audio file
            self.progressUpdated.emit("Loading audio file...", 0.1)
            processor.load_file(self.file_path)
            
            # Check if cancelled
            if self.is_cancelled:
                logger.info("Processing cancelled")
                return
                
            # Process audio
            self.progressUpdated.emit("Processing audio...", 0.3)
            results = processor.process(
                progress_callback=lambda msg, val: self.progressUpdated.emit(msg, val)
            )
            
            # Check if cancelled
            if self.is_cancelled:
                logger.info("Processing cancelled")
                return
                
            # Emit completion signal
            self.processingComplete.emit(results)
            
        except Exception as e:
            logger.exception("Error in processing worker: %s", str(e))
            self.errorOccurred.emit(f"Processing error: {str(e)}")
            
    def cancel(self):
        """Cancel the processing task."""
        self.is_cancelled = True
