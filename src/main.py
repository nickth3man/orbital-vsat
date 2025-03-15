#!/usr/bin/env python3
"""
Main entry point for the Voice Separation & Analysis Tool (VSAT).

This module initializes the application, sets up logging, and launches the UI.
"""

import sys
import logging
import os
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.ui.app import MainWindow
from src.utils.error_handler import install_global_error_handler

# Configure logging
def configure_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path.home() / ".vsat" / "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    log_file = log_dir / "vsat.log"
    
    # Configure logging format
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log startup message
    logging.info("VSAT application starting up")

def main():
    """Main entry point for the application."""
    # Configure logging
    configure_logging()
    
    # Install global error handler
    install_global_error_handler()
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Voice Separation & Analysis Tool")
    app.setApplicationVersion("1.0.0")
    
    # Create main window
    window = MainWindow()
    window.show()
    
    # Run application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 