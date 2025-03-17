"""
Window state management module for VSAT.

This module provides window state management functionality.
"""

import logging
from PyQt6.QtCore import QSettings, QSize, QPoint
from PyQt6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)


class WindowStateManager:
    """Manages window state including geometry and settings."""
    
    def __init__(self, main_window):
        """Initialize the window state manager.
        
        Args:
            main_window: The parent MainWindow instance
        """
        self.main_window = main_window
        self.settings = QSettings("VSAT", "Voice Separation & Analysis Tool")
        
        logger.debug("Window state manager initialized")
        
    def restore_geometry(self):
        """Restore window geometry from settings."""
        # Check if geometry settings exist
        if self.settings.contains("window/geometry"):
            geometry = self.settings.value("window/geometry")
            self.main_window.restoreGeometry(geometry)
            logger.debug("Window geometry restored from settings")
        else:
            # Set default size and position
            self.main_window.resize(1200, 800)
            self.main_window.move(100, 100)
            logger.debug("Default window geometry applied")
            
        # Check if state settings exist
        if self.settings.contains("window/state"):
            state = self.settings.value("window/state")
            self.main_window.restoreState(state)
            logger.debug("Window state restored from settings")
            
    def save_geometry(self):
        """Save window geometry to settings."""
        self.settings.setValue("window/geometry", self.main_window.saveGeometry())
        self.settings.setValue("window/state", self.main_window.saveState())
        logger.debug("Window geometry and state saved to settings")
        
    def handle_close_event(self, event):
        """Handle window close event.
        
        Args:
            event: Close event
        """
        # Save window geometry
        self.save_geometry()
        
        # Accept the event to close the window
        event.accept()
        
        logger.debug("Window close event handled")
        
    def get_settings(self):
        """Get the settings object.
        
        Returns:
            QSettings: The settings object
        """
        return self.settings
