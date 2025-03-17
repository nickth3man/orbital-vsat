"""
Main window for the VSAT UI.

This module defines the MainWindow class that provides the main user interface.
"""

import logging
import os
from pathlib import Path

from PyQt6.QtWidgets import QMainWindow, QMessageBox
from PyQt6.QtCore import QSettings

from src.ui.components import UIComponentManager
from src.ui.menu_manager import MenuManager
from src.ui.event_handlers import EventHandler
from src.ui.file_operations import FileOperations
from src.ui.window_state import WindowStateManager
from src.ui.export_handlers import ExportHandlers
from src.ui.batch_processing_dialog import BatchProcessingDialog
from src.ui.accessibility_dialog import AccessibilityDialog
from src.ui.data_management_dialog import DataManagementDialog
from src.audio.audio_player import AudioPlayer
from src.audio.processor import AudioProcessor
from src.database.data_manager import DataManager

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
        
        # Initialize data manager
        self.data_manager = None
        self._init_data_manager()
        
        # Get the database manager from data manager
        db_manager = self.data_manager.db_manager if self.data_manager else None
        
        # Initialize component managers
        self.export_handlers = ExportHandlers(self)
        self.ui_components = UIComponentManager(self)
        self.menu_manager = MenuManager(self)
        self.file_operations = FileOperations(self)
        self.window_state = WindowStateManager(self)
        self.event_handler = EventHandler(self)
        
        # Initialize UI components
        self.ui_components.initialize()
        self.menu_manager.create_menus_and_toolbars()
        self.event_handler.connect_signals()
        
        # Initialize audio processor with db_manager
        if db_manager:
            self.audio_processor = AudioProcessor(db_manager)
            logger.info("Audio processor initialized successfully")
        else:
            # Create a fallback if data manager initialization failed
            from src.database.db_manager import DatabaseManager
            fallback_db = DatabaseManager(":memory:")  # Use in-memory database
            self.audio_processor = AudioProcessor(fallback_db)
            logger.warning(
                "Using in-memory database for audio processor due to "
                "data manager initialization failure"
            )
        
        # Restore window geometry
        self.window_state.restore_geometry()
        
        logger.info("Main window initialized")
    
    def _init_data_manager(self):
        """Initialize the data manager."""
        try:
            # Get settings from window state manager
            settings = QSettings("VSAT", "Voice Separation & Analysis Tool")
            
            # Get database path from settings or use default
            db_path = settings.value(
                "database/path", 
                str(Path.home() / ".vsat" / "vsat.db")
            )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Initialize data manager
            db_manager = DataManager(db_path)
            db_manager.initialize_database()
            
            # Store data manager
            self.data_manager = db_manager
            
            logger.info(f"Data manager initialized with database at {db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize data manager: {str(e)}")
            self.data_manager = None
    
    def toggle_playback(self):
        """Toggle playback of the current audio file."""
        if not self.current_file:
            return
        
        if self.audio_player.is_playing():
            self.audio_player.pause()
        else:
            # Get waveform view from UI components
            waveform_view = self.ui_components.get_waveform_view()
            
            # If there's a selection, play that range
            if waveform_view and waveform_view.has_selection():
                start, end = waveform_view.get_selection_range()
                self.audio_player.play_segment(start, end)
            else:
                # Otherwise play from current position
                self.audio_player.play()
    
    def stop_playback(self):
        """Stop playback of the current audio file."""
        if self.audio_player:
            self.audio_player.stop()
    
    def show_about_dialog(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About VSAT",
            "Voice Separation & Analysis Tool\n\n"
            "A tool for separating and analyzing voice recordings."
        )
    
    def show_data_management_dialog(self):
        """Show the data management dialog."""
        if not self.data_manager:
            QMessageBox.warning(
                self,
                "Data Management",
                "Data manager is not initialized. Please try again later."
            )
            return
        
        dialog = DataManagementDialog(self.data_manager, self)
        dialog.exec()
    
    def show_batch_processing_dialog(self):
        """Show the batch processing dialog."""
        if not self.audio_processor:
            QMessageBox.warning(
                self,
                "Batch Processing",
                "Audio processor is not initialized. Please try again later."
            )
            return
        
        dialog = BatchProcessingDialog(self.audio_processor, self)
        dialog.exec()
    
    def show_accessibility_dialog(self):
        """Show the accessibility settings dialog."""
        dialog = AccessibilityDialog(self)
        dialog.exec()
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.window_state.handle_close_event(event)