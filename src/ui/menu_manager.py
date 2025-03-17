"""
Menu and toolbar management module for VSAT.

This module provides menu and toolbar creation and management functionality.
"""

import logging
from PyQt6.QtWidgets import QMenu, QToolBar
from PyQt6.QtGui import QAction

logger = logging.getLogger(__name__)


class MenuManager:
    """Manages menu and toolbar creation and management."""

    def __init__(self, main_window):
        """Initialize the menu manager.

        Args:
            main_window: The parent MainWindow instance
        """
        self.main_window = main_window
        self.menu_bar = None
        self.file_menu = None
        self.edit_menu = None
        self.view_menu = None
        self.tools_menu = None
        self.help_menu = None
        self.toolbar = None
        self.play_action = None

        logger.debug("Menu manager initialized")

    def create_menus_and_toolbars(self):
        """Create menus and toolbars."""
        self.create_menu_bar()
        self.create_toolbar()

        logger.debug("Menus and toolbars created")

    def create_menu_bar(self):
        """Create the menu bar."""
        self.menu_bar = self.main_window.menuBar()

        # File menu
        self.file_menu = self.menu_bar.addMenu("&File")

        open_action = QAction("&Open...", self.main_window)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(
            lambda: self.main_window.file_operations.open_file()
        )
        self.file_menu.addAction(open_action)

        # Add batch processing action
        batch_action = QAction("&Batch Processing...", self.main_window)
        batch_action.setShortcut("Ctrl+B")
        batch_action.triggered.connect(self.main_window.show_batch_processing_dialog)
        self.file_menu.addAction(batch_action)

        self.file_menu.addSeparator()

        exit_action = QAction("E&xit", self.main_window)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.main_window.close)
        self.file_menu.addAction(exit_action)

        # Edit menu
        self.edit_menu = self.menu_bar.addMenu("&Edit")

        # Preferences action
        preferences_action = QAction("&Preferences...", self.main_window)
        preferences_action.setShortcut("Ctrl+P")
        self.edit_menu.addAction(preferences_action)

        # Data management action
        data_management_action = QAction("&Data Management...", self.main_window)
        data_management_action.triggered.connect(
            self.main_window.show_data_management_dialog
        )
        self.edit_menu.addAction(data_management_action)

        # View menu
        self.view_menu = self.menu_bar.addMenu("&View")

        # Accessibility settings action
        accessibility_action = QAction("&Accessibility Settings...", self.main_window)
        accessibility_action.triggered.connect(
            self.main_window.show_accessibility_dialog
        )
        self.view_menu.addAction(accessibility_action)

        # Tools menu
        self.tools_menu = self.menu_bar.addMenu("&Tools")

        # Export submenu
        export_menu = QMenu("&Export", self.main_window)

        export_transcript_action = QAction("Export &Transcript...", self.main_window)
        export_transcript_action.triggered.connect(
            lambda: self.main_window.export_handlers.export_transcript()
        )
        export_menu.addAction(export_transcript_action)

        export_audio_action = QAction("Export &Audio Segments...", self.main_window)
        export_audio_action.triggered.connect(
            lambda: self.main_window.export_handlers.export_audio_segments()
        )
        export_menu.addAction(export_audio_action)

        self.tools_menu.addMenu(export_menu)

        # Help menu
        self.help_menu = self.menu_bar.addMenu("&Help")

        # About action
        about_action = QAction("&About", self.main_window)
        about_action.triggered.connect(self.main_window.show_about_dialog)
        self.help_menu.addAction(about_action)

    def create_toolbar(self):
        """Create the toolbar."""
        self.toolbar = QToolBar("Main Toolbar")
        self.main_window.addToolBar(self.toolbar)

        # Open action
        open_action = QAction("Open", self.main_window)
        open_action.setToolTip("Open audio file")
        open_action.triggered.connect(
            lambda: self.main_window.file_operations.open_file()
        )
        self.toolbar.addAction(open_action)

        self.toolbar.addSeparator()

        # Play/pause action
        self.play_action = QAction("Play", self.main_window)
        self.play_action.setToolTip("Play/Pause audio")
        self.play_action.triggered.connect(self.main_window.toggle_playback)
        self.toolbar.addAction(self.play_action)

        # Stop action
        stop_action = QAction("Stop", self.main_window)
        stop_action.setToolTip("Stop audio")
        stop_action.triggered.connect(self.main_window.stop_playback)
        self.toolbar.addAction(stop_action)

        self.toolbar.addSeparator()

        # Export action
        export_action = QAction("Export", self.main_window)
        export_action.setToolTip("Export data")
        self.toolbar.addAction(export_action)

    def update_play_action_icon(self, is_playing):
        """Update the play action icon based on playback state.

        Args:
            is_playing: Whether audio is currently playing
        """
        if is_playing:
            self.play_action.setText("Pause")
            self.play_action.setToolTip("Pause audio")
        else:
            self.play_action.setText("Play")
            self.play_action.setToolTip("Play audio")
