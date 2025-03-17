"""
Accessibility features for the VSAT UI.

This module provides accessibility enhancements for the VSAT application,
including keyboard navigation, screen reader support, and high contrast modes.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from PyQt6.QtWidgets import (
    QWidget, QApplication, QMainWindow, QPushButton, QLabel, 
    QSlider, QComboBox, QLineEdit, QCheckBox, QRadioButton,
    QMessageBox, QMenu
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QEvent
from PyQt6.QtGui import QPalette, QColor, QFont, QKeySequence, QShortcut, QAction

logger = logging.getLogger(__name__)

class ColorScheme(Enum):
    """Enumeration of color schemes for the application."""
    STANDARD = 0
    HIGH_CONTRAST = 1
    DARK = 2
    LIGHT = 3


class AccessibilityManager:
    """Class for managing accessibility features in the application."""
    
    def __init__(self, main_window: QMainWindow):
        """Initialize the accessibility manager.
        
        Args:
            main_window: Main window of the application
        """
        self.main_window = main_window
        self.color_scheme = ColorScheme.STANDARD
        self.screen_reader_enabled = False
        self.keyboard_focus_visible = True
        self.font_scale = 1.0
        self.shortcuts = {}
        
        # Initialize accessibility features
        self._setup_keyboard_navigation()
        self._setup_screen_reader_support()
        self._setup_color_schemes()
        self._setup_font_scaling()
    
    def _setup_keyboard_navigation(self):
        """Set up keyboard navigation for the application."""
        # Create keyboard shortcuts
        self._create_shortcut(QKeySequence(Qt.Key.Key_F6), self._cycle_focus_areas)
        self._create_shortcut(QKeySequence(Qt.Key.Key_F1), self._show_help)
        
        # Enable tab focus for all widgets
        self._set_tab_focus_for_all_widgets(self.main_window)
    
    def _setup_screen_reader_support(self):
        """Set up screen reader support for the application."""
        # Set accessibility properties for important widgets
        self._set_accessibility_properties(self.main_window)
    
    def _setup_color_schemes(self):
        """Set up color schemes for the application."""
        # Create color scheme palettes
        self.palettes = {
            ColorScheme.STANDARD: self._create_standard_palette(),
            ColorScheme.HIGH_CONTRAST: self._create_high_contrast_palette(),
            ColorScheme.DARK: self._create_dark_palette(),
            ColorScheme.LIGHT: self._create_light_palette()
        }
    
    def _setup_font_scaling(self):
        """Set up font scaling for the application."""
        # Store the default font
        self.default_font = QApplication.font()
    
    def _create_shortcut(self, key_sequence: QKeySequence, callback: Callable):
        """Create a keyboard shortcut.
        
        Args:
            key_sequence: Key sequence for the shortcut
            callback: Function to call when the shortcut is triggered
        """
        shortcut = QShortcut(key_sequence, self.main_window)
        shortcut.activated.connect(callback)
        self.shortcuts[key_sequence.toString()] = shortcut
    
    def _cycle_focus_areas(self):
        """Cycle through main focus areas of the application."""
        # Get all focusable widgets
        focusable_widgets = self._get_focusable_widgets(self.main_window)
        
        # Get the current focus widget
        current_widget = QApplication.focusWidget()
        
        if current_widget in focusable_widgets:
            # Get the index of the current widget
            current_index = focusable_widgets.index(current_widget)
            
            # Calculate the next index
            next_index = (current_index + 1) % len(focusable_widgets)
            
            # Set focus to the next widget
            focusable_widgets[next_index].setFocus()
        elif focusable_widgets:
            # Set focus to the first widget
            focusable_widgets[0].setFocus()
    
    def _show_help(self):
        """Show help dialog with keyboard shortcuts."""
        help_text = "Keyboard Shortcuts:\n\n"
        help_text += "F1: Show this help dialog\n"
        help_text += "F6: Cycle through main focus areas\n"
        help_text += "Tab: Move to next control\n"
        help_text += "Shift+Tab: Move to previous control\n"
        help_text += "Space/Enter: Activate button or toggle control\n"
        help_text += "Arrow keys: Navigate in lists and sliders\n"
        
        QMessageBox.information(self.main_window, "Keyboard Shortcuts", help_text)
    
    def _set_tab_focus_for_all_widgets(self, parent_widget: QWidget):
        """Set tab focus policy for all widgets.
        
        Args:
            parent_widget: Parent widget to process
        """
        # Process the parent widget
        if isinstance(parent_widget, (QPushButton, QSlider, QComboBox, QLineEdit, QCheckBox, QRadioButton)):
            parent_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Process all child widgets
        for child in parent_widget.findChildren(QWidget):
            self._set_tab_focus_for_all_widgets(child)
    
    def _set_accessibility_properties(self, parent_widget: QWidget):
        """Set accessibility properties for all widgets.
        
        Args:
            parent_widget: Parent widget to process
        """
        # Process the parent widget
        if hasattr(parent_widget, 'accessibleName') and not parent_widget.accessibleName():
            # Try to set a reasonable accessible name
            if hasattr(parent_widget, 'text') and callable(getattr(parent_widget, 'text')):
                parent_widget.setAccessibleName(parent_widget.text())
            elif hasattr(parent_widget, 'title') and callable(getattr(parent_widget, 'title')):
                parent_widget.setAccessibleName(parent_widget.title())
            elif hasattr(parent_widget, 'windowTitle') and callable(getattr(parent_widget, 'windowTitle')):
                parent_widget.setAccessibleName(parent_widget.windowTitle())
        
        # Process all child widgets
        for child in parent_widget.findChildren(QWidget):
            self._set_accessibility_properties(child)
    
    def _get_focusable_widgets(self, parent_widget: QWidget) -> List[QWidget]:
        """Get all focusable widgets.
        
        Args:
            parent_widget: Parent widget to process
            
        Returns:
            List of focusable widgets
        """
        focusable_widgets = []
        
        # Check if the parent widget is focusable
        if parent_widget.focusPolicy() != Qt.FocusPolicy.NoFocus:
            focusable_widgets.append(parent_widget)
        
        # Check all child widgets
        for child in parent_widget.findChildren(QWidget):
            if child.isVisible() and child.focusPolicy() != Qt.FocusPolicy.NoFocus:
                focusable_widgets.append(child)
        
        return focusable_widgets
    
    def _create_standard_palette(self) -> QPalette:
        """Create the standard color palette.
        
        Returns:
            Standard color palette
        """
        return QApplication.palette()
    
    def _create_high_contrast_palette(self) -> QPalette:
        """Create a high contrast color palette.
        
        Returns:
            High contrast color palette
        """
        palette = QPalette()
        
        # Set high contrast colors
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(20, 20, 20))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(0, 255, 255))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(255, 255, 0))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        
        return palette
    
    def _create_dark_palette(self) -> QPalette:
        """Create a dark color palette.
        
        Returns:
            Dark color palette
        """
        palette = QPalette()
        
        # Set dark theme colors
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        
        return palette
    
    def _create_light_palette(self) -> QPalette:
        """Create a light color palette.
        
        Returns:
            Light color palette
        """
        palette = QPalette()
        
        # Set light theme colors
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 233, 233))
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(51, 153, 255))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        
        return palette
    
    def set_color_scheme(self, scheme: ColorScheme):
        """Set the color scheme for the application.
        
        Args:
            scheme: Color scheme to set
        """
        self.color_scheme = scheme
        QApplication.setPalette(self.palettes[scheme])
        logger.info(f"Set color scheme to {scheme.name}")
    
    def set_font_scale(self, scale: float):
        """Set the font scale for the application.
        
        Args:
            scale: Font scale factor (1.0 = 100%)
        """
        self.font_scale = scale
        
        # Create a new font with the scaled size
        font = QFont(self.default_font)
        font.setPointSizeF(self.default_font.pointSizeF() * scale)
        
        # Apply the font to the application
        QApplication.setFont(font)
        logger.info(f"Set font scale to {scale:.2f}")
    
    def toggle_screen_reader_support(self, enabled: bool):
        """Toggle screen reader support.
        
        Args:
            enabled: Whether screen reader support is enabled
        """
        self.screen_reader_enabled = enabled
        
        # Set screen reader properties for all widgets
        if enabled:
            self._set_accessibility_properties(self.main_window)
        
        logger.info(f"{'Enabled' if enabled else 'Disabled'} screen reader support")
    
    def toggle_keyboard_focus_visible(self, visible: bool):
        """Toggle visibility of keyboard focus.
        
        Args:
            visible: Whether keyboard focus is visible
        """
        self.keyboard_focus_visible = visible
        
        # Set focus policy for all widgets
        if visible:
            self._set_tab_focus_for_all_widgets(self.main_window)
        
        logger.info(f"{'Enabled' if visible else 'Disabled'} keyboard focus visibility") 