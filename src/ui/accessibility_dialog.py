"""
Accessibility settings dialog for VSAT.

This module provides a dialog for configuring accessibility settings.
"""

import logging
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QCheckBox, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from .accessibility import ColorScheme, AccessibilityManager

logger = logging.getLogger(__name__)

class AccessibilityDialog(QDialog):
    """Dialog for configuring accessibility settings."""
    
    # Signal emitted when settings are applied
    settings_applied = pyqtSignal()
    
    def __init__(self, accessibility_manager: AccessibilityManager, parent=None):
        """Initialize the dialog.
        
        Args:
            accessibility_manager: Accessibility manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.accessibility_manager = accessibility_manager
        
        self.setWindowTitle("Accessibility Settings")
        self.setMinimumWidth(400)
        
        self._init_ui()
        self._load_current_settings()
    
    def _init_ui(self):
        """Initialize the dialog UI."""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Color scheme group
        color_group = QGroupBox("Color Scheme")
        color_layout = QFormLayout()
        
        self.color_combo = QComboBox()
        self.color_combo.addItem("Standard", ColorScheme.STANDARD)
        self.color_combo.addItem("High Contrast", ColorScheme.HIGH_CONTRAST)
        self.color_combo.addItem("Dark", ColorScheme.DARK)
        self.color_combo.addItem("Light", ColorScheme.LIGHT)
        
        color_layout.addRow("Color scheme:", self.color_combo)
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        # Font scaling group
        font_group = QGroupBox("Font Size")
        font_layout = QVBoxLayout()
        
        font_slider_layout = QHBoxLayout()
        self.font_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_slider.setMinimum(50)
        self.font_slider.setMaximum(200)
        self.font_slider.setValue(100)
        self.font_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.font_slider.setTickInterval(25)
        
        self.font_value_label = QLabel("100%")
        self.font_slider.valueChanged.connect(self._update_font_label)
        
        font_slider_layout.addWidget(QLabel("Smaller"))
        font_slider_layout.addWidget(self.font_slider)
        font_slider_layout.addWidget(QLabel("Larger"))
        font_slider_layout.addWidget(self.font_value_label)
        
        font_layout.addLayout(font_slider_layout)
        font_group.setLayout(font_layout)
        layout.addWidget(font_group)
        
        # Screen reader group
        screen_reader_group = QGroupBox("Screen Reader Support")
        screen_reader_layout = QVBoxLayout()
        
        self.screen_reader_checkbox = QCheckBox("Enable screen reader support")
        self.screen_reader_checkbox.setToolTip(
            "Enhances compatibility with screen readers by providing additional accessibility information"
        )
        
        screen_reader_layout.addWidget(self.screen_reader_checkbox)
        screen_reader_group.setLayout(screen_reader_layout)
        layout.addWidget(screen_reader_group)
        
        # Keyboard navigation group
        keyboard_group = QGroupBox("Keyboard Navigation")
        keyboard_layout = QVBoxLayout()
        
        self.keyboard_focus_checkbox = QCheckBox("Show keyboard focus indicators")
        self.keyboard_focus_checkbox.setToolTip(
            "Highlights the currently focused element when navigating with the keyboard"
        )
        
        keyboard_layout.addWidget(self.keyboard_focus_checkbox)
        keyboard_group.setLayout(keyboard_layout)
        layout.addWidget(keyboard_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._apply_settings)
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._ok_clicked)
        self.ok_button.setDefault(True)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def _load_current_settings(self):
        """Load current settings from the accessibility manager."""
        # Set color scheme
        index = self.color_combo.findData(self.accessibility_manager.color_scheme)
        if index >= 0:
            self.color_combo.setCurrentIndex(index)
        
        # Set font scale
        font_scale = int(self.accessibility_manager.font_scale * 100)
        self.font_slider.setValue(font_scale)
        
        # Set screen reader support
        self.screen_reader_checkbox.setChecked(self.accessibility_manager.screen_reader_enabled)
        
        # Set keyboard focus visibility
        self.keyboard_focus_checkbox.setChecked(self.accessibility_manager.keyboard_focus_visible)
    
    def _update_font_label(self, value):
        """Update the font size label when the slider changes.
        
        Args:
            value: New slider value
        """
        self.font_value_label.setText(f"{value}%")
    
    def _apply_settings(self):
        """Apply the selected settings."""
        # Get color scheme
        color_scheme = self.color_combo.currentData()
        self.accessibility_manager.set_color_scheme(color_scheme)
        
        # Get font scale
        font_scale = self.font_slider.value() / 100.0
        self.accessibility_manager.set_font_scale(font_scale)
        
        # Get screen reader support
        screen_reader_enabled = self.screen_reader_checkbox.isChecked()
        self.accessibility_manager.toggle_screen_reader_support(screen_reader_enabled)
        
        # Get keyboard focus visibility
        keyboard_focus_visible = self.keyboard_focus_checkbox.isChecked()
        self.accessibility_manager.toggle_keyboard_focus_visible(keyboard_focus_visible)
        
        # Emit signal
        self.settings_applied.emit()
        
        logger.info("Applied accessibility settings")
    
    def _ok_clicked(self):
        """Handle OK button click."""
        self._apply_settings()
        self.accept() 