#!/usr/bin/env python3
"""
Demo launcher for VSAT.

This script provides a simple UI to launch different VSAT demos.
"""

import sys
import os
import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, 
    QPushButton, QLabel, QGridLayout, QGroupBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.demos.vad_demo import VADDemoWindow
from src.demos.content_analysis_demo import ContentAnalysisDemo

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoLauncher(QMainWindow):
    """Main window for launching different demos."""
    
    def __init__(self):
        """Initialize the demo launcher."""
        super().__init__()
        
        # Set up UI
        self.setWindowTitle("VSAT - Demo Launcher")
        self.setGeometry(100, 100, 600, 400)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        main_layout = QVBoxLayout(central_widget)
        
        # Add title
        title_label = QLabel("VSAT Demo Launcher")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # Add description
        desc_label = QLabel(
            "Select a demo to launch from the options below."
        )
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(desc_label)
        
        # Create demo grid
        demos_group = QGroupBox("Available Demos")
        demos_layout = QGridLayout(demos_group)
        
        # VAD Demo
        vad_button = QPushButton("Voice Activity Detection")
        vad_button.setMinimumSize(QSize(200, 80))
        vad_button.clicked.connect(self._launch_vad_demo)
        vad_desc = QLabel(
            "Detect speech segments in audio files.\n"
            "Adjust sensitivity, visualize results,\n"
            "and export speech segments."
        )
        demos_layout.addWidget(vad_button, 0, 0)
        demos_layout.addWidget(vad_desc, 0, 1)
        
        # Content Analysis Demo
        content_button = QPushButton("Content Analysis")
        content_button.setMinimumSize(QSize(200, 80))
        content_button.clicked.connect(self._launch_content_analysis_demo)
        content_desc = QLabel(
            "Analyze transcript content.\n"
            "Extract topics, keywords, and sentiment.\n"
            "Identify important moments."
        )
        demos_layout.addWidget(content_button, 1, 0)
        demos_layout.addWidget(content_desc, 1, 1)
        
        # Add more demos here in the future
        
        main_layout.addWidget(demos_group)
        
        # Add help text
        help_label = QLabel(
            "Note: These demos showcase individual components of VSAT.\n"
            "The full application integrates all these features in a unified workflow."
        )
        help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        help_label.setStyleSheet("color: gray;")
        main_layout.addWidget(help_label)
    
    def _launch_vad_demo(self):
        """Launch the Voice Activity Detection demo."""
        self.vad_demo = VADDemoWindow()
        self.vad_demo.show()
    
    def _launch_content_analysis_demo(self):
        """Launch the Content Analysis demo."""
        self.content_demo = ContentAnalysisDemo()
        self.content_demo.show()

def main():
    """Run the demo launcher."""
    app = QApplication(sys.argv)
    launcher = DemoLauncher()
    launcher.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 