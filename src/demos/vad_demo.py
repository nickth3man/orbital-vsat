"""
Demo script for Voice Activity Detection functionality.

This script demonstrates the VAD capabilities of VSAT with a simple UI.
"""

import sys
import os
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLabel
from PyQt6.QtCore import Qt

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.audio.processor import AudioProcessor
from src.ui.vad_visualization import VADVisualizationWidget

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VADDemoWindow(QMainWindow):
    """Main window for the VAD demo application."""
    
    def __init__(self):
        """Initialize the demo window."""
        super().__init__()
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        # Set up UI
        self.setWindowTitle("VSAT - Voice Activity Detection Demo")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Add title
        title_label = QLabel("Voice Activity Detection Demo")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # Add description
        desc_label = QLabel(
            "This demo shows the Voice Activity Detection capabilities of VSAT.\n"
            "Load an audio file and adjust sensitivity to detect speech segments."
        )
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)
        
        # Add file selection button
        self.load_button = QPushButton("Load Audio File")
        self.load_button.clicked.connect(self.load_audio_file)
        layout.addWidget(self.load_button)
        
        # Add VAD visualization widget
        self.vad_widget = VADVisualizationWidget(self.audio_processor)
        layout.addWidget(self.vad_widget, 1)
    
    def load_audio_file(self):
        """Load an audio file for processing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        
        if file_path:
            logger.info(f"Loading audio file: {file_path}")
            self.vad_widget.set_file(file_path)

def main():
    """Run the VAD demo application."""
    app = QApplication(sys.argv)
    window = VADDemoWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 