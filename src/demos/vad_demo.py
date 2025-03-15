"""
Demo script for Voice Activity Detection functionality.

This script demonstrates the VAD capabilities of VSAT with an interactive UI.
"""

import sys
import os
import logging
from pathlib import Path
import tempfile

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QPushButton, QFileDialog, QLabel, QSlider, QComboBox, QCheckBox,
    QGroupBox, QSplitter, QToolBar, QStatusBar, QDialog, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.audio.processor import AudioProcessor
from src.audio.audio_player import AudioPlayer
from src.ui.vad_visualization import VADVisualizationWidget
from src.ml.voice_activity_detection import VoiceActivityDetector

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedSettingsDialog(QDialog):
    """Dialog for advanced VAD settings."""
    
    def __init__(self, vad_detector, parent=None):
        """Initialize the advanced settings dialog."""
        super().__init__(parent)
        self.vad_detector = vad_detector
        self.settings = vad_detector.DEFAULT_SETTINGS.copy()
        
        self.setWindowTitle("Advanced VAD Settings")
        self.setMinimumWidth(400)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QGridLayout(self)
        
        # Energy threshold (0.0-1.0)
        layout.addWidget(QLabel("Energy Threshold:"), 0, 0)
        self.energy_slider = QSlider(Qt.Orientation.Horizontal)
        self.energy_slider.setRange(0, 100)
        self.energy_slider.setValue(int(self.settings["energy_threshold"] * 100))
        self.energy_slider.valueChanged.connect(self._update_energy_label)
        layout.addWidget(self.energy_slider, 0, 1)
        self.energy_label = QLabel(f"{self.settings['energy_threshold']:.2f}")
        layout.addWidget(self.energy_label, 0, 2)
        
        # Min speech duration (ms)
        layout.addWidget(QLabel("Min Speech Duration (ms):"), 1, 0)
        self.speech_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.speech_duration_slider.setRange(50, 1000)
        self.speech_duration_slider.setValue(self.settings["min_speech_duration_ms"])
        self.speech_duration_slider.valueChanged.connect(self._update_speech_duration_label)
        layout.addWidget(self.speech_duration_slider, 1, 1)
        self.speech_duration_label = QLabel(f"{self.settings['min_speech_duration_ms']} ms")
        layout.addWidget(self.speech_duration_label, 1, 2)
        
        # Min silence duration (ms)
        layout.addWidget(QLabel("Min Silence Duration (ms):"), 2, 0)
        self.silence_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.silence_duration_slider.setRange(50, 1000)
        self.silence_duration_slider.setValue(self.settings["min_silence_duration_ms"])
        self.silence_duration_slider.valueChanged.connect(self._update_silence_duration_label)
        layout.addWidget(self.silence_duration_slider, 2, 1)
        self.silence_duration_label = QLabel(f"{self.settings['min_silence_duration_ms']} ms")
        layout.addWidget(self.silence_duration_label, 2, 2)
        
        # Speech padding (ms)
        layout.addWidget(QLabel("Speech Padding (ms):"), 3, 0)
        self.padding_slider = QSlider(Qt.Orientation.Horizontal)
        self.padding_slider.setRange(0, 200)
        self.padding_slider.setValue(self.settings["speech_pad_ms"])
        self.padding_slider.valueChanged.connect(self._update_padding_label)
        layout.addWidget(self.padding_slider, 3, 1)
        self.padding_label = QLabel(f"{self.settings['speech_pad_ms']} ms")
        layout.addWidget(self.padding_label, 3, 2)
        
        # Method selection
        layout.addWidget(QLabel("Detection Method:"), 4, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["ML-based", "Energy-based"])
        self.method_combo.setCurrentIndex(0 if self.settings["use_model"] else 1)
        layout.addWidget(self.method_combo, 4, 1, 1, 2)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_button)
        
        ok_button = QPushButton("Apply")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout, 5, 0, 1, 3)
    
    def _update_energy_label(self, value):
        """Update the energy threshold label."""
        energy = value / 100.0
        self.energy_label.setText(f"{energy:.2f}")
        self.settings["energy_threshold"] = energy
    
    def _update_speech_duration_label(self, value):
        """Update the speech duration label."""
        self.speech_duration_label.setText(f"{value} ms")
        self.settings["min_speech_duration_ms"] = value
    
    def _update_silence_duration_label(self, value):
        """Update the silence duration label."""
        self.silence_duration_label.setText(f"{value} ms")
        self.settings["min_silence_duration_ms"] = value
    
    def _update_padding_label(self, value):
        """Update the padding label."""
        self.padding_label.setText(f"{value} ms")
        self.settings["speech_pad_ms"] = value
    
    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.settings = self.vad_detector.DEFAULT_SETTINGS.copy()
        self.energy_slider.setValue(int(self.settings["energy_threshold"] * 100))
        self.speech_duration_slider.setValue(self.settings["min_speech_duration_ms"])
        self.silence_duration_slider.setValue(self.settings["min_silence_duration_ms"])
        self.padding_slider.setValue(self.settings["speech_pad_ms"])
        self.method_combo.setCurrentIndex(0 if self.settings["use_model"] else 1)
    
    def get_settings(self):
        """Get the current settings."""
        self.settings["use_model"] = (self.method_combo.currentIndex() == 0)
        return self.settings

class VADDemoWindow(QMainWindow):
    """Main window for the VAD demo application."""
    
    def __init__(self):
        """Initialize the demo window."""
        super().__init__()
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.audio_player = AudioPlayer()
        self.vad_detector = VoiceActivityDetector()
        
        # Current file and segments
        self.current_file = None
        self.segments = []
        self.current_segment = None
        
        # Set up UI
        self.setWindowTitle("VSAT - Voice Activity Detection Demo")
        self.setGeometry(100, 100, 1000, 700)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        main_layout = QVBoxLayout(central_widget)
        
        # Add title
        title_label = QLabel("Voice Activity Detection Demo")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # Create toolbar
        self._create_toolbar()
        
        # Create main content area with splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top section: VAD visualization
        self.vad_widget = VADVisualizationWidget(self.audio_processor)
        splitter.addWidget(self.vad_widget)
        
        # Bottom section: Controls and settings
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Audio controls group
        audio_group = QGroupBox("Audio Controls")
        audio_layout = QHBoxLayout(audio_group)
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_playback)
        self.play_button.setEnabled(False)
        audio_layout.addWidget(self.play_button)
        
        self.play_segment_button = QPushButton("Play Selected Segment")
        self.play_segment_button.clicked.connect(self._play_selected_segment)
        self.play_segment_button.setEnabled(False)
        audio_layout.addWidget(self.play_segment_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_playback)
        self.stop_button.setEnabled(False)
        audio_layout.addWidget(self.stop_button)
        
        # Add volume control
        audio_layout.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self._set_volume)
        audio_layout.addWidget(self.volume_slider)
        
        controls_layout.addWidget(audio_group)
        
        # VAD settings group
        vad_group = QGroupBox("VAD Settings")
        vad_layout = QHBoxLayout(vad_group)
        
        vad_layout.addWidget(QLabel("Sensitivity Preset:"))
        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItems(self.vad_detector.get_available_presets())
        self.sensitivity_combo.setCurrentText("medium")
        vad_layout.addWidget(self.sensitivity_combo)
        
        self.detect_button = QPushButton("Detect Speech")
        self.detect_button.clicked.connect(self._detect_speech)
        self.detect_button.setEnabled(False)
        vad_layout.addWidget(self.detect_button)
        
        self.advanced_button = QPushButton("Advanced Settings")
        self.advanced_button.clicked.connect(self._show_advanced_settings)
        vad_layout.addWidget(self.advanced_button)
        
        controls_layout.addWidget(vad_group)
        
        # Export group
        export_group = QGroupBox("Export Options")
        export_layout = QHBoxLayout(export_group)
        
        self.export_all_button = QPushButton("Export All Segments")
        self.export_all_button.clicked.connect(self._export_all_segments)
        self.export_all_button.setEnabled(False)
        export_layout.addWidget(self.export_all_button)
        
        self.export_selected_button = QPushButton("Export Selected Segment")
        self.export_selected_button.clicked.connect(self._export_selected_segment)
        self.export_selected_button.setEnabled(False)
        export_layout.addWidget(self.export_selected_button)
        
        controls_layout.addWidget(export_group)
        
        splitter.addWidget(controls_widget)
        
        # Set splitter proportions
        splitter.setSizes([600, 200])
        
        main_layout.addWidget(splitter)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load an audio file to begin.")
        
        # Connect VAD widget signals
        self.vad_widget.segment_widget.segmentClicked.connect(self._on_segment_clicked)
        
        # Connect audio player signals
        self.audio_player.playback_state_changed.connect(self._update_playback_ui)
        
        # Set initial volume
        self._set_volume(self.volume_slider.value())
    
    def _create_toolbar(self):
        """Create the application toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Load file action
        load_action = QAction("Load Audio", self)
        load_action.triggered.connect(self._load_audio_file)
        toolbar.addAction(load_action)
        
        toolbar.addSeparator()
        
        # Help action
        help_action = QAction("Help", self)
        help_action.triggered.connect(self._show_help)
        toolbar.addAction(help_action)
    
    def _load_audio_file(self):
        """Load an audio file for processing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        
        if file_path:
            try:
                logger.info(f"Loading audio file: {file_path}")
                self.current_file = file_path
                self.vad_widget.set_file(file_path)
                
                # Enable controls
                self.play_button.setEnabled(True)
                self.detect_button.setEnabled(True)
                
                # Load the file in the audio player
                self.audio_player.load_file(file_path)
                
                # Update status
                file_name = os.path.basename(file_path)
                duration = self.audio_player.get_duration()
                self.status_bar.showMessage(f"Loaded: {file_name} ({duration:.2f} seconds)")
            except Exception as e:
                logger.error(f"Error loading file: {e}")
                self.status_bar.showMessage(f"Error loading file: {str(e)}")
    
    def _detect_speech(self):
        """Detect speech segments using current settings."""
        if not self.current_file:
            return
        
        # Show progress in status bar
        self.status_bar.showMessage("Detecting speech segments...")
        self.detect_button.setEnabled(False)
        
        # Get selected sensitivity preset or advanced settings
        sensitivity = self.sensitivity_combo.currentText()
        
        # Start detection
        self.vad_widget.detect_speech()
        
        # Update UI after detection
        self.detect_button.setEnabled(True)
        self.export_all_button.setEnabled(True)
        
        # Get segments for our use
        self.segments = self.vad_widget.segments
    
    def _toggle_playback(self):
        """Toggle audio playback."""
        if not self.current_file:
            return
        
        if self.audio_player.is_playing():
            self.audio_player.pause()
        else:
            self.audio_player.play(self.current_file)
            self.status_bar.showMessage("Playing audio...")
    
    def _stop_playback(self):
        """Stop audio playback."""
        self.audio_player.stop()
        self.status_bar.showMessage("Playback stopped")
    
    def _play_selected_segment(self):
        """Play the currently selected speech segment."""
        if not self.current_segment:
            return
        
        start = self.current_segment["start"]
        end = self.current_segment["end"]
        
        self.audio_player.play(self.current_file, start, end)
        self.status_bar.showMessage(f"Playing segment: {start:.2f}s - {end:.2f}s")
    
    def _set_volume(self, value):
        """Set the audio player volume."""
        volume = value / 100.0
        self.audio_player.set_volume(volume)
    
    def _update_playback_ui(self, is_playing):
        """Update UI based on playback state."""
        self.play_button.setText("Pause" if is_playing else "Play")
        self.stop_button.setEnabled(is_playing)
    
    def _on_segment_clicked(self, segment):
        """Handle segment click event."""
        self.current_segment = segment
        self.play_segment_button.setEnabled(True)
        self.export_selected_button.setEnabled(True)
        
        # Update status bar
        start = segment["start"]
        end = segment["end"]
        confidence = segment.get("confidence", 0)
        self.status_bar.showMessage(f"Selected segment: {start:.2f}s - {end:.2f}s (Confidence: {confidence:.2f})")
    
    def _show_advanced_settings(self):
        """Show advanced VAD settings dialog."""
        dialog = AdvancedSettingsDialog(self.vad_detector, self)
        if dialog.exec():
            # Apply custom settings
            custom_settings = dialog.get_settings()
            self.vad_widget.audio_processor.set_vad_settings(custom_settings)
            self.status_bar.showMessage("Applied custom VAD settings")
    
    def _export_all_segments(self):
        """Export all detected speech segments."""
        if not self.segments:
            return
        
        # Ask for export directory
        export_dir = QFileDialog.getExistingDirectory(
            self, 
            "Select Export Directory",
            ""
        )
        
        if not export_dir:
            return
        
        try:
            # Load audio data
            audio_data, sample_rate = self.audio_processor.file_handler.load_file(self.current_file)
            
            # Create base filename
            base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            
            # Export each segment
            for i, segment in enumerate(self.segments):
                start_sample = int(segment["start"] * sample_rate)
                end_sample = int(segment["end"] * sample_rate)
                
                # Extract segment audio
                segment_audio = audio_data[start_sample:end_sample]
                
                # Create output filename
                output_file = os.path.join(export_dir, f"{base_name}_segment_{i+1}.wav")
                
                # Save segment
                self.audio_processor.file_handler.save_file(
                    output_file, segment_audio, sample_rate
                )
            
            self.status_bar.showMessage(f"Exported {len(self.segments)} segments to {export_dir}")
        except Exception as e:
            logger.error(f"Error exporting segments: {e}")
            self.status_bar.showMessage(f"Error exporting segments: {str(e)}")
    
    def _export_selected_segment(self):
        """Export the selected speech segment."""
        if not self.current_segment:
            return
        
        # Ask for export file name
        export_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Segment As",
            "",
            "WAV Files (*.wav);;All Files (*)"
        )
        
        if not export_file:
            return
        
        try:
            # Load audio data
            audio_data, sample_rate = self.audio_processor.file_handler.load_file(self.current_file)
            
            # Extract segment
            start_sample = int(self.current_segment["start"] * sample_rate)
            end_sample = int(self.current_segment["end"] * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            # Save segment
            self.audio_processor.file_handler.save_file(
                export_file, segment_audio, sample_rate
            )
            
            self.status_bar.showMessage(f"Exported selected segment to {export_file}")
        except Exception as e:
            logger.error(f"Error exporting segment: {e}")
            self.status_bar.showMessage(f"Error exporting segment: {str(e)}")
    
    def _show_help(self):
        """Show help information."""
        help_text = (
            "<h3>VSAT Voice Activity Detection Demo</h3>"
            "<p>This demo shows how to detect speech segments in audio files.</p>"
            "<h4>Instructions:</h4>"
            "<ol>"
            "<li>Load an audio file using the toolbar or File menu</li>"
            "<li>Select a sensitivity preset or configure advanced settings</li>"
            "<li>Click 'Detect Speech' to analyze the audio</li>"
            "<li>Click on segments to select them</li>"
            "<li>Use the audio controls to play the entire file or selected segments</li>"
            "<li>Export segments individually or all at once</li>"
            "</ol>"
        )
        
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Help", help_text)
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop playback if playing
        if self.audio_player.is_playing():
            self.audio_player.stop()
        
        # Accept the close event
        event.accept()

def main():
    """Run the VAD demo application."""
    app = QApplication(sys.argv)
    window = VADDemoWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 