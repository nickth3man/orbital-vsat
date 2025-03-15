"""
Export dialog for VSAT.

This module provides a dialog for exporting audio segments, transcripts, and other data.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QLineEdit, QPushButton, QFileDialog, QCheckBox, QGroupBox,
    QRadioButton, QButtonGroup, QSpinBox, QMessageBox, QTabWidget,
    QWidget, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from src.export.export_manager import ExportManager
from src.ui.export_tabs import (
    create_transcript_tab, create_audio_tab, 
    create_selection_tab, create_speaker_tab
)

logger = logging.getLogger(__name__)

class ExportDialog(QDialog):
    """Dialog for exporting audio segments, transcripts, and other data."""
    
    def __init__(self, parent=None, audio_file: str = None, segments: List[Dict[str, Any]] = None,
                selected_words: List[Dict[str, Any]] = None, selected_speaker: int = None):
        """Initialize the export dialog.
        
        Args:
            parent: Parent widget
            audio_file: Path to the audio file
            segments: List of transcript segments
            selected_words: List of selected words
            selected_speaker: ID of the selected speaker
        """
        super().__init__(parent)
        
        self.setWindowTitle("Export")
        self.setMinimumWidth(500)
        
        # Store parameters
        self.audio_file = audio_file
        self.segments = segments or []
        self.selected_words = selected_words or []
        self.selected_speaker = selected_speaker
        
        # Initialize export manager
        self.export_manager = ExportManager()
        
        # Initialize UI components
        self.transcript_path = None
        self.transcript_format = None
        self.include_speaker = None
        self.include_timestamps = None
        
        self.audio_path = None
        self.audio_format = None
        self.audio_start = None
        self.audio_end = None
        
        self.selection_path = None
        self.selection_format = None
        self.include_selection_transcript = None
        
        self.speaker_dir = None
        self.speaker_format = None
        self.speaker_id = None
        self.speaker_combo = None
        
        # Initialize UI
        self.init_ui()
        
        # Update UI state
        self.update_ui_state()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Create tabs
        transcript_tab = create_transcript_tab(self)
        audio_tab = create_audio_tab(self)
        selection_tab = create_selection_tab(self)
        speaker_tab = create_speaker_tab(self)
        
        # Add tabs to tab widget
        tab_widget.addTab(transcript_tab, "Transcript")
        tab_widget.addTab(audio_tab, "Audio Segment")
        tab_widget.addTab(selection_tab, "Selection")
        tab_widget.addTab(speaker_tab, "Speaker Audio")
        
        # Add tab widget to layout
        layout.addWidget(tab_widget)
        
        # Create buttons
        button_layout = QHBoxLayout()
        
        export_button = QPushButton("Export")
        export_button.clicked.connect(self.on_export)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(export_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def update_ui_state(self):
        """Update the UI state based on available data."""
        # Update speaker combo box
        self.update_speaker_combo()
    
    def update_speaker_combo(self):
        """Update the speaker combo box with available speakers."""
        if not self.speaker_combo or not self.segments:
            return
        
        # Clear combo box
        self.speaker_combo.clear()
        
        # Add "All Speakers" option
        self.speaker_combo.addItem("All Speakers", None)
        
        # Get unique speakers
        speakers = set()
        for segment in self.segments:
            if 'speaker' in segment and segment['speaker'] is not None:
                speakers.add(segment['speaker'])
        
        # Add speaker options
        for speaker in sorted(speakers):
            # Get speaker name
            speaker_name = next(
                (s.get('speaker_name', f"Speaker {s['speaker']}") 
                 for s in self.segments if s.get('speaker') == speaker),
                f"Speaker {speaker}"
            )
            
            # Add to combo box
            self.speaker_combo.addItem(f"{speaker_name}", speaker)
        
        # Set selected speaker if provided
        if self.selected_speaker is not None:
            for i in range(self.speaker_combo.count()):
                if self.speaker_combo.itemData(i) == self.selected_speaker:
                    self.speaker_combo.setCurrentIndex(i)
                    break
    
    @pyqtSlot()
    def on_browse_transcript(self):
        """Handle browse button click for transcript export."""
        # Get selected format
        format_key = self.transcript_format.currentData()
        
        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Transcript",
            f"transcript.{format_key}",
            f"{format_key.upper()} Files (*.{format_key});;All Files (*.*)"
        )
        
        if file_path:
            self.transcript_path.setText(file_path)
    
    @pyqtSlot()
    def on_browse_audio(self):
        """Handle browse button click for audio export."""
        # Get selected format
        format_key = self.audio_format.currentData()
        
        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Audio Segment",
            f"segment.{format_key}",
            f"{format_key.upper()} Files (*.{format_key});;All Files (*.*)"
        )
        
        if file_path:
            self.audio_path.setText(file_path)
    
    @pyqtSlot()
    def on_browse_selection(self):
        """Handle browse button click for selection export."""
        # Get selected format
        format_key = self.selection_format.currentData()
        
        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Selection",
            f"selection.{format_key}",
            f"{format_key.upper()} Files (*.{format_key});;All Files (*.*)"
        )
        
        if file_path:
            self.selection_path.setText(file_path)
    
    @pyqtSlot()
    def on_browse_speaker(self):
        """Handle browse button click for speaker export."""
        # Show directory dialog
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Export Speaker Audio",
            ""
        )
        
        if dir_path:
            self.speaker_dir.setText(dir_path)
    
    @pyqtSlot()
    def on_export(self):
        """Handle export button click."""
        # Get active tab
        tab_widget = self.findChild(QTabWidget)
        active_tab = tab_widget.currentIndex()
        
        # Export based on active tab
        if active_tab == 0:
            self.export_transcript()
        elif active_tab == 1:
            self.export_audio_segment()
        elif active_tab == 2:
            self.export_selection()
        elif active_tab == 3:
            self.export_speaker()
    
    def export_transcript(self):
        """Export transcript."""
        # Check if we have segments to export
        if not self.segments:
            QMessageBox.warning(self, "Export Error", "No transcript available to export.")
            return
        
        # Get export parameters
        output_path = self.transcript_path.text()
        format_key = self.transcript_format.currentData()
        include_speaker = self.include_speaker.isChecked()
        include_timestamps = self.include_timestamps.isChecked()
        
        # Validate parameters
        if not output_path:
            QMessageBox.warning(self, "Export Error", "Please specify an output file.")
            return
        
        # Export transcript
        try:
            success = self.export_manager.export_transcript(
                self.segments,
                output_path,
                format_key,
                include_speaker,
                include_timestamps
            )
            
            if success:
                QMessageBox.information(self, "Export Complete", f"Transcript exported to {output_path}")
                self.accept()
            else:
                QMessageBox.warning(self, "Export Error", "Failed to export transcript.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting transcript: {str(e)}")
    
    def export_audio_segment(self):
        """Export audio segment."""
        # Check if we have an audio file
        if not self.audio_file:
            QMessageBox.warning(self, "Export Error", "No audio file available to export.")
            return
        
        # Get export parameters
        output_path = self.audio_path.text()
        format_key = self.audio_format.currentData()
        start = self.audio_start.value()
        end = self.audio_end.value()
        
        # Validate parameters
        if not output_path:
            QMessageBox.warning(self, "Export Error", "Please specify an output file.")
            return
        
        if start >= end:
            QMessageBox.warning(self, "Export Error", "Start time must be less than end time.")
            return
        
        # Export audio segment
        try:
            success = self.export_manager.export_audio_segment(
                self.audio_file,
                output_path,
                start,
                end,
                format_key
            )
            
            if success:
                QMessageBox.information(
                    self, 
                    "Export Complete", 
                    f"Audio segment exported to {output_path} ({start:.2f}s to {end:.2f}s)"
                )
                self.accept()
            else:
                QMessageBox.warning(self, "Export Error", "Failed to export audio segment.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting audio segment: {str(e)}")
    
    def export_selection(self):
        """Export selection."""
        # Check if we have an audio file and selected words
        if not self.audio_file:
            QMessageBox.warning(self, "Export Error", "No audio file available to export.")
            return
        
        if not self.selected_words:
            QMessageBox.warning(self, "Export Error", "No words selected to export.")
            return
        
        # Get export parameters
        output_path = self.selection_path.text()
        format_key = self.selection_format.currentData()
        include_transcript = self.include_selection_transcript.isChecked()
        
        # Validate parameters
        if not output_path:
            QMessageBox.warning(self, "Export Error", "Please specify an output file.")
            return
        
        # Export selection
        try:
            success = self.export_manager.export_selection(
                self.audio_file,
                self.selected_words,
                output_path,
                format_key,
                include_transcript
            )
            
            if success:
                QMessageBox.information(self, "Export Complete", f"Selection exported to {output_path}")
                self.accept()
            else:
                QMessageBox.warning(self, "Export Error", "Failed to export selection.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting selection: {str(e)}")
    
    def export_speaker(self):
        """Export speaker audio."""
        # Check if we have an audio file and segments
        if not self.audio_file or not self.segments:
            QMessageBox.warning(self, "Export Error", "No audio file or transcript available to export.")
            return
        
        # Get export parameters
        output_dir = self.speaker_dir.text()
        format_key = self.speaker_format.currentData()
        speaker_id = self.speaker_combo.currentData()
        
        # Validate parameters
        if not output_dir:
            QMessageBox.warning(self, "Export Error", "Please specify an output directory.")
            return
        
        # Export speaker audio
        try:
            success = self.export_manager.export_speaker_audio(
                self.audio_file,
                self.segments,
                output_dir,
                speaker_id,
                format_key
            )
            
            if success:
                QMessageBox.information(self, "Export Complete", f"Speaker audio exported to {output_dir}")
                self.accept()
            else:
                QMessageBox.warning(self, "Export Error", "Failed to export speaker audio.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting speaker audio: {str(e)}") 