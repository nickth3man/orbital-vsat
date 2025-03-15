"""
Export handlers for the VSAT UI.

This module provides handlers for exporting transcripts, audio segments, and other data.
"""

import os
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QDialogButtonBox,
    QMessageBox, QFileDialog, QInputDialog, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from src.export.export_manager import ExportManager
from src.utils.error_handler import ErrorHandler, ExportError, FileError, ErrorSeverity
from src.ui.export_error_tracker import ExportErrorTracker, ExportOperation, ExportAttempt
from src.ui.export_error_dialog import ExportErrorDialog

logger = logging.getLogger(__name__)

class ExportHandlers:
    """Handles all export functionality for the VSAT UI."""
    
    def __init__(self, parent):
        """Initialize the export handlers.
        
        Args:
            parent: Parent window (MainWindow)
        """
        self.parent = parent
        self.export_manager = ExportManager()
        self.error_tracker = ExportErrorTracker()
        
        # Connect error tracker signals
        self.error_tracker.exportFailed.connect(self._on_export_failed)
        
        # Create error dialog
        self.error_dialog = ExportErrorDialog(parent)
        self.error_dialog.retryAllRequested.connect(self._on_retry_exports)
    
    def export_transcript(self):
        """Export the transcript to a file."""
        try:
            # Check if we have segments to export
            if not self.parent.segments:
                raise ExportError(
                    "No transcript available to export",
                    ErrorSeverity.WARNING,
                    {"method": "export_transcript"}
                )
            
            # Get export format
            formats = list(self.export_manager.TRANSCRIPT_FORMATS.items())
            format_names = [f"{key.upper()} - {desc}" for key, desc in formats]
            
            format_idx, ok = QInputDialog.getItem(
                self.parent,
                "Export Transcript",
                "Select export format:",
                format_names,
                0,
                False
            )
            
            if not ok:
                return
                
            format_key = formats[format_names.index(format_idx)][0]
            
            # Get file path
            file_filter = f"{format_key.upper()} Files (*.{format_key})"
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent,
                "Export Transcript",
                str(Path.home()),
                file_filter
            )
            
            if not file_path:
                return
                
            # Add extension if missing
            if not file_path.lower().endswith(f".{format_key}"):
                file_path += f".{format_key}"
                
            # Show export options dialog
            options = {}
            
            if format_key in ["srt", "vtt"]:
                options = self._get_subtitle_options()
                if options is None:
                    return
            elif format_key in ["json", "csv"]:
                options = self._get_structured_options()
                if options is None:
                    return
            
            # Start tracking export operation
            self.error_tracker.start_tracking(
                ExportOperation.TRANSCRIPT,
                target_path=file_path
            )
            
            # Export in a separate thread to avoid UI freezing
            def export_thread():
                try:
                    # Export transcript
                    self.export_manager.export_transcript(
                        self.parent.segments,
                        file_path,
                        format_key,
                        options
                    )
                    
                    # Mark as success
                    self.error_tracker.mark_success()
                    
                    # Show success message
                    self.parent.show_message(
                        "Export Successful",
                        f"Transcript exported to {file_path}",
                        QMessageBox.Icon.Information
                    )
                    
                except Exception as e:
                    # Mark as failed
                    self.error_tracker.mark_failure(e)
                    
                    # Show error message
                    ErrorHandler.handle_exception(e)
            
            # Start thread
            thread = threading.Thread(target=export_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            ErrorHandler.handle_exception(e)
    
    def export_audio_segment(self):
        """Export an audio segment to a file."""
        try:
            # Check if we have audio data to export
            if not self.parent.audio_processor or not self.parent.audio_processor.audio:
                raise ExportError(
                    "No audio data available to export",
                    ErrorSeverity.WARNING,
                    {"method": "export_audio_segment"}
                )
            
            # Get export format
            formats = list(self.export_manager.AUDIO_FORMATS.items())
            format_names = [f"{key.upper()} - {desc}" for key, desc in formats]
            
            format_idx, ok = QInputDialog.getItem(
                self.parent,
                "Export Audio Segment",
                "Select export format:",
                format_names,
                0,
                False
            )
            
            if not ok:
                return
                
            format_key = formats[format_names.index(format_idx)][0]
            
            # Get file path
            file_filter = f"{format_key.upper()} Files (*.{format_key})"
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent,
                "Export Audio Segment",
                str(Path.home()),
                file_filter
            )
            
            if not file_path:
                return
                
            # Add extension if missing
            if not file_path.lower().endswith(f".{format_key}"):
                file_path += f".{format_key}"
                
            # Start tracking export operation
            self.error_tracker.start_tracking(
                ExportOperation.AUDIO_SEGMENT,
                target_path=file_path
            )
            
            # Export in a separate thread to avoid UI freezing
            def export_thread():
                try:
                    # Export audio segment
                    self.export_manager.export_audio(
                        self.parent.audio_processor.audio,
                        self.parent.audio_processor.sample_rate,
                        file_path,
                        format_key
                    )
                    
                    # Mark as success
                    self.error_tracker.mark_success()
                    
                    # Show success message
                    self.parent.show_message(
                        "Export Successful",
                        f"Audio segment exported to {file_path}",
                        QMessageBox.Icon.Information
                    )
                    
                except Exception as e:
                    # Mark as failed
                    self.error_tracker.mark_failure(e)
                    
                    # Show error message
                    ErrorHandler.handle_exception(e)
            
            # Start thread
            thread = threading.Thread(target=export_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            ErrorHandler.handle_exception(e)
    
    def export_speaker_audio(self):
        """Export audio segments for a specific speaker."""
        try:
            # Check if we have segments and audio data
            if not self.parent.segments or not self.parent.audio_processor:
                raise ExportError(
                    "No segments or audio data available to export",
                    ErrorSeverity.WARNING,
                    {"method": "export_speaker_audio"}
                )
            
            # Get list of unique speaker IDs
            speakers = set()
            for segment in self.parent.segments:
                if "speaker" in segment and segment["speaker"]:
                    speakers.add(segment["speaker"])
            
            if not speakers:
                raise ExportError(
                    "No speaker information available in segments",
                    ErrorSeverity.WARNING,
                    {"method": "export_speaker_audio"}
                )
            
            # Get the speaker to export
            speaker_list = sorted(list(speakers))
            
            speaker_idx, ok = QInputDialog.getItem(
                self.parent,
                "Export Speaker Audio",
                "Select speaker:",
                [str(s) for s in speaker_list],
                0,
                False
            )
            
            if not ok:
                return
                
            selected_speaker = speaker_list[speaker_list.index(int(speaker_idx) if speaker_idx.isdigit() else speaker_idx)]
            
            # Get export format
            formats = list(self.export_manager.AUDIO_FORMATS.items())
            format_names = [f"{key.upper()} - {desc}" for key, desc in formats]
            
            format_idx, ok = QInputDialog.getItem(
                self.parent,
                "Export Speaker Audio",
                "Select export format:",
                format_names,
                0,
                False
            )
            
            if not ok:
                return
                
            format_key = formats[format_names.index(format_idx)][0]
            
            # Get export directory
            directory = QFileDialog.getExistingDirectory(
                self.parent,
                "Select Export Directory",
                str(Path.home())
            )
            
            if not directory:
                return
                
            # Start tracking export operation
            self.error_tracker.start_tracking(
                ExportOperation.SPEAKER_AUDIO,
                target_path=directory
            )
            
            # Filter segments for selected speaker
            speaker_segments = [s for s in self.parent.segments if s.get("speaker") == selected_speaker]
            
            # Export in a separate thread
            def export_thread():
                try:
                    # Export each segment
                    exported_files = []
                    
                    for i, segment in enumerate(speaker_segments):
                        # Generate file name
                        file_name = f"speaker_{selected_speaker}_segment_{i+1:03d}.{format_key}"
                        file_path = os.path.join(directory, file_name)
                        
                        # Export segment
                        self.export_manager.export_segment(
                            self.parent.audio_processor.audio,
                            self.parent.audio_processor.sample_rate,
                            segment["start"],
                            segment["end"],
                            file_path,
                            format_key
                        )
                        
                        exported_files.append(file_path)
                    
                    # Mark as success
                    self.error_tracker.mark_success()
                    
                    # Show success message
                    self.parent.show_message(
                        "Export Successful",
                        f"Exported {len(exported_files)} audio segments for speaker {selected_speaker}",
                        QMessageBox.Icon.Information
                    )
                    
                except Exception as e:
                    # Mark as failed
                    self.error_tracker.mark_failure(e)
                    
                    # Show error message
                    ErrorHandler.handle_exception(e)
            
            # Start thread
            thread = threading.Thread(target=export_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            ErrorHandler.handle_exception(e)
    
    def export_selection(self):
        """Export the current selection as audio and/or transcript."""
        try:
            # Check if we have a selection
            if (self.parent.waveform_view.selection_start is None or 
                self.parent.waveform_view.selection_end is None):
                raise ExportError(
                    "No selection to export",
                    ErrorSeverity.WARNING,
                    {"method": "export_selection"}
                )
            
            # Get selection bounds
            start = min(self.parent.waveform_view.selection_start, self.parent.waveform_view.selection_end)
            end = max(self.parent.waveform_view.selection_start, self.parent.waveform_view.selection_end)
            
            # Create dialog to ask what to export
            dialog = QDialog(self.parent)
            dialog.setWindowTitle("Export Selection")
            layout = QVBoxLayout(dialog)
            
            layout.addWidget(QLabel(f"Export selection from {start:.2f}s to {end:.2f}s:"))
            
            # Checkboxes for what to export
            audio_checkbox = QCheckBox("Export audio")
            audio_checkbox.setChecked(True)
            layout.addWidget(audio_checkbox)
            
            transcript_checkbox = QCheckBox("Export transcript")
            transcript_checkbox.setChecked(True)
            layout.addWidget(transcript_checkbox)
            
            # Buttons
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            
            # Get options
            export_audio = audio_checkbox.isChecked()
            export_transcript = transcript_checkbox.isChecked()
            
            if not export_audio and not export_transcript:
                return
            
            # Get directory for export
            directory = QFileDialog.getExistingDirectory(
                self.parent,
                "Select Export Directory",
                str(Path.home())
            )
            
            if not directory:
                return
            
            # Generate base filename
            base_name = f"selection_{start:.2f}s-{end:.2f}s"
            
            # Start tracking export operation
            self.error_tracker.start_tracking(
                ExportOperation.SELECTION,
                target_path=directory
            )
            
            def export_thread():
                try:
                    results = []
                    
                    # Export audio if requested
                    if export_audio:
                        # Get format
                        format_key = "wav"  # Default format
                        file_path = os.path.join(directory, f"{base_name}.{format_key}")
                        
                        # Export audio segment
                        self.export_manager.export_segment(
                            self.parent.audio_processor.audio,
                            self.parent.audio_processor.sample_rate,
                            start,
                            end,
                            file_path,
                            format_key
                        )
                        
                        results.append(f"Audio: {file_path}")
                    
                    # Export transcript if requested
                    if export_transcript:
                        # Filter segments that overlap with selection
                        selected_segments = []
                        
                        for segment in self.parent.segments:
                            seg_start = segment.get("start", 0)
                            seg_end = segment.get("end", 0)
                            
                            # Check for overlap
                            if (seg_start <= end and seg_end >= start):
                                # Adjust segment boundaries to selection if needed
                                selected_segment = segment.copy()
                                
                                if seg_start < start:
                                    selected_segment["start"] = start
                                
                                if seg_end > end:
                                    selected_segment["end"] = end
                                
                                selected_segments.append(selected_segment)
                        
                        if selected_segments:
                            # Get format
                            format_key = "txt"  # Default format
                            file_path = os.path.join(directory, f"{base_name}.{format_key}")
                            
                            # Export transcript segments
                            self.export_manager.export_transcript(
                                selected_segments,
                                file_path,
                                format_key
                            )
                            
                            results.append(f"Transcript: {file_path}")
                    
                    # Mark as success
                    self.error_tracker.mark_success()
                    
                    # Show success message
                    self.parent.show_message(
                        "Export Successful",
                        "Exported:\n" + "\n".join(results),
                        QMessageBox.Icon.Information
                    )
                    
                except Exception as e:
                    # Mark as failed
                    self.error_tracker.mark_failure(e)
                    
                    # Show error message
                    ErrorHandler.handle_exception(e)
            
            # Start thread
            thread = threading.Thread(target=export_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            ErrorHandler.handle_exception(e)
    
    def _get_subtitle_options(self) -> Dict[str, Any]:
        """Get options for subtitle export formats.
        
        Returns:
            Dict[str, Any]: Options for subtitle export
        """
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Subtitle Export Options")
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("Subtitle Options:"))
        
        include_speaker_checkbox = QCheckBox("Include speaker information")
        include_speaker_checkbox.setChecked(True)
        layout.addWidget(include_speaker_checkbox)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        
        return {
            "include_speaker": include_speaker_checkbox.isChecked()
        }
    
    def _get_structured_options(self) -> Dict[str, Any]:
        """Get options for structured format export.
        
        Returns:
            Dict[str, Any]: Options for structured format export
        """
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Export Options")
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("Include fields:"))
        
        include_speaker_checkbox = QCheckBox("Speaker information")
        include_speaker_checkbox.setChecked(True)
        layout.addWidget(include_speaker_checkbox)
        
        include_timestamps_checkbox = QCheckBox("Timestamps")
        include_timestamps_checkbox.setChecked(True)
        layout.addWidget(include_timestamps_checkbox)
        
        include_words_checkbox = QCheckBox("Word-level details")
        include_words_checkbox.setChecked(False)
        layout.addWidget(include_words_checkbox)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        
        return {
            "include_speaker": include_speaker_checkbox.isChecked(),
            "include_timestamps": include_timestamps_checkbox.isChecked(),
            "include_words": include_words_checkbox.isChecked()
        }
    
    @pyqtSlot(ExportOperation, str, str)
    def _on_export_failed(self, operation: ExportOperation, error_message: str, target_path: str):
        """Handle export failure signal.
        
        Args:
            operation: Export operation type
            error_message: Error message
            target_path: Target path for the export
        """
        # Log the failure
        logger.error(f"Export failed: {operation.value} to {target_path}: {error_message}")
        
        # Check if we have multiple failures
        failures = self.error_tracker.get_recent_failures()
        
        if len(failures) >= 3:
            # Show error dialog with retry options
            self.error_dialog.set_failures(failures)
            self.error_dialog.exec()
    
    @pyqtSlot(list)
    def _on_retry_exports(self, failures: List[ExportAttempt]):
        """Handle retry exports request.
        
        Args:
            failures: List of failed export attempts to retry
        """
        logger.info(f"Retrying {len(failures)} failed exports")
        
        # Group failures by operation type
        by_operation = {}
        for failure in failures:
            if failure.operation not in by_operation:
                by_operation[failure.operation] = []
            by_operation[failure.operation].append(failure)
        
        # Process each operation type
        for operation, op_failures in by_operation.items():
            if operation == ExportOperation.TRANSCRIPT:
                # Find the newest transcript failure and retry it
                newest = max(op_failures, key=lambda f: f.start_time)
                
                # Reset the current attempt
                self.error_tracker.current_attempt = newest
                
                # Call export transcript with the same path
                def retry_func():
                    self.export_transcript()
                
                self.error_tracker.attempt_retry(retry_func)
                
            elif operation == ExportOperation.AUDIO_SEGMENT:
                # Find the newest audio segment failure and retry it
                newest = max(op_failures, key=lambda f: f.start_time)
                
                # Reset the current attempt
                self.error_tracker.current_attempt = newest
                
                # Call export audio segment with the same path
                def retry_func():
                    self.export_audio_segment()
                
                self.error_tracker.attempt_retry(retry_func)
                
            elif operation == ExportOperation.SPEAKER_AUDIO:
                # Find the newest speaker audio failure and retry it
                newest = max(op_failures, key=lambda f: f.start_time)
                
                # Reset the current attempt
                self.error_tracker.current_attempt = newest
                
                # Call export speaker audio with the same path
                def retry_func():
                    self.export_speaker_audio()
                
                self.error_tracker.attempt_retry(retry_func)
                
            elif operation == ExportOperation.SELECTION:
                # Find the newest selection failure and retry it
                newest = max(op_failures, key=lambda f: f.start_time)
                
                # Reset the current attempt
                self.error_tracker.current_attempt = newest
                
                # Call export selection with the same path
                def retry_func():
                    self.export_selection()
                
                self.error_tracker.attempt_retry(retry_func) 