"""
Export handlers for the VSAT UI.

This module provides handlers for exporting transcripts, audio segments,
and other data.
"""

import os
import logging
import threading
from typing import Dict, Any, Optional, Callable

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QCheckBox, QDialogButtonBox,
    QMessageBox, QFileDialog, QInputDialog, QLabel
)
from PyQt6.QtCore import pyqtSlot, QObject

from src.export.export_manager import ExportManager
from src.utils.error_handler import ErrorHandler, ExportError, ErrorSeverity
from src.ui.export_error_tracker import (
    ExportErrorTracker, ExportOperation, ExportAttempt
)
from src.ui.export_error_dialog import ExportErrorDialog

logger = logging.getLogger(__name__)


class ExportHandlers(QObject):
    """Handles all export functionality for the VSAT UI."""

    def __init__(self, parent: QObject) -> None:
        """Initialize the export handlers.

        Args:
            parent: Parent window (MainWindow)
        """
        super().__init__(parent)
        self.parent = parent
        self.export_manager = ExportManager()
        self.error_tracker = ExportErrorTracker()

        # Connect error tracker signals
        self.error_tracker.exportFailed.connect(self._on_export_failed)

        # Create error dialog
        self.error_dialog = ExportErrorDialog(parent)
        self.error_dialog.retryAllRequested.connect(self._on_retry_exports)

    def export_transcript(self) -> None:
        """Export the current transcript as a text file."""
        try:
            # Get export path
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent,
                "Export Transcript",
                os.path.expanduser("~/Documents"),
                "Text Files (*.txt);;All Files (*.*)"
            )

            if not file_path:
                return

            # Register attempt with error tracker
            self.error_tracker.register_attempt(
                ExportOperation.TRANSCRIPT,
                target_path=file_path
            )

            # Get transcript text
            transcript = self.parent.transcript_view.toPlainText()

            # Export in a separate thread
            def export_thread() -> None:
                try:
                    # Export transcript
                    self.export_manager.export_transcript(
                        transcript_text=transcript,
                        output_path=file_path
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
                    logger.error(f"Export failed: {str(e)}", exc_info=True)
                    self.error_tracker.mark_failure(str(e))

            # Start export thread
            threading.Thread(target=export_thread).start()

        except Exception as e:
            logger.error(f"Error in export_transcript: {str(e)}", exc_info=True)
            ErrorHandler.show_error(
                f"Error exporting transcript: {str(e)}",
                ErrorSeverity.ERROR,
                self.parent
            )

    def export_transcript_time_segments(self) -> None:
        """Export the transcript with time segments as a text file."""
        try:
            # Get export path
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent,
                "Export Transcript with Time Segments",
                os.path.expanduser("~/Documents"),
                "Text Files (*.txt);;All Files (*.*)"
            )

            if not file_path:
                return

            # Register attempt with error tracker
            self.error_tracker.register_attempt(
                ExportOperation.TRANSCRIPT_TIME_SEGMENTS,
                target_path=file_path
            )

            # Get segments
            segments = self.parent.segments

            # Export in a separate thread
            def export_thread() -> None:
                try:
                    # Export transcript with time segments
                    self.export_manager.export_transcript_time_segments(
                        segments=segments,
                        output_path=file_path
                    )

                    # Mark as success
                    self.error_tracker.mark_success()

                    # Show success message
                    self.parent.show_message(
                        "Export Successful",
                        f"Transcript with time segments exported to {file_path}",
                        QMessageBox.Icon.Information
                    )
                except Exception as e:
                    logger.error(f"Export failed: {str(e)}", exc_info=True)
                    self.error_tracker.mark_failure(str(e))

            # Start export thread
            threading.Thread(target=export_thread).start()

        except Exception as e:
            logger.error(
                f"Error in export_transcript_time_segments: {str(e)}",
                exc_info=True
            )
            ErrorHandler.show_error(
                f"Error exporting transcript: {str(e)}",
                ErrorSeverity.ERROR,
                self.parent
            )

    def export_speaker_audio(self) -> None:
        """Export audio segments for a specific speaker."""
        try:
            # Get list of speakers
            speaker_set = {
                s.get("speaker", "Unknown") for s in self.parent.segments
                if s.get("speaker") not in ["", None]
            }
            if not speaker_set:
                raise ExportError(
                    "No speaker segments found",
                    ErrorSeverity.WARNING,
                    {"method": "export_speaker_audio"}
                )

            # Convert to sorted list
            speaker_list = sorted(list(speaker_set))

            # Ask user to select a speaker
            speaker_idx, ok = QInputDialog.getItem(
                self.parent,
                "Select Speaker",
                "Export audio for speaker:",
                [str(s) for s in speaker_list],
                0,
                False
            )

            if not ok:
                return

            # Get the selected speaker
            selected_speaker = speaker_list[
                speaker_list.index(
                    int(speaker_idx) if speaker_idx.isdigit() else speaker_idx
                )
            ]

            # Get export format
            formats = ["WAV", "MP3"]
            format_idx, ok = QInputDialog.getItem(
                self.parent,
                "Select Format",
                "Export format:",
                formats,
                0,
                False
            )

            if not ok:
                return

            # Get export directory
            directory = QFileDialog.getExistingDirectory(
                self.parent,
                "Select Export Directory",
                os.path.expanduser("~/Documents")
            )

            if not directory:
                return

            # Register attempt with error tracker
            self.error_tracker.register_attempt(
                ExportOperation.SPEAKER_AUDIO,
                target_path=directory
            )

            # Filter segments for selected speaker
            speaker_segments = [
                s for s in self.parent.segments
                if s.get("speaker") == selected_speaker
            ]

            # Export in a separate thread
            def export_thread() -> None:
                try:
                    # Export speaker audio segments
                    exported_files = []
                    export_format = formats[formats.index(format_idx)].lower()

                    for i, segment in enumerate(speaker_segments):
                        # Export segment audio
                        file_path = self.export_manager.export_segment_audio(
                            audio_path=self.parent.audio_path,
                            start_time=segment["start"],
                            end_time=segment["end"],
                            output_dir=directory,
                            filename=f"speaker_{selected_speaker}_{i+1:03d}",
                            export_format=export_format
                        )
                        exported_files.append(file_path)

                    # Mark as success
                    self.error_tracker.mark_success()

                    # Show success message
                    self.parent.show_message(
                        "Export Successful",
                        f"Exported {len(exported_files)} audio segments for "
                        f"speaker {selected_speaker}",
                        QMessageBox.Icon.Information
                    )
                except Exception as e:
                    logger.error(f"Export failed: {str(e)}", exc_info=True)
                    self.error_tracker.mark_failure(str(e))

            # Start export thread
            threading.Thread(target=export_thread).start()

        except Exception as e:
            logger.error(f"Error in export_speaker_audio: {str(e)}", exc_info=True)
            ErrorHandler.show_error(
                f"Error exporting speaker audio: {str(e)}",
                ErrorSeverity.ERROR,
                self.parent
            )

    def export_selection(self) -> None:
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
            start = self.parent.waveform_view.selection_start
            end = self.parent.waveform_view.selection_end

            # Create dialog to select export options
            dialog = QDialog(self.parent)
            dialog.setWindowTitle("Export Selection")
            layout = QVBoxLayout(dialog)

            layout.addWidget(
                QLabel(f"Export selection from {start:.2f}s to {end:.2f}s:")
            )

            # Checkboxes for what to export
            audio_checkbox = QCheckBox("Export audio")
            audio_checkbox.setChecked(True)
            layout.addWidget(audio_checkbox)

            transcript_checkbox = QCheckBox("Export transcript for this selection")
            transcript_checkbox.setChecked(True)
            layout.addWidget(transcript_checkbox)

            # Get available formats
            formats = ["WAV", "MP3"]
            format_idx = 0  # Default to WAV

            # Add dialog buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok |
                QDialogButtonBox.StandardButton.Cancel
            )
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            # Show dialog
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            # Get export options
            export_audio = audio_checkbox.isChecked()
            export_transcript = transcript_checkbox.isChecked()

            # At least one option must be selected
            if not export_audio and not export_transcript:
                self.parent.show_message(
                    "Export Error",
                    "Please select at least one export option",
                    QMessageBox.Icon.Warning
                )
                return

            # Get export directory
            directory = QFileDialog.getExistingDirectory(
                self.parent,
                "Select Export Directory",
                os.path.expanduser("~/Documents")
            )

            if not directory:
                return

            # Register attempt with error tracker
            self.error_tracker.register_attempt(
                ExportOperation.SELECTION,
                target_path=directory
            )

            # Export in a separate thread
            def export_thread() -> None:
                try:
                    export_format = formats[format_idx].lower()
                    timestamp = self.export_manager.get_timestamp()
                    exported_files = []

                    # Export audio if selected
                    if export_audio:
                        filename = f"selection_{start:.2f}_{end:.2f}_{timestamp}"
                        audio_path = self.export_manager.export_segment_audio(
                            audio_path=self.parent.audio_path,
                            start_time=start,
                            end_time=end,
                            output_dir=directory,
                            filename=filename,
                            export_format=export_format
                        )
                        exported_files.append(audio_path)

                    # Export transcript if selected
                    if export_transcript:
                        # Find segments in the selection
                        selected_segments = [
                            s for s in self.parent.segments
                            if s.get("start") >= start and s.get("end") <= end
                        ]

                        # Export transcript segments
                        if selected_segments:
                            transcript_path = os.path.join(
                                directory,
                                f"selection_{start:.2f}_{end:.2f}_{timestamp}.txt"
                            )
                            self.export_manager.export_transcript_time_segments(
                                segments=selected_segments,
                                output_path=transcript_path
                            )
                            exported_files.append(transcript_path)
                        else:
                            logger.warning(
                                "No transcript segments found in the selection"
                            )

                    # Mark as success
                    self.error_tracker.mark_success()

                    # Show success message
                    self.parent.show_message(
                        "Export Successful",
                        f"Exported {len(exported_files)} files to {directory}",
                        QMessageBox.Icon.Information
                    )
                except Exception as e:
                    logger.error(f"Export failed: {str(e)}", exc_info=True)
                    self.error_tracker.mark_failure(str(e))

            # Start export thread
            threading.Thread(target=export_thread).start()

        except Exception as e:
            logger.error(f"Error in export_selection: {str(e)}", exc_info=True)
            ErrorHandler.show_error(
                f"Error exporting selection: {str(e)}",
                ErrorSeverity.ERROR,
                self.parent
            )

    @pyqtSlot(ExportAttempt)
    def _on_export_failed(self, attempt: ExportAttempt) -> None:
        """Handle export failure.

        Args:
            attempt: The failed export attempt
        """
        # Add the failure to the error dialog
        self.error_dialog.add_failure(attempt)

        # Show the error dialog if not already visible
        if not self.error_dialog.isVisible():
            self.error_dialog.exec()

    @pyqtSlot()
    def _on_retry_exports(self) -> None:
        """Retry all failed exports."""
        # Get failed attempts
        failed_attempts = self.error_tracker.get_failed_attempts()

        # Retry each attempt based on operation type
        for attempt in failed_attempts:
            self._retry_attempt(attempt)

    def _retry_attempt(self, attempt: ExportAttempt) -> None:
        """Retry a specific export attempt.

        Args:
            attempt: The export attempt to retry
        """
        # Map operations to retry methods
        retry_methods = {
            ExportOperation.TRANSCRIPT: self._retry_transcript_export,
            ExportOperation.TRANSCRIPT_TIME_SEGMENTS: self._retry_transcript_time_export,
            ExportOperation.SPEAKER_AUDIO: self._retry_speaker_audio_export,
            ExportOperation.SELECTION: self._retry_selection_export
        }

        # Call appropriate retry method
        if attempt.operation in retry_methods:
            retry_methods[attempt.operation](attempt)
        else:
            logger.error(f"Unknown operation type: {attempt.operation}")

    def _get_parent(self) -> QObject:
        """Get the parent window.

        Returns:
            The parent window
        """
        return self.parent

    def _show_export_dialog(self, parent: Optional[QObject] = None) -> Dict[str, Any]:
        """Show a dialog for export options.

        Args:
            parent: Parent widget

        Returns:
            Dictionary of export options
        """
        # Create dialog
        dialog = QDialog(parent if parent is not None else self._get_parent())
        dialog.setWindowTitle("Export Options")

        # Add options
        layout = QVBoxLayout(dialog)

        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Show dialog
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return {}

        return {}

    def _show_export_format_dialog(
            self, parent: Optional[QObject] = None
    ) -> Dict[str, Any]:
        """Show a dialog for selecting export format.

        Args:
            parent: Parent widget

        Returns:
            Dictionary of export format options
        """
        # Create dialog
        dialog = QDialog(parent if parent is not None else self._get_parent())
        dialog.setWindowTitle("Export Format")

        # Add options
        layout = QVBoxLayout(dialog)

        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Show dialog
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return {}

        return {}

    def export_audio(self) -> None:
        """Export the entire audio file."""
        try:
            # Get export path
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent,
                "Export Audio",
                os.path.expanduser("~/Documents"),
                "Audio Files (*.wav *.mp3);;All Files (*.*)"
            )

            if not file_path:
                return

            # Get file extension
            _, ext = os.path.splitext(file_path)
            if not ext:
                file_path += ".wav"
                ext = ".wav"

            # Map extension to format
            format_map = {
                ".wav": "wav",
                ".mp3": "mp3"
            }
            export_format = format_map.get(ext.lower(), "wav")

            # Register attempt with error tracker
            self.error_tracker.register_attempt(
                ExportOperation.AUDIO,
                target_path=file_path
            )

            # Export in a separate thread
            def export_thread() -> None:
                try:
                    # Export audio
                    self.export_manager.export_audio(
                        audio_path=self.parent.audio_path,
                        output_path=file_path,
                        export_format=export_format
                    )

                    # Mark as success
                    self.error_tracker.mark_success()

                    # Show success message
                    self.parent.show_message(
                        "Export Successful",
                        f"Audio exported to {file_path}",
                        QMessageBox.Icon.Information
                    )
                except Exception as e:
                    logger.error(f"Export failed: {str(e)}", exc_info=True)
                    self.error_tracker.mark_failure(str(e))

            # Start export thread
            threading.Thread(target=export_thread).start()

        except Exception as e:
            logger.error(f"Error in export_audio: {str(e)}", exc_info=True)
            ErrorHandler.show_error(
                f"Error exporting audio: {str(e)}",
                ErrorSeverity.ERROR,
                self.parent
            )

    def _retry_transcript_export(self, attempt: ExportAttempt) -> None:
        """Retry a failed transcript export.

        Args:
            attempt: The failed export attempt
        """
        def retry_func() -> None:
            self.export_transcript()

        self._schedule_retry(retry_func)

    def _retry_transcript_time_export(self, attempt: ExportAttempt) -> None:
        """Retry a failed transcript with time segments export.

        Args:
            attempt: The failed export attempt
        """
        def retry_func() -> None:
            self.export_transcript_time_segments()

        self._schedule_retry(retry_func)

    def _retry_speaker_audio_export(self, attempt: ExportAttempt) -> None:
        """Retry a failed speaker audio export.

        Args:
            attempt: The failed export attempt
        """
        def retry_func() -> None:
            self.export_speaker_audio()

        self._schedule_retry(retry_func)

    def _retry_selection_export(self, attempt: ExportAttempt) -> None:
        """Retry a failed selection export.

        Args:
            attempt: The failed export attempt
        """
        def retry_func() -> None:
            self.export_selection()

        self._schedule_retry(retry_func)

    def _schedule_retry(self, retry_func: Callable[[], None]) -> None:
        """Schedule a retry function to be executed.

        Args:
            retry_func: The function to retry the export
        """
        # Execute the retry function
        retry_func()
