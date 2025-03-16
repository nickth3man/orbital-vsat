#!/usr/bin/env python3
"""
Batch processing dialog for VSAT.

This module provides a dialog for batch processing multiple audio files
with progress tracking and result reporting.
"""

import os
import logging
import threading
from typing import List, Dict, Any, Optional, Callable

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QListWidget, QListWidgetItem, QCheckBox,
    QTabWidget, QTextEdit, QFileDialog, QMessageBox,
    QGroupBox, QSpinBox, QComboBox, QFormLayout, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QFont, QColor

from src.audio.batch_processor import BatchProcessor
from src.database.data_manager import DataManager
from src.utils.performance_optimizer import get_performance_monitor

logger = logging.getLogger(__name__)

class BatchProcessingDialog(QDialog):
    """Dialog for batch processing multiple audio files."""
    
    # Signals
    progress_updated = pyqtSignal(dict)
    processing_completed = pyqtSignal(dict)
    
    def __init__(
        self,
        parent=None,
        data_manager: Optional[DataManager] = None,
        initial_files: Optional[List[str]] = None
    ):
        """Initialize the batch processing dialog.
        
        Args:
            parent: Parent widget
            data_manager: Data manager for storing results
            initial_files: Initial list of files to process
        """
        super().__init__(parent)
        self.data_manager = data_manager
        self.initial_files = initial_files or []
        
        # Batch processor
        self.batch_processor = None
        self.processing_thread = None
        
        # UI state
        self.is_processing = False
        self.files_to_process = []
        
        # Set up UI
        self._setup_ui()
        
        # Connect signals
        self.progress_updated.connect(self._update_progress)
        self.processing_completed.connect(self._handle_completion)
        
        # Initialize batch processor
        self._initialize_batch_processor()
        
        # Add initial files if provided
        if self.initial_files:
            self._add_files(self.initial_files)
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        # Dialog properties
        self.setWindowTitle("Batch Processing")
        self.setMinimumSize(800, 600)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)
        
        # Top section: File list and controls
        top_widget = QGroupBox("Files to Process")
        top_layout = QVBoxLayout(top_widget)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        top_layout.addWidget(self.file_list)
        
        # File controls
        file_controls = QHBoxLayout()
        self.add_files_button = QPushButton("Add Files")
        self.add_folder_button = QPushButton("Add Folder")
        self.remove_files_button = QPushButton("Remove Selected")
        self.clear_files_button = QPushButton("Clear All")
        
        file_controls.addWidget(self.add_files_button)
        file_controls.addWidget(self.add_folder_button)
        file_controls.addWidget(self.remove_files_button)
        file_controls.addWidget(self.clear_files_button)
        top_layout.addLayout(file_controls)
        
        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QFormLayout(options_group)
        
        self.diarization_checkbox = QCheckBox("Perform Speaker Diarization")
        self.diarization_checkbox.setChecked(True)
        
        self.transcription_checkbox = QCheckBox("Perform Transcription")
        self.transcription_checkbox.setChecked(True)
        
        self.speaker_id_checkbox = QCheckBox("Perform Speaker Identification")
        self.speaker_id_checkbox.setChecked(True)
        
        self.store_results_checkbox = QCheckBox("Store Results in Database")
        self.store_results_checkbox.setChecked(True)
        
        self.concurrent_tasks_spinner = QSpinBox()
        self.concurrent_tasks_spinner.setMinimum(1)
        self.concurrent_tasks_spinner.setMaximum(16)
        self.concurrent_tasks_spinner.setValue(4)
        
        options_layout.addRow("", self.diarization_checkbox)
        options_layout.addRow("", self.transcription_checkbox)
        options_layout.addRow("", self.speaker_id_checkbox)
        options_layout.addRow("", self.store_results_checkbox)
        options_layout.addRow("Concurrent Tasks:", self.concurrent_tasks_spinner)
        
        top_layout.addWidget(options_group)
        
        # Add top widget to splitter
        splitter.addWidget(top_widget)
        
        # Middle section: Progress
        progress_widget = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_widget)
        
        # Overall progress
        progress_layout.addWidget(QLabel("Overall Progress:"))
        self.overall_progress = QProgressBar()
        progress_layout.addWidget(self.overall_progress)
        
        # Status labels
        status_layout = QHBoxLayout()
        self.total_label = QLabel("Total: 0")
        self.completed_label = QLabel("Completed: 0")
        self.failed_label = QLabel("Failed: 0")
        self.pending_label = QLabel("Pending: 0")
        
        status_layout.addWidget(self.total_label)
        status_layout.addWidget(self.completed_label)
        status_layout.addWidget(self.failed_label)
        status_layout.addWidget(self.pending_label)
        progress_layout.addLayout(status_layout)
        
        # Current file
        progress_layout.addWidget(QLabel("Current File:"))
        self.current_file_label = QLabel("None")
        progress_layout.addWidget(self.current_file_label)
        
        # Add progress widget to splitter
        splitter.addWidget(progress_widget)
        
        # Bottom section: Results tabs
        results_tabs = QTabWidget()
        
        # Results tab
        self.results_list = QListWidget()
        results_tabs.addTab(self.results_list, "Results")
        
        # Failed files tab
        self.failed_list = QListWidget()
        results_tabs.addTab(self.failed_list, "Failed Files")
        
        # Performance tab
        self.performance_text = QTextEdit()
        self.performance_text.setReadOnly(True)
        self.performance_text.setFont(QFont("Courier New", 9))
        results_tabs.addTab(self.performance_text, "Performance")
        
        # Report tab
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Courier New", 9))
        results_tabs.addTab(self.report_text, "Report")
        
        # Add results tabs to splitter
        splitter.addWidget(results_tabs)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 150, 300])
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Processing")
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.setEnabled(False)
        self.close_button = QPushButton("Close")
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        main_layout.addLayout(button_layout)
        
        # Connect button signals
        self.add_files_button.clicked.connect(self._on_add_files)
        self.add_folder_button.clicked.connect(self._on_add_folder)
        self.remove_files_button.clicked.connect(self._on_remove_files)
        self.clear_files_button.clicked.connect(self._on_clear_files)
        self.start_button.clicked.connect(self._on_start_processing)
        self.stop_button.clicked.connect(self._on_stop_processing)
        self.close_button.clicked.connect(self.close)
        
        # Update UI state
        self._update_ui_state()
    
    def _initialize_batch_processor(self):
        """Initialize the batch processor."""
        max_concurrent_tasks = self.concurrent_tasks_spinner.value()
        
        self.batch_processor = BatchProcessor(
            data_manager=self.data_manager if self.store_results_checkbox.isChecked() else None,
            max_concurrent_tasks=max_concurrent_tasks,
            enable_performance_monitoring=True
        )
        
        # Set progress callback
        self.batch_processor.set_progress_callback(self._on_progress_update)
    
    def _add_files(self, file_paths: List[str]):
        """Add files to the list.
        
        Args:
            file_paths: List of file paths to add
        """
        for file_path in file_paths:
            # Check if file already exists in the list
            existing_items = self.file_list.findItems(
                file_path, Qt.MatchFlag.MatchExactly
            )
            if not existing_items:
                # Add to list widget
                item = QListWidgetItem(file_path)
                self.file_list.addItem(item)
                
                # Add to files to process
                self.files_to_process.append(file_path)
        
        # Update UI state
        self._update_ui_state()
    
    def _update_ui_state(self):
        """Update UI state based on current processing state."""
        has_files = len(self.files_to_process) > 0
        
        # Update button states
        self.start_button.setEnabled(has_files and not self.is_processing)
        self.stop_button.setEnabled(self.is_processing)
        self.add_files_button.setEnabled(not self.is_processing)
        self.add_folder_button.setEnabled(not self.is_processing)
        self.remove_files_button.setEnabled(has_files and not self.is_processing)
        self.clear_files_button.setEnabled(has_files and not self.is_processing)
        
        # Update option controls
        self.diarization_checkbox.setEnabled(not self.is_processing)
        self.transcription_checkbox.setEnabled(not self.is_processing)
        self.speaker_id_checkbox.setEnabled(not self.is_processing)
        self.store_results_checkbox.setEnabled(not self.is_processing)
        self.concurrent_tasks_spinner.setEnabled(not self.is_processing)
        
        # Update file count
        self.total_label.setText(f"Total: {len(self.files_to_process)}")
    
    def _on_add_files(self):
        """Handle add files button click."""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)")
        
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            self._add_files(file_paths)
    
    def _on_add_folder(self):
        """Handle add folder button click."""
        folder_dialog = QFileDialog(self)
        folder_dialog.setFileMode(QFileDialog.FileMode.Directory)
        folder_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        
        if folder_dialog.exec():
            folder_path = folder_dialog.selectedFiles()[0]
            
            # Find all audio files in the folder
            audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
            file_paths = []
            
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        file_paths.append(os.path.join(root, file))
            
            if file_paths:
                self._add_files(file_paths)
            else:
                QMessageBox.information(
                    self, "No Audio Files", 
                    f"No audio files found in {folder_path}"
                )
    
    def _on_remove_files(self):
        """Handle remove files button click."""
        selected_items = self.file_list.selectedItems()
        
        for item in selected_items:
            file_path = item.text()
            
            # Remove from list widget
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            
            # Remove from files to process
            if file_path in self.files_to_process:
                self.files_to_process.remove(file_path)
        
        # Update UI state
        self._update_ui_state()
    
    def _on_clear_files(self):
        """Handle clear files button click."""
        self.file_list.clear()
        self.files_to_process = []
        
        # Update UI state
        self._update_ui_state()
    
    def _on_start_processing(self):
        """Handle start processing button click."""
        if not self.files_to_process:
            return
        
        # Update UI state
        self.is_processing = True
        self._update_ui_state()
        
        # Clear results
        self.results_list.clear()
        self.failed_list.clear()
        self.performance_text.clear()
        self.report_text.clear()
        
        # Reset progress
        self.overall_progress.setValue(0)
        self.completed_label.setText("Completed: 0")
        self.failed_label.setText("Failed: 0")
        self.pending_label.setText(f"Pending: {len(self.files_to_process)}")
        self.current_file_label.setText("Starting...")
        
        # Re-initialize batch processor with current settings
        self._initialize_batch_processor()
        
        # Add files to batch processor
        options = {
            "perform_diarization": self.diarization_checkbox.isChecked(),
            "perform_transcription": self.transcription_checkbox.isChecked(),
            "perform_speaker_identification": self.speaker_id_checkbox.isChecked()
        }
        
        self.batch_processor.add_files(self.files_to_process, options)
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(
            target=self._process_files_thread,
            daemon=True
        )
        self.processing_thread.start()
    
    def _process_files_thread(self):
        """Process files in a separate thread."""
        try:
            # Process all files
            result = self.batch_processor.process_all(optimize_queue=True)
            
            # Emit completion signal
            self.processing_completed.emit(result)
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            
            # Emit completion signal with error
            self.processing_completed.emit({
                "error": str(e),
                "total_files": len(self.files_to_process),
                "completed_files": 0,
                "failed_files": len(self.files_to_process)
            })
    
    def _on_stop_processing(self):
        """Handle stop processing button click."""
        if self.batch_processor and self.is_processing:
            # Request stop
            self.batch_processor.stop()
            self.current_file_label.setText("Stopping...")
    
    def _on_progress_update(self, progress: Dict[str, Any]):
        """Handle progress update from batch processor.
        
        Args:
            progress: Progress information
        """
        # Emit signal to update UI in the main thread
        self.progress_updated.emit(progress)
    
    def _update_progress(self, progress: Dict[str, Any]):
        """Update progress UI with the latest information.
        
        Args:
            progress: Progress information
        """
        # Update progress bar
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        
        if total > 0:
            percent = int((completed + failed) / total * 100)
            self.overall_progress.setValue(percent)
        
        # Update status labels
        self.total_label.setText(f"Total: {total}")
        self.completed_label.setText(f"Completed: {completed}")
        self.failed_label.setText(f"Failed: {failed}")
        self.pending_label.setText(f"Pending: {progress.get('pending', 0)}")
        
        # Update current file
        current_task = progress.get("current_task")
        if current_task:
            self.current_file_label.setText(str(current_task))
        
        # Update performance tab if available
        if "performance" in progress:
            performance = progress["performance"]
            memory = performance.get("memory", {})
            cpu = performance.get("cpu", {})
            
            # Create performance text
            perf_text = []
            perf_text.append("Memory Usage:")
            perf_text.append(f"  Current: {memory.get('current_mb', 0):.2f} MB")
            perf_text.append(f"  Maximum: {memory.get('max_mb', 0):.2f} MB")
            perf_text.append(f"  Increase: {memory.get('increase_mb', 0):.2f} MB")
            perf_text.append("")
            perf_text.append("CPU Usage:")
            perf_text.append(f"  Current: {cpu.get('current_percent', 0):.2f}%")
            
            self.performance_text.setText("\n".join(perf_text))
    
    def _handle_completion(self, result: Dict[str, Any]):
        """Handle completion of batch processing.
        
        Args:
            result: Processing result
        """
        # Update UI state
        self.is_processing = False
        self._update_ui_state()
        
        # Check for error
        if "error" in result:
            QMessageBox.critical(
                self, "Batch Processing Error",
                f"Error during batch processing: {result['error']}"
            )
            return
        
        # Update results list
        for file_path, file_result in result.get("results", {}).items():
            item = QListWidgetItem(os.path.basename(file_path))
            item.setToolTip(file_path)
            
            # Set color based on success
            if file_result:
                item.setForeground(QColor("green"))
            else:
                item.setForeground(QColor("red"))
            
            self.results_list.addItem(item)
        
        # Update failed list
        status = self.batch_processor.get_task_status()
        for task in status.get("failed", []):
            item = QListWidgetItem(os.path.basename(task.file_path))
            item.setToolTip(f"{task.file_path}: {str(task.error)}")
            item.setForeground(QColor("red"))
            self.failed_list.addItem(item)
        
        # Update performance tab
        if "performance_report" in result:
            self.performance_text.setText(result["performance_report"])
        
        # Update report tab
        if self.batch_processor:
            self.report_text.setText(self.batch_processor.generate_report())
        
        # Show completion message
        total_files = result.get("total_files", 0)
        completed_files = result.get("completed_files", 0)
        failed_files = result.get("failed_files", 0)
        
        QMessageBox.information(
            self, "Batch Processing Complete",
            f"Processed {total_files} files:\n"
            f"- {completed_files} completed successfully\n"
            f"- {failed_files} failed"
        )
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.is_processing:
            # Ask for confirmation
            reply = QMessageBox.question(
                self, "Confirm Close",
                "Batch processing is still running. Stop processing and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Stop processing
                self._on_stop_processing()
                
                # Wait a bit for processing to stop
                for _ in range(10):
                    if not self.is_processing:
                        break
                    QTimer.singleShot(100, lambda: None)
                
                event.accept()
            else:
                event.ignore()
        else:
            event.accept() 