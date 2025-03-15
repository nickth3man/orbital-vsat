"""
Data Management Dialog for VSAT.

This module provides a dialog for managing data, including
statistics, backup/restore, archiving, and pruning operations.
"""

import os
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QFileDialog, QProgressBar, QTreeWidget, QTreeWidgetItem, QGroupBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QMessageBox, QListWidget, QComboBox,
    QDialogButtonBox, QScrollArea, QFrame, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread, pyqtSlot, QSize
from PyQt6.QtGui import QFont, QIcon

from src.database.db_manager import DatabaseManager
from src.database.data_manager import DataManager, DataManagerError
from src.database.models import Recording, Speaker
from src.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class BackgroundWorker(QObject):
    """Worker class for background operations."""
    
    finished = pyqtSignal(object)
    error = pyqtSignal(Exception)
    progress = pyqtSignal(int, str)
    
    def __init__(self, func, *args, **kwargs):
        """Initialize the worker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        """Run the function in a background thread."""
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(e)

class DataManagementDialog(QDialog):
    """Dialog for data management operations."""
    
    def __init__(self, db_manager: DatabaseManager, parent=None):
        """Initialize the dialog.
        
        Args:
            db_manager: Database manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.db_manager = db_manager
        self.data_manager = DataManager(db_manager)
        self.session = self.db_manager.get_session()
        
        self.setWindowTitle("Data Management")
        self.setMinimumSize(800, 600)
        
        self._init_ui()
        
        # Load initial data
        self.refresh_statistics()
        self.load_recordings()
        self.load_speakers()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Add tabs
        self.tab_statistics = self._create_statistics_tab()
        self.tab_backup = self._create_backup_tab()
        self.tab_archive = self._create_archive_tab()
        self.tab_pruning = self._create_pruning_tab()
        
        self.tab_widget.addTab(self.tab_statistics, "Statistics")
        self.tab_widget.addTab(self.tab_backup, "Backup & Restore")
        self.tab_widget.addTab(self.tab_archive, "Archiving")
        self.tab_widget.addTab(self.tab_pruning, "Data Pruning")
        
        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _create_statistics_tab(self) -> QWidget:
        """Create the statistics tab.
        
        Returns:
            QWidget: The tab widget
        """
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Add refresh button
        refresh_layout = QHBoxLayout()
        refresh_button = QPushButton("Refresh Statistics")
        refresh_button.clicked.connect(self.refresh_statistics)
        refresh_layout.addWidget(refresh_button)
        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)
        
        # Create a scroll area for the statistics
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create a container widget for the statistics
        self.stats_widget = QWidget()
        self.stats_layout = QVBoxLayout()
        self.stats_widget.setLayout(self.stats_layout)
        
        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(self.stats_widget)
        layout.addWidget(scroll_area)
        
        return tab
    
    def _create_backup_tab(self) -> QWidget:
        """Create the backup & restore tab.
        
        Returns:
            QWidget: The tab widget
        """
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Backup section
        backup_group = QGroupBox("Create Backup")
        backup_layout = QVBoxLayout()
        backup_group.setLayout(backup_layout)
        
        backup_desc = QLabel("Create a backup of the entire database. This backup can be restored later.")
        backup_desc.setWordWrap(True)
        backup_layout.addWidget(backup_desc)
        
        backup_button_layout = QHBoxLayout()
        self.backup_button = QPushButton("Create Backup...")
        self.backup_button.clicked.connect(self.create_backup)
        backup_button_layout.addWidget(self.backup_button)
        backup_button_layout.addStretch()
        backup_layout.addLayout(backup_button_layout)
        
        layout.addWidget(backup_group)
        
        # Restore section
        restore_group = QGroupBox("Restore from Backup")
        restore_layout = QVBoxLayout()
        restore_group.setLayout(restore_layout)
        
        restore_desc = QLabel(
            "Restore the database from a backup file. This will replace the current database. "
            "A backup of the current database will be created before restoring."
        )
        restore_desc.setWordWrap(True)
        restore_layout.addWidget(restore_desc)
        
        restore_warning = QLabel(
            "WARNING: This operation cannot be undone. Make sure you have a backup of your current data."
        )
        restore_warning.setWordWrap(True)
        restore_warning.setStyleSheet("color: red;")
        restore_layout.addWidget(restore_warning)
        
        restore_button_layout = QHBoxLayout()
        self.restore_button = QPushButton("Restore from Backup...")
        self.restore_button.clicked.connect(self.restore_backup)
        restore_button_layout.addWidget(self.restore_button)
        restore_button_layout.addStretch()
        restore_layout.addLayout(restore_button_layout)
        
        layout.addWidget(restore_group)
        layout.addStretch()
        
        return tab
    
    def _create_archive_tab(self) -> QWidget:
        """Create the archiving tab.
        
        Returns:
            QWidget: The tab widget
        """
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Recordings list section
        recordings_group = QGroupBox("Recordings")
        recordings_layout = QVBoxLayout()
        recordings_group.setLayout(recordings_layout)
        
        recordings_desc = QLabel("Select a recording to archive.")
        recordings_desc.setWordWrap(True)
        recordings_layout.addWidget(recordings_desc)
        
        self.recordings_list = QTreeWidget()
        self.recordings_list.setHeaderLabels(["ID", "Filename", "Duration", "Date", "Speakers"])
        self.recordings_list.setColumnWidth(0, 50)
        self.recordings_list.setColumnWidth(1, 200)
        self.recordings_list.setColumnWidth(2, 100)
        self.recordings_list.setColumnWidth(3, 150)
        self.recordings_list.setAlternatingRowColors(True)
        recordings_layout.addWidget(self.recordings_list)
        
        # Archive/Restore buttons
        button_layout = QHBoxLayout()
        
        self.archive_button = QPushButton("Archive Selected Recording...")
        self.archive_button.clicked.connect(self.archive_recording)
        button_layout.addWidget(self.archive_button)
        
        self.restore_archive_button = QPushButton("Restore Archive...")
        self.restore_archive_button.clicked.connect(self.restore_archive)
        button_layout.addWidget(self.restore_archive_button)
        
        refresh_recordings_button = QPushButton("Refresh")
        refresh_recordings_button.clicked.connect(self.load_recordings)
        button_layout.addWidget(refresh_recordings_button)
        
        button_layout.addStretch()
        recordings_layout.addLayout(button_layout)
        
        layout.addWidget(recordings_group)
        
        return tab
    
    def _create_pruning_tab(self) -> QWidget:
        """Create the data pruning tab.
        
        Returns:
            QWidget: The tab widget
        """
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Pruning description
        pruning_desc = QLabel(
            "Data pruning allows you to remove unused or old data from the database. "
            "Select the criteria for data to be removed."
        )
        pruning_desc.setWordWrap(True)
        layout.addWidget(pruning_desc)
        
        pruning_warning = QLabel(
            "WARNING: This operation cannot be undone. Make sure you have a backup of your data."
        )
        pruning_warning.setWordWrap(True)
        pruning_warning.setStyleSheet("color: red;")
        layout.addWidget(pruning_warning)
        
        # Pruning rules
        rules_group = QGroupBox("Pruning Rules")
        rules_layout = QVBoxLayout()
        rules_group.setLayout(rules_layout)
        
        # Age-based pruning
        age_layout = QHBoxLayout()
        self.age_checkbox = QCheckBox("Remove recordings older than")
        self.age_spinbox = QSpinBox()
        self.age_spinbox.setMinimum(1)
        self.age_spinbox.setMaximum(3650)  # 10 years
        self.age_spinbox.setValue(90)  # Default to 90 days
        age_layout.addWidget(self.age_checkbox)
        age_layout.addWidget(self.age_spinbox)
        age_layout.addWidget(QLabel("days"))
        age_layout.addStretch()
        rules_layout.addLayout(age_layout)
        
        # Unprocessed recordings
        self.unprocessed_checkbox = QCheckBox("Remove unprocessed recordings")
        rules_layout.addWidget(self.unprocessed_checkbox)
        
        # Duration-based pruning
        duration_layout = QHBoxLayout()
        self.min_duration_checkbox = QCheckBox("Remove recordings shorter than")
        self.min_duration_spinbox = QDoubleSpinBox()
        self.min_duration_spinbox.setMinimum(0.1)
        self.min_duration_spinbox.setMaximum(300.0)  # 5 minutes
        self.min_duration_spinbox.setValue(5.0)  # Default to 5 seconds
        duration_layout.addWidget(self.min_duration_checkbox)
        duration_layout.addWidget(self.min_duration_spinbox)
        duration_layout.addWidget(QLabel("seconds"))
        duration_layout.addStretch()
        rules_layout.addLayout(duration_layout)
        
        max_duration_layout = QHBoxLayout()
        self.max_duration_checkbox = QCheckBox("Remove recordings longer than")
        self.max_duration_spinbox = QDoubleSpinBox()
        self.max_duration_spinbox.setMinimum(60.0)  # 1 minute
        self.max_duration_spinbox.setMaximum(86400.0)  # 24 hours
        self.max_duration_spinbox.setValue(3600.0)  # Default to 1 hour
        max_duration_layout.addWidget(self.max_duration_checkbox)
        max_duration_layout.addWidget(self.max_duration_spinbox)
        max_duration_layout.addWidget(QLabel("seconds"))
        max_duration_layout.addStretch()
        rules_layout.addLayout(max_duration_layout)
        
        # Speaker-based pruning
        speaker_layout = QVBoxLayout()
        self.speaker_checkbox = QCheckBox("Remove selected speakers")
        self.speakers_list = QListWidget()
        self.speakers_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        speaker_layout.addWidget(self.speaker_checkbox)
        speaker_layout.addWidget(self.speakers_list)
        rules_layout.addLayout(speaker_layout)
        
        # Orphaned speakers
        self.orphaned_checkbox = QCheckBox("Remove speakers with no segments")
        rules_layout.addWidget(self.orphaned_checkbox)
        
        layout.addWidget(rules_group)
        
        # Pruning execution
        button_layout = QHBoxLayout()
        
        self.pruning_button = QPushButton("Execute Pruning")
        self.pruning_button.clicked.connect(self.execute_pruning)
        button_layout.addWidget(self.pruning_button)
        
        # Refresh speakers button
        refresh_speakers_button = QPushButton("Refresh Speakers")
        refresh_speakers_button.clicked.connect(self.load_speakers)
        button_layout.addWidget(refresh_speakers_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return tab
    
    def refresh_statistics(self):
        """Refresh database statistics."""
        try:
            # Clear existing statistics
            while self.stats_layout.count():
                item = self.stats_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            
            # Show loading indicator
            loading_label = QLabel("Loading statistics...")
            self.stats_layout.addWidget(loading_label)
            
            # Create background worker
            worker = BackgroundWorker(self.data_manager.get_database_statistics, self.session)
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(lambda stats: self._display_statistics(stats))
            worker.error.connect(lambda e: self._handle_error(e, "Error collecting statistics"))
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.start()
            
        except Exception as e:
            ErrorHandler.handle_exception(e, parent=self)
    
    def _display_statistics(self, stats: Dict[str, Any]):
        """Display database statistics.
        
        Args:
            stats: Statistics dictionary
        """
        # Clear existing statistics
        while self.stats_layout.count():
            item = self.stats_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Timestamp
        timestamp_label = QLabel(f"Statistics collected at: {stats['timestamp']}")
        timestamp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        timestamp_label.setStyleSheet("font-weight: bold;")
        self.stats_layout.addWidget(timestamp_label)
        
        # Database statistics
        db_group = QGroupBox("Database Information")
        db_layout = QVBoxLayout()
        db_group.setLayout(db_layout)
        
        db_info = stats['database']
        db_layout.addWidget(QLabel(f"Database path: {db_info['path']}"))
        db_layout.addWidget(QLabel(f"Database size: {db_info['size_formatted']}"))
        db_layout.addWidget(QLabel(f"Integrity check: {db_info['integrity_check']}"))
        
        # Table statistics
        table_tree = QTreeWidget()
        table_tree.setHeaderLabels(["Table", "Rows", "Size"])
        table_tree.setColumnWidth(0, 150)
        table_tree.setColumnWidth(1, 100)
        
        for table_name, table_stats in db_info['tables'].items():
            item = QTreeWidgetItem([
                table_name,
                str(table_stats['row_count']),
                self.data_manager._format_size(table_stats['size_bytes'])
            ])
            table_tree.addTopLevelItem(item)
        
        db_layout.addWidget(table_tree)
        self.stats_layout.addWidget(db_group)
        
        # Recording statistics
        rec_group = QGroupBox("Recordings")
        rec_layout = QVBoxLayout()
        rec_group.setLayout(rec_layout)
        
        rec_info = stats['recordings']
        rec_layout.addWidget(QLabel(f"Number of recordings: {rec_info['count']}"))
        rec_layout.addWidget(QLabel(f"Total duration: {rec_info['total_duration_formatted']}"))
        rec_layout.addWidget(QLabel(f"Average duration: {self.data_manager._format_duration(rec_info['avg_duration'])}"))
        
        self.stats_layout.addWidget(rec_group)
        
        # Speaker statistics
        speaker_group = QGroupBox("Speakers")
        speaker_layout = QVBoxLayout()
        speaker_group.setLayout(speaker_layout)
        
        speaker_info = stats['speakers']
        speaker_layout.addWidget(QLabel(f"Number of speakers: {speaker_info['count']}"))
        speaker_layout.addWidget(QLabel(f"Speakers with voice prints: {speaker_info['with_voice_prints']}"))
        speaker_layout.addWidget(QLabel(
            f"Average speaking time per segment: {self.data_manager._format_duration(speaker_info['avg_speaking_time'])}"
        ))
        
        self.stats_layout.addWidget(speaker_group)
        
        # Transcript statistics
        transcript_group = QGroupBox("Transcripts")
        transcript_layout = QVBoxLayout()
        transcript_group.setLayout(transcript_layout)
        
        transcript_info = stats['transcripts']
        transcript_layout.addWidget(QLabel(f"Number of segments: {transcript_info['segment_count']}"))
        transcript_layout.addWidget(QLabel(f"Number of words: {transcript_info['word_count']}"))
        transcript_layout.addWidget(QLabel(f"Average words per segment: {transcript_info['avg_words_per_segment']:.2f}"))
        
        self.stats_layout.addWidget(transcript_group)
        
        # Add some spacing
        self.stats_layout.addStretch()
    
    def create_backup(self):
        """Create a database backup."""
        try:
            # Ask for backup location
            backup_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Backup",
                str(Path.home()),
                "Backup Files (*.zip)"
            )
            
            if not backup_path:
                return  # User canceled
            
            # Show progress dialog
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("Creating Backup")
            progress_dialog.setFixedSize(300, 100)
            
            progress_layout = QVBoxLayout()
            progress_dialog.setLayout(progress_layout)
            
            progress_label = QLabel("Creating database backup...")
            progress_layout.addWidget(progress_label)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 0)  # Indeterminate
            progress_layout.addWidget(progress_bar)
            
            # Create background worker
            worker = BackgroundWorker(self.data_manager.create_backup, backup_path)
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(lambda path: self._backup_completed(path, progress_dialog))
            worker.error.connect(lambda e: self._handle_error(e, "Error creating backup", progress_dialog))
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            
            # Show dialog and start thread
            progress_dialog.show()
            thread.start()
            
        except Exception as e:
            ErrorHandler.handle_exception(e, parent=self)
    
    def _backup_completed(self, backup_path: str, progress_dialog: QDialog):
        """Handle backup completion.
        
        Args:
            backup_path: Path to the created backup file
            progress_dialog: Progress dialog to close
        """
        # Close progress dialog
        progress_dialog.accept()
        
        # Show success message
        QMessageBox.information(
            self,
            "Backup Created",
            f"Database backup created successfully at:\n{backup_path}"
        )
    
    def restore_backup(self):
        """Restore the database from a backup."""
        try:
            # Ask for confirmation
            confirm = QMessageBox.warning(
                self,
                "Confirm Restore",
                "This will replace the current database with the backup. This operation cannot be undone.\n\n"
                "A backup of the current database will be created before restoring.\n\n"
                "Do you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if confirm != QMessageBox.StandardButton.Yes:
                return
            
            # Ask for backup file
            backup_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Backup File",
                str(Path.home()),
                "Backup Files (*.zip)"
            )
            
            if not backup_path:
                return  # User canceled
            
            # Show progress dialog
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("Restoring Backup")
            progress_dialog.setFixedSize(300, 100)
            
            progress_layout = QVBoxLayout()
            progress_dialog.setLayout(progress_layout)
            
            progress_label = QLabel("Restoring database from backup...")
            progress_layout.addWidget(progress_label)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 0)  # Indeterminate
            progress_layout.addWidget(progress_bar)
            
            # Create background worker
            worker = BackgroundWorker(self.data_manager.restore_backup, backup_path)
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(lambda result: self._restore_completed(result, progress_dialog))
            worker.error.connect(lambda e: self._handle_error(e, "Error restoring backup", progress_dialog))
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            
            # Show dialog and start thread
            progress_dialog.show()
            thread.start()
            
        except Exception as e:
            ErrorHandler.handle_exception(e, parent=self)
    
    def _restore_completed(self, result: bool, progress_dialog: QDialog):
        """Handle restore completion.
        
        Args:
            result: Result of the restore operation
            progress_dialog: Progress dialog to close
        """
        # Close progress dialog
        progress_dialog.accept()
        
        # Show success message
        QMessageBox.information(
            self,
            "Backup Restored",
            "Database has been restored successfully from the backup."
        )
        
        # Refresh data
        self.refresh_statistics()
        self.load_recordings()
        self.load_speakers()
    
    def load_recordings(self):
        """Load recordings into the list."""
        try:
            # Clear existing items
            self.recordings_list.clear()
            
            # Get recordings
            recordings = self.db_manager.get_all_recordings(self.session)
            
            # Add recordings to list
            for recording in recordings:
                item = QTreeWidgetItem([
                    str(recording.id),
                    recording.filename,
                    self.data_manager._format_duration(recording.duration),
                    recording.created_at.strftime("%Y-%m-%d %H:%M:%S") if recording.created_at else "",
                    ", ".join([s.name or f"Speaker {s.id}" for s in recording.speakers])
                ])
                self.recordings_list.addTopLevelItem(item)
            
        except Exception as e:
            ErrorHandler.handle_exception(e, parent=self)
    
    def load_speakers(self):
        """Load speakers into the list."""
        try:
            # Clear existing items
            self.speakers_list.clear()
            
            # Get speakers
            speakers = self.db_manager.get_all_speakers(self.session)
            
            # Add speakers to list
            for speaker in speakers:
                self.speakers_list.addItem(speaker.name or f"Speaker {speaker.id}")
                self.speakers_list.item(self.speakers_list.count() - 1).setData(Qt.ItemDataRole.UserRole, speaker.id)
            
        except Exception as e:
            ErrorHandler.handle_exception(e, parent=self)
    
    def archive_recording(self):
        """Archive the selected recording."""
        try:
            # Get selected recording
            selected_items = self.recordings_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(
                    self,
                    "No Recording Selected",
                    "Please select a recording to archive."
                )
                return
            
            recording_id = int(selected_items[0].text(0))
            
            # Ask for archive location
            archive_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Archive Directory",
                str(Path.home())
            )
            
            if not archive_dir:
                return  # User canceled
            
            # Show progress dialog
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("Archiving Recording")
            progress_dialog.setFixedSize(300, 100)
            
            progress_layout = QVBoxLayout()
            progress_dialog.setLayout(progress_layout)
            
            progress_label = QLabel("Archiving recording...")
            progress_layout.addWidget(progress_label)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 0)  # Indeterminate
            progress_layout.addWidget(progress_bar)
            
            # Create background worker
            worker = BackgroundWorker(
                self.data_manager.archive_recording,
                self.session,
                recording_id,
                archive_dir
            )
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(lambda path: self._archive_completed(path, progress_dialog))
            worker.error.connect(lambda e: self._handle_error(e, "Error archiving recording", progress_dialog))
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            
            # Show dialog and start thread
            progress_dialog.show()
            thread.start()
            
        except Exception as e:
            ErrorHandler.handle_exception(e, parent=self)
    
    def _archive_completed(self, archive_path: str, progress_dialog: QDialog):
        """Handle archive completion.
        
        Args:
            archive_path: Path to the created archive file
            progress_dialog: Progress dialog to close
        """
        # Close progress dialog
        progress_dialog.accept()
        
        # Show success message
        QMessageBox.information(
            self,
            "Recording Archived",
            f"Recording archived successfully to:\n{archive_path}"
        )
    
    def restore_archive(self):
        """Restore a recording from an archive."""
        try:
            # Ask for archive file
            archive_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Archive File",
                str(Path.home()),
                "Archive Files (*.zip)"
            )
            
            if not archive_path:
                return  # User canceled
            
            # Show progress dialog
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("Restoring Archive")
            progress_dialog.setFixedSize(300, 100)
            
            progress_layout = QVBoxLayout()
            progress_dialog.setLayout(progress_layout)
            
            progress_label = QLabel("Restoring recording from archive...")
            progress_layout.addWidget(progress_label)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 0)  # Indeterminate
            progress_layout.addWidget(progress_bar)
            
            # Create background worker
            worker = BackgroundWorker(
                self.data_manager.restore_archive,
                self.session,
                archive_path
            )
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(lambda recording_id: self._restore_archive_completed(recording_id, progress_dialog))
            worker.error.connect(lambda e: self._handle_error(e, "Error restoring archive", progress_dialog))
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            
            # Show dialog and start thread
            progress_dialog.show()
            thread.start()
            
        except Exception as e:
            ErrorHandler.handle_exception(e, parent=self)
    
    def _restore_archive_completed(self, recording_id: int, progress_dialog: QDialog):
        """Handle archive restore completion.
        
        Args:
            recording_id: ID of the restored recording
            progress_dialog: Progress dialog to close
        """
        # Close progress dialog
        progress_dialog.accept()
        
        # Show success message
        QMessageBox.information(
            self,
            "Archive Restored",
            f"Recording restored successfully with ID: {recording_id}"
        )
        
        # Refresh recordings list
        self.load_recordings()
    
    def execute_pruning(self):
        """Execute data pruning with the selected rules."""
        try:
            # Collect pruning rules
            rules = {}
            
            if self.age_checkbox.isChecked():
                rules['older_than_days'] = self.age_spinbox.value()
            
            if self.unprocessed_checkbox.isChecked():
                rules['remove_unprocessed'] = True
            
            if self.min_duration_checkbox.isChecked():
                rules['min_duration'] = self.min_duration_spinbox.value()
            
            if self.max_duration_checkbox.isChecked():
                rules['max_duration'] = self.max_duration_spinbox.value()
            
            if self.speaker_checkbox.isChecked():
                selected_items = self.speakers_list.selectedItems()
                if selected_items:
                    rules['speakers'] = [
                        item.data(Qt.ItemDataRole.UserRole)
                        for item in selected_items
                    ]
            
            if self.orphaned_checkbox.isChecked():
                rules['remove_orphaned_speakers'] = True
            
            # Check if any rules are selected
            if not rules:
                QMessageBox.warning(
                    self,
                    "No Rules Selected",
                    "Please select at least one pruning rule."
                )
                return
            
            # Ask for confirmation
            confirm = QMessageBox.warning(
                self,
                "Confirm Data Pruning",
                "This will permanently remove data based on the selected rules. This operation cannot be undone.\n\n"
                "Do you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if confirm != QMessageBox.StandardButton.Yes:
                return
            
            # Show progress dialog
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("Pruning Data")
            progress_dialog.setFixedSize(300, 100)
            
            progress_layout = QVBoxLayout()
            progress_dialog.setLayout(progress_layout)
            
            progress_label = QLabel("Pruning database...")
            progress_layout.addWidget(progress_label)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 0)  # Indeterminate
            progress_layout.addWidget(progress_bar)
            
            # Create background worker
            worker = BackgroundWorker(
                self.data_manager.prune_data,
                self.session,
                rules
            )
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(lambda results: self._pruning_completed(results, progress_dialog))
            worker.error.connect(lambda e: self._handle_error(e, "Error pruning data", progress_dialog))
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            
            # Show dialog and start thread
            progress_dialog.show()
            thread.start()
            
        except Exception as e:
            ErrorHandler.handle_exception(e, parent=self)
    
    def _pruning_completed(self, results: Dict[str, Any], progress_dialog: QDialog):
        """Handle pruning completion.
        
        Args:
            results: Results of the pruning operation
            progress_dialog: Progress dialog to close
        """
        # Close progress dialog
        progress_dialog.accept()
        
        # Show success message
        QMessageBox.information(
            self,
            "Data Pruning Complete",
            f"Pruning operation completed successfully.\n\n"
            f"Recordings removed: {results['recordings_removed']}\n"
            f"Segments removed: {results['segments_removed']}\n"
            f"Words removed: {results['words_removed']}\n"
            f"Speakers removed: {results['speakers_removed']}\n"
            f"Space freed: {results.get('size_freed_formatted', '0 B')}"
        )
        
        # Refresh data
        self.refresh_statistics()
        self.load_recordings()
        self.load_speakers()
    
    def _handle_error(self, exception: Exception, title: str, progress_dialog: Optional[QDialog] = None):
        """Handle errors from background operations.
        
        Args:
            exception: The exception that occurred
            title: Title for the error dialog
            progress_dialog: Optional progress dialog to close
        """
        # Close progress dialog if provided
        if progress_dialog:
            progress_dialog.accept()
        
        # Handle the exception
        ErrorHandler.handle_exception(exception, title=title, parent=self)
    
    def closeEvent(self, event):
        """Handle dialog close event.
        
        Args:
            event: Close event
        """
        # Close session
        self.db_manager.close_session(self.session)
        
        # Accept the event
        event.accept() 