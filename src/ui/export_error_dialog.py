"""
Export error dialog for VSAT.

This module provides a dialog for displaying export errors and offering recovery options.
"""

import logging
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QCheckBox, QDialogButtonBox,
    QHeaderView, QMessageBox, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from src.ui.export_error_tracker import ExportAttempt, ExportOperation

logger = logging.getLogger(__name__)

class ExportErrorDialog(QDialog):
    """Dialog for displaying and managing export errors."""
    
    # Signal emitted when the user wants to retry an export
    retryRequested = pyqtSignal(object)  # ExportAttempt
    
    # Signal emitted when the user wants to retry all failed exports
    retryAllRequested = pyqtSignal(object)  # List[ExportAttempt]
    
    def __init__(self, parent=None, failures: List[ExportAttempt] = None):
        """Initialize the export error dialog.
        
        Args:
            parent: Parent widget
            failures: List of failed export attempts
        """
        super().__init__(parent)
        
        self.failures = failures or []
        self.selected_failures = []
        
        self._setup_ui()
        self._populate_table()
        
    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Export Errors")
        self.setMinimumSize(800, 400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add header
        header_label = QLabel("The following export operations failed:")
        header_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(header_label)
        
        # Add table for failed exports
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Select", "Operation", "Error", "Time", "Retries", "Target Path"
        ])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.table)
        
        # Add error details section
        layout.addWidget(QLabel("Error Details:"))
        self.details_edit = QTextEdit()
        self.details_edit.setReadOnly(True)
        self.details_edit.setMaximumHeight(100)
        layout.addWidget(self.details_edit)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._on_select_all)
        button_layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self._on_deselect_all)
        button_layout.addWidget(self.deselect_all_btn)
        
        button_layout.addStretch()
        
        self.retry_selected_btn = QPushButton("Retry Selected")
        self.retry_selected_btn.setEnabled(False)
        self.retry_selected_btn.clicked.connect(self._on_retry_selected)
        button_layout.addWidget(self.retry_selected_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def _populate_table(self):
        """Populate the table with failed export attempts."""
        self.table.setRowCount(len(self.failures))
        
        for row, failure in enumerate(self.failures):
            # Checkbox for selection
            checkbox = QCheckBox()
            self.table.setCellWidget(row, 0, checkbox)
            checkbox.stateChanged.connect(lambda state, r=row: self._on_checkbox_changed(r, state))
            
            # Operation type
            op_item = QTableWidgetItem(failure.operation.value.replace("_", " ").title())
            op_item.setData(Qt.ItemDataRole.UserRole, failure)
            self.table.setItem(row, 1, op_item)
            
            # Error message
            error_item = QTableWidgetItem(failure.error_message or "Unknown error")
            self.table.setItem(row, 2, error_item)
            
            # Time
            time_str = datetime.fromtimestamp(failure.start_time).strftime("%H:%M:%S")
            time_item = QTableWidgetItem(time_str)
            self.table.setItem(row, 3, time_item)
            
            # Retry count
            retry_item = QTableWidgetItem(str(failure.retry_count))
            self.table.setItem(row, 4, retry_item)
            
            # Target path
            path_item = QTableWidgetItem(failure.target_path or "N/A")
            self.table.setItem(row, 5, path_item)
    
    def _on_checkbox_changed(self, row: int, state: int):
        """Handle checkbox state changes.
        
        Args:
            row: Row index
            state: Checkbox state
        """
        if row < 0 or row >= len(self.failures):
            return
            
        failure = self.failures[row]
        
        if state == Qt.CheckState.Checked.value:
            if failure not in self.selected_failures:
                self.selected_failures.append(failure)
        else:
            if failure in self.selected_failures:
                self.selected_failures.remove(failure)
                
        # Update retry button state
        self.retry_selected_btn.setEnabled(len(self.selected_failures) > 0)
    
    def _on_item_clicked(self, item: QTableWidgetItem):
        """Handle item clicks.
        
        Args:
            item: The clicked item
        """
        row = item.row()
        failure = self.failures[row]
        
        # Show error details
        details = f"Error: {failure.error_message or 'Unknown error'}\n\n"
        
        if failure.error_details:
            details += "Details:\n"
            for key, value in failure.error_details.items():
                details += f"  {key}: {value}\n"
        
        self.details_edit.setText(details)
    
    def _on_select_all(self):
        """Select all failures."""
        self.selected_failures = self.failures.copy()
        
        for row in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(True)
                
        self.retry_selected_btn.setEnabled(len(self.selected_failures) > 0)
    
    def _on_deselect_all(self):
        """Deselect all failures."""
        self.selected_failures = []
        
        for row in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(False)
                
        self.retry_selected_btn.setEnabled(False)
    
    def _on_retry_selected(self):
        """Handle retry selected button click."""
        if not self.selected_failures:
            return
            
        # Emit signal for all selected failures
        self.retryAllRequested.emit(self.selected_failures)
        
        # Close dialog
        self.accept()
        
    def set_failures(self, failures: List[ExportAttempt]):
        """Set the list of failures to display.
        
        Args:
            failures: List of failed export attempts
        """
        self.failures = failures or []
        self.selected_failures = []
        self._populate_table()
        self.retry_selected_btn.setEnabled(False) 