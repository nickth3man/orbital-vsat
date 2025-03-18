"""
Export dialog tabs for VSAT.

This module provides functions for creating the tabs in the export dialog.
"""

import os
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QCheckBox, QGroupBox,
    QSpinBox, QFormLayout
)

logger = logging.getLogger(__name__)


def create_transcript_tab(dialog) -> QWidget:
    """Create the transcript export tab.

    Args:
        dialog: Parent export dialog

    Returns:
        QWidget: Tab widget
    """
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Create form layout
    form_layout = QFormLayout()

    # Format selection
    dialog.transcript_format = QComboBox()
    for format_key, format_name in dialog.export_manager.TRANSCRIPT_FORMATS.items():
        dialog.transcript_format.addItem(
            format_name, 
            format_key
        )
    form_layout.addRow("Format:", dialog.transcript_format)

    # Output file
    output_layout = QHBoxLayout()
    dialog.transcript_path = QLineEdit()

    # Set default transcript output filename
    if dialog.audio_file:
        base_dir = os.path.dirname(dialog.audio_file)
        base_name = os.path.splitext(os.path.basename(dialog.audio_file))[0]
        dialog.transcript_path.setText(
            os.path.join(base_dir, f"{base_name}_transcript.txt")
        )

    output_layout.addWidget(dialog.transcript_path)

    browse_button = QPushButton("Browse...")
    browse_button.clicked.connect(dialog.on_browse_transcript)
    output_layout.addWidget(browse_button)

    form_layout.addRow("Output file:", output_layout)

    # Options
    options_group = QGroupBox("Options")
    options_layout = QVBoxLayout(options_group)

    dialog.include_speaker = QCheckBox("Include speaker information")
    dialog.include_speaker.setChecked(True)
    options_layout.addWidget(dialog.include_speaker)

    dialog.include_timestamps = QCheckBox("Include timestamps")
    dialog.include_timestamps.setChecked(True)
    options_layout.addWidget(dialog.include_timestamps)

    layout.addLayout(form_layout)
    layout.addWidget(options_group)
    layout.addStretch()

    return tab


def create_audio_tab(dialog) -> QWidget:
    """Create the audio segment export tab.

    Args:
        dialog: Parent export dialog

    Returns:
        QWidget: Tab widget
    """
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Create form layout
    form_layout = QFormLayout()

    # Format selection
    dialog.audio_format = QComboBox()
    for format_key, format_name in dialog.export_manager.AUDIO_FORMATS.items():
        dialog.audio_format.addItem(
            format_name, 
            format_key
        )
    form_layout.addRow("Format:", dialog.audio_format)

    # Output file
    output_layout = QHBoxLayout()
    dialog.audio_path = QLineEdit()

    # Set default audio output filename
    if dialog.audio_file:
        base_dir = os.path.dirname(dialog.audio_file)
        base_name = os.path.splitext(os.path.basename(dialog.audio_file))[0]
        dialog.audio_path.setText(
            os.path.join(base_dir, f"{base_name}_segment.wav")
        )

    output_layout.addWidget(dialog.audio_path)

    browse_button = QPushButton("Browse...")
    browse_button.clicked.connect(dialog.on_browse_audio)
    output_layout.addWidget(browse_button)

    form_layout.addRow("Output file:", output_layout)

    # Time range
    range_group = QGroupBox("Time Range")
    range_layout = QFormLayout(range_group)

    dialog.audio_start = QSpinBox()
    dialog.audio_start.setRange(0, 86400)  # 0 to 24 hours in seconds
    dialog.audio_start.setValue(0)
    dialog.audio_start.setSuffix(" s")
    range_layout.addRow("Start time:", dialog.audio_start)

    dialog.audio_end = QSpinBox()
    dialog.audio_end.setRange(0, 86400)  # 0 to 24 hours in seconds

    # Set end time to file duration if available
    if dialog.segments:
        max_end = max(segment['end'] for segment in dialog.segments)
        dialog.audio_end.setValue(int(max_end))
    else:
        dialog.audio_end.setValue(60)  # Default to 60 seconds

    dialog.audio_end.setSuffix(" s")
    range_layout.addRow("End time:", dialog.audio_end)

    layout.addLayout(form_layout)
    layout.addWidget(range_group)
    layout.addStretch()

    return tab


def create_selection_tab(dialog) -> QWidget:
    """Create the selection export tab.

    Args:
        dialog: Parent export dialog

    Returns:
        QWidget: Tab widget
    """
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Create form layout
    form_layout = QFormLayout()

    # Format selection
    dialog.selection_format = QComboBox()
    for format_key, format_name in dialog.export_manager.AUDIO_FORMATS.items():
        dialog.selection_format.addItem(
            format_name, 
            format_key
        )
    form_layout.addRow("Format:", dialog.selection_format)

    # Output file
    output_layout = QHBoxLayout()
    dialog.selection_path = QLineEdit()

    # Set default selection output filename
    if dialog.audio_file and dialog.selected_words:
        base_dir = os.path.dirname(dialog.audio_file)
        base_name = os.path.splitext(os.path.basename(dialog.audio_file))[0]
        dialog.selection_path.setText(
            os.path.join(base_dir, f"{base_name}_selection.wav")
        )

    output_layout.addWidget(dialog.selection_path)

    browse_button = QPushButton("Browse...")
    browse_button.clicked.connect(dialog.on_browse_selection)
    output_layout.addWidget(browse_button)

    form_layout.addRow("Output file:", output_layout)

    # Options
    options_group = QGroupBox("Options")
    options_layout = QVBoxLayout(options_group)

    dialog.include_selection_transcript = QCheckBox("Include transcript")
    dialog.include_selection_transcript.setChecked(True)
    options_layout.addWidget(dialog.include_selection_transcript)

    # Selection info
    info_group = QGroupBox("Selection Information")
    info_layout = QFormLayout(info_group)

    word_count = len(dialog.selected_words)
    word_count_label = QLabel(f"{word_count} words")
    info_layout.addRow("Words:", word_count_label)

    duration = 0.0
    if dialog.selected_words:
        start = min(word['start'] for word in dialog.selected_words)
        end = max(word['end'] for word in dialog.selected_words)
        duration = end - start

    duration_label = QLabel(f"{duration:.2f} seconds")
    info_layout.addRow("Duration:", duration_label)

    layout.addLayout(form_layout)
    layout.addWidget(options_group)
    layout.addWidget(info_group)
    layout.addStretch()

    # Disable tab if no selection
    if not dialog.selected_words:
        tab.setEnabled(False)

    return tab


def create_speaker_tab(dialog) -> QWidget:
    """Create the speaker export tab.

    Args:
        dialog: Parent export dialog

    Returns:
        QWidget: Tab widget
    """
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # Create form layout
    form_layout = QFormLayout()

    # Format selection
    dialog.speaker_format = QComboBox()
    for format_key, format_name in dialog.export_manager.AUDIO_FORMATS.items():
        dialog.speaker_format.addItem(
            format_name, 
            format_key
        )
    form_layout.addRow("Format:", dialog.speaker_format)

    # Output directory
    output_layout = QHBoxLayout()
    dialog.speaker_dir = QLineEdit()

    # Set default speaker output directory
    if dialog.audio_file:
        base_dir = os.path.dirname(dialog.audio_file)
        dialog.speaker_dir.setText(os.path.join(base_dir, "speaker_export"))

    output_layout.addWidget(dialog.speaker_dir)

    browse_button = QPushButton("Browse...")
    browse_button.clicked.connect(dialog.on_browse_speaker)
    output_layout.addWidget(browse_button)

    form_layout.addRow("Output directory:", output_layout)

    # Speaker selection
    speaker_group = QGroupBox("Speaker")
    speaker_layout = QVBoxLayout(speaker_group)

    dialog.speaker_combo = QComboBox()
    speaker_layout.addWidget(dialog.speaker_combo)

    layout.addLayout(form_layout)
    layout.addWidget(speaker_group)
    layout.addStretch()

    # Disable tab if no segments
    if not dialog.segments:
        tab.setEnabled(False)

    return tab