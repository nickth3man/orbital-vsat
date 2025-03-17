"""
User interface module for VSAT.

This module provides UI components for the VSAT application:
- Main application window
- Audio waveform visualization
- Transcript view
- Search panel
- Export dialog
- Accessibility features
- VAD visualization
- Content analysis panel
- Data management dialog
"""

from src.ui.app import MainWindow
from src.ui.waveform_view import WaveformView
from src.ui.waveform_widget import WaveformWidget
from src.ui.waveform_renderer import WaveformRenderer
from src.ui.waveform_interaction import WaveformInteraction
from src.ui.transcript_view import TranscriptView
from src.ui.transcript_segment_widget import TranscriptSegmentWidget
from src.ui.search_panel import SearchPanel
from src.ui.search_result import SearchResult
from src.ui.export_dialog import ExportDialog
from src.ui.export_tabs import (
    create_transcript_tab,
    create_audio_tab,
    create_selection_tab,
    create_speaker_tab
)
from src.ui.export_handlers import ExportHandlers
from src.ui.export_error_dialog import ExportErrorDialog
from src.ui.export_error_tracker import ExportErrorTracker
from src.ui.accessibility import AccessibilityManager
from src.ui.accessibility_dialog import AccessibilityDialog
from src.ui.flow_layout import FlowLayout
from src.ui.vad_visualization import VADVisualizationWidget
from src.ui.content_analysis_panel import ContentAnalysisPanel
from src.ui.data_management_dialog import DataManagementDialog

__all__ = [
    'MainWindow',
    'WaveformView',
    'WaveformWidget',
    'WaveformRenderer',
    'WaveformInteraction',
    'TranscriptView',
    'TranscriptSegmentWidget',
    'SearchPanel',
    'SearchResult',
    'ExportDialog',
    'create_transcript_tab',
    'create_audio_tab',
    'create_selection_tab',
    'create_speaker_tab',
    'ExportHandlers',
    'ExportErrorDialog',
    'ExportErrorTracker',
    'AccessibilityManager',
    'AccessibilityDialog',
    'FlowLayout',
    'VADVisualizationWidget',
    'ContentAnalysisPanel',
    'DataManagementDialog'
]