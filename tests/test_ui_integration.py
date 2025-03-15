"""
Tests for UI component integration.

This module contains tests to verify that UI components work together correctly.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt

from src.ui.transcript_view import TranscriptView
from src.ui.search_panel import SearchPanel
from src.ui.waveform_view import WaveformView

# Mock PyQt6 classes
class MockQApplication:
    def __init__(self, *args, **kwargs):
        pass

class MockQMainWindow:
    def __init__(self, *args, **kwargs):
        self.widgets = []
        self.setWindowTitle = MagicMock()
        self.show = MagicMock()
        self.close = MagicMock()
    
    def centralWidget(self):
        return MockQWidget()
    
    def findChildren(self, widget_type):
        return self.widgets

class MockQWidget:
    def __init__(self, *args, **kwargs):
        self.setProperty = MagicMock()
        self.property = MagicMock()
        self.setFocus = MagicMock()
        self.hasFocus = MagicMock(return_value=False)
        self.setStyleSheet = MagicMock()
        self.font = MagicMock(return_value=MockQFont())
        self.setFont = MagicMock()
        self.palette = MagicMock(return_value=MockQPalette())
        self.setPalette = MagicMock()

class MockQAction:
    def __init__(self, *args, **kwargs):
        self.setText = MagicMock()
        self.setShortcut = MagicMock()
        self.triggered = MagicMock()
        self.setEnabled = MagicMock()
        self.isEnabled = MagicMock(return_value=True)

class MockQMenu:
    def __init__(self, *args, **kwargs):
        self.setTitle = MagicMock()
        self.addAction = MagicMock(return_value=MockQAction())
        self.addSeparator = MagicMock()

class MockQMenuBar:
    def __init__(self, *args, **kwargs):
        self.addMenu = MagicMock(return_value=MockQMenu())

class MockQPalette:
    def __init__(self):
        self.setColor = MagicMock()

class MockQColor:
    def __init__(self, *args, **kwargs):
        pass

class MockQFont:
    def __init__(self):
        self.setPointSize = MagicMock()
        self.pointSize = MagicMock(return_value=10)

# Patch the imports
with patch.dict('sys.modules', {
    'PyQt6.QtWidgets': MagicMock(),
    'PyQt6.QtGui': MagicMock(),
    'PyQt6.QtCore': MagicMock()
}):
    # Now import our modules that depend on PyQt6
    from src.ui.main_window import MainWindow
    from src.ui.accessibility import AccessibilityManager

class TestUIIntegration(unittest.TestCase):
    """Test case for UI component integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test case."""
        # Create QApplication instance
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
    
    def setUp(self):
        """Set up the test."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test audio file
        self.audio_file = os.path.join(self.temp_dir.name, "test_audio.wav")
        self.create_test_audio_file()
        
        # Create test segments
        self.segments = self.create_test_segments()
        
        # Create UI components
        self.transcript_view = TranscriptView()
        self.search_panel = SearchPanel()
        self.waveform_view = WaveformView()
        
        # Set up segments
        self.transcript_view.set_segments(self.segments)
        self.search_panel.set_segments(self.segments)
    
    def tearDown(self):
        """Clean up after the test."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def create_test_audio_file(self):
        """Create a test audio file with a sine wave."""
        # Generate a 440 Hz sine wave
        sample_rate = 16000
        duration = 5.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Save as WAV file
        sf.write(self.audio_file, audio_data, sample_rate)
    
    def create_test_segments(self):
        """Create test transcript segments."""
        return [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "This is the first test segment.",
                "speaker": "A",
                "words": [
                    {"text": "This", "start": 0.0, "end": 0.3},
                    {"text": "is", "start": 0.3, "end": 0.5},
                    {"text": "the", "start": 0.5, "end": 0.7},
                    {"text": "first", "start": 0.7, "end": 1.0},
                    {"text": "test", "start": 1.0, "end": 1.5},
                    {"text": "segment", "start": 1.5, "end": 2.3},
                    {"text": ".", "start": 2.3, "end": 2.5}
                ]
            },
            {
                "start": 2.5,
                "end": 5.0,
                "text": "This is the second test segment.",
                "speaker": "B",
                "words": [
                    {"text": "This", "start": 2.5, "end": 2.8},
                    {"text": "is", "start": 2.8, "end": 3.0},
                    {"text": "the", "start": 3.0, "end": 3.2},
                    {"text": "second", "start": 3.2, "end": 3.7},
                    {"text": "test", "start": 3.7, "end": 4.2},
                    {"text": "segment", "start": 4.2, "end": 4.8},
                    {"text": ".", "start": 4.8, "end": 5.0}
                ]
            }
        ]
    
    def test_transcript_view_initialization(self):
        """Test that the transcript view initializes correctly."""
        # Check that segments were set
        self.assertEqual(len(self.transcript_view.segments), 2)
        
        # Check that word positions were created
        self.assertGreater(len(self.transcript_view.word_positions), 0)
    
    def test_search_panel_initialization(self):
        """Test that the search panel initializes correctly."""
        # Check that segments were set
        self.assertEqual(len(self.search_panel.segments), 2)
    
    def test_search_functionality(self):
        """Test that the search functionality works correctly."""
        # Perform a search
        self.search_panel.search("test", False, False)
        
        # Check that results were found
        self.assertEqual(len(self.search_panel.search_results), 2)
        
        # Check that the results are correct
        self.assertEqual(self.search_panel.search_results[0].text, "test")
        self.assertEqual(self.search_panel.search_results[1].text, "test")
    
    def test_transcript_word_selection(self):
        """Test that word selection in the transcript view works correctly."""
        # Create a signal spy
        selected_words = []
        
        def on_words_selected(words):
            nonlocal selected_words
            selected_words = words
        
        # Connect signal
        self.transcript_view.wordsSelected.connect(on_words_selected)
        
        # Select a word
        word = self.segments[0]["words"][4]  # "test" in first segment
        self.transcript_view.export_word(word)
        
        # Check that the signal was emitted with the correct word
        self.assertEqual(len(selected_words), 1)
        self.assertEqual(selected_words[0]["text"], "test")
    
    def test_component_interaction(self):
        """Test that components interact correctly."""
        # Create signal spies
        transcript_position = None
        waveform_position = None
        
        def on_transcript_position_changed(position):
            nonlocal transcript_position
            transcript_position = position
        
        def on_waveform_position_clicked(position):
            nonlocal waveform_position
            waveform_position = position
        
        # Connect signals
        self.transcript_view.positionChanged.connect(on_transcript_position_changed)
        self.waveform_view.positionClicked.connect(on_waveform_position_clicked)
        
        # Simulate a click on the waveform
        position = 1.25  # Middle of "test" in first segment
        self.waveform_view.positionClicked.emit(position)
        
        # Check that the transcript view was updated
        self.assertEqual(self.transcript_view.current_position, position)
        
        # Simulate a word click in the transcript
        word = self.segments[0]["words"][4]  # "test" in first segment
        self.transcript_view.wordClicked.emit(word)
        
        # Check that the waveform position was updated
        self.assertEqual(waveform_position, (word["start"] + word["end"]) / 2)


if __name__ == "__main__":
    unittest.main() 