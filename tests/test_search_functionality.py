#!/usr/bin/env python3
"""
Tests for search functionality in VSAT.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import re

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.ui.search_panel import SearchPanel
from src.ui.search_result import SearchResult
from src.ui.main_window import MainWindow


class TestSearchPanel(unittest.TestCase):
    """Test case for the search panel."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        # Create QApplication instance if not already created
        cls.app = QApplication.instance()
        if not cls.app:
            cls.app = QApplication(sys.argv)
    
    def setUp(self):
        """Set up test case."""
        self.search_panel = SearchPanel()
    
    def test_search_panel_initialization(self):
        """Test that search panel initializes correctly."""
        # Check that search panel is properly initialized
        self.assertIsNotNone(self.search_panel.search_input)
        self.assertIsNotNone(self.search_panel.search_button)
        self.assertIsNotNone(self.search_panel.results_label)
        self.assertIsNotNone(self.search_panel.results_scroll)
        self.assertIsNotNone(self.search_panel.case_sensitive_check)
        self.assertIsNotNone(self.search_panel.whole_word_check)
        self.assertIsNotNone(self.search_panel.regex_check)
        self.assertIsNotNone(self.search_panel.prev_button)
        self.assertIsNotNone(self.search_panel.next_button)
        
        # Check default state
        self.assertEqual(self.search_panel.search_input.text(), "")
        self.assertEqual(self.search_panel.results_label.text(), "No results")
        self.assertFalse(self.search_panel.case_sensitive_check.isChecked())
        self.assertFalse(self.search_panel.whole_word_check.isChecked())
        self.assertFalse(self.search_panel.regex_check.isChecked())
        self.assertFalse(self.search_panel.prev_button.isEnabled())
        self.assertFalse(self.search_panel.next_button.isEnabled())
    
    def test_search_signal_emission(self):
        """Test that search emits correct signal."""
        # Create a mock to receive the signal
        mock_receiver = MagicMock()
        self.search_panel.searchRequested.connect(mock_receiver)
        
        # Set search text and trigger search
        test_query = "test query"
        self.search_panel.search_input.setText(test_query)
        self.search_panel.search()
        
        # Check that signal was emitted with correct query and options
        expected_options = {
            'case_sensitive': False,
            'whole_word': False,
            'regex': False
        }
        mock_receiver.assert_called_once_with(test_query, expected_options)
        
        # Now test with different options
        mock_receiver.reset_mock()
        self.search_panel.case_sensitive_check.setChecked(True)
        self.search_panel.whole_word_check.setChecked(True)
        self.search_panel.search()
        
        expected_options = {
            'case_sensitive': True,
            'whole_word': True,
            'regex': False
        }
        mock_receiver.assert_called_once_with(test_query, expected_options)
    
    def test_clear_results(self):
        """Test clearing results."""
        # First add some dummy results
        dummy_results = [
            {"text": "test1", "start": 0, "end": 1, "speaker": "Speaker 1"},
            {"text": "test2", "start": 2, "end": 3, "speaker": "Speaker 2"}
        ]
        self.search_panel.display_results(dummy_results)
        
        # Now clear results
        self.search_panel.clear_results()
        
        # Check that results are cleared
        self.assertEqual(len(self.search_panel.search_results), 0)
        self.assertEqual(self.search_panel.results_label.text(), "No results")
        self.assertEqual(self.search_panel.current_page, 0)
        self.assertFalse(self.search_panel.prev_button.isEnabled())
        self.assertFalse(self.search_panel.next_button.isEnabled())
    
    def test_display_results(self):
        """Test displaying search results."""
        # Create dummy results
        dummy_results = [
            {"text": "test1", "start": 0, "end": 1, "speaker": "Speaker 1", "context": "this is **test1** context"},
            {"text": "test2", "start": 2, "end": 3, "speaker": "Speaker 2", "context": "this is **test2** context"}
        ]
        
        # Display results
        self.search_panel.display_results(dummy_results)
        
        # Check that results are stored
        self.assertEqual(self.search_panel.search_results, dummy_results)
        
        # Check that results label is updated
        self.assertEqual(self.search_panel.results_label.text(), "2 results found")
        
        # Check that result widgets are created (count-1 because there's a stretch at the end)
        self.assertEqual(self.search_panel.results_layout.count() - 1, 2)
    
    def test_pagination(self):
        """Test pagination functionality."""
        # Create enough dummy results to span multiple pages
        dummy_results = []
        for i in range(25):  # More than 2 pages with default results_per_page = 10
            dummy_results.append({
                "text": f"test{i}", 
                "start": i, 
                "end": i+1, 
                "speaker": f"Speaker {i%5+1}",
                "context": f"this is **test{i}** context"
            })
        
        # Display results
        self.search_panel.display_results(dummy_results)
        
        # Check initial pagination state
        self.assertEqual(self.search_panel.current_page, 0)
        self.assertTrue(self.search_panel.next_button.isEnabled())
        self.assertFalse(self.search_panel.prev_button.isEnabled())
        
        # Check that the correct number of results are displayed (10 per page by default)
        self.assertEqual(self.search_panel.results_layout.count() - 1, 10)
        
        # Test next page
        self.search_panel.next_page()
        self.assertEqual(self.search_panel.current_page, 1)
        self.assertTrue(self.search_panel.next_button.isEnabled())
        self.assertTrue(self.search_panel.prev_button.isEnabled())
        
        # Go to last page
        self.search_panel.next_page()
        self.assertEqual(self.search_panel.current_page, 2)
        self.assertFalse(self.search_panel.next_button.isEnabled())
        self.assertTrue(self.search_panel.prev_button.isEnabled())
        
        # Test previous page
        self.search_panel.previous_page()
        self.assertEqual(self.search_panel.current_page, 1)
        self.assertTrue(self.search_panel.next_button.isEnabled())
        self.assertTrue(self.search_panel.prev_button.isEnabled())
    
    def test_set_segments(self):
        """Test setting transcript segments."""
        # Create dummy segments
        dummy_segments = [
            {"text": "Segment 1", "start": 0, "end": 1, "speaker": "Speaker 1"},
            {"text": "Segment 2", "start": 2, "end": 3, "speaker": "Speaker 2"}
        ]
        
        # Set dummy segments
        self.search_panel.set_segments(dummy_segments)
        
        # Check that segments are stored
        self.assertEqual(self.search_panel.current_transcript, dummy_segments)
        
        # Check that search is cleared
        self.assertEqual(self.search_panel.search_input.text(), "")
        self.assertEqual(self.search_panel.results_label.text(), "No results")


class TestSearchResult(unittest.TestCase):
    """Test case for the search result widget."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        # Create QApplication instance if not already created
        cls.app = QApplication.instance()
        if not cls.app:
            cls.app = QApplication(sys.argv)
    
    def setUp(self):
        """Set up test case."""
        self.result_data = {
            "text": "test result",
            "start": 10.5,
            "end": 12.75,
            "speaker": "Speaker 1",
            "context": "this is **test result** context"
        }
        self.result_widget = SearchResult(self.result_data, 1)
    
    def test_result_initialization(self):
        """Test that search result initializes correctly."""
        # Check that result widget is properly initialized
        self.assertIsNotNone(self.result_widget.number_label)
        self.assertIsNotNone(self.result_widget.speaker_label)
        self.assertIsNotNone(self.result_widget.time_label)
        self.assertIsNotNone(self.result_widget.context_label)
        self.assertIsNotNone(self.result_widget.play_button)
        self.assertIsNotNone(self.result_widget.jump_button)
        self.assertIsNotNone(self.result_widget.copy_button)
        self.assertIsNotNone(self.result_widget.speaker_indicator)
        
        # Check content
        self.assertEqual(self.result_widget.number_label.text(), "Result #1")
        self.assertEqual(self.result_widget.speaker_label.text(), "Speaker: Speaker 1")
        
        # Check formatted time (format changed, now includes decimal points)
        time_text = self.result_widget.time_label.text()
        self.assertTrue(time_text.startswith("00:10"))
        self.assertTrue(time_text.endswith("s)"))  # Ends with duration in seconds
        
        # Check that context is set and includes HTML formatting
        self.assertTrue("background-color: #FFFF00" in self.result_widget.context_label.text())
        self.assertTrue("test result" in self.result_widget.context_label.text())
    
    def test_get_speaker_color(self):
        """Test speaker color generation."""
        # Test predefined speaker colors
        self.assertEqual(self.result_widget._get_speaker_color("Speaker 1"), "#FF6B6B")
        self.assertEqual(self.result_widget._get_speaker_color("Speaker 2"), "#4ECDC4")
        self.assertEqual(self.result_widget._get_speaker_color("Unknown"), "#CCCCCC")
        
        # Test dynamic color generation for non-predefined speaker
        color = self.result_widget._get_speaker_color("Custom Speaker")
        self.assertTrue(color.startswith("#"))
        self.assertEqual(len(color), 7)  # Standard hex color format #RRGGBB
    
    def test_result_signal_emission(self):
        """Test that clicking buttons emits correct signals."""
        # Create a mock to receive the signal
        mock_receiver = MagicMock()
        self.result_widget.result_clicked.connect(mock_receiver)
        
        # Trigger play button click
        self.result_widget.play_button.click()
        mock_receiver.assert_called_once_with(self.result_data)
        
        # Reset mock and test jump button
        mock_receiver.reset_mock()
        self.result_widget.jump_button.click()
        mock_receiver.assert_called_once_with(self.result_data)
    
    def test_time_formatting(self):
        """Test time formatting."""
        # Test some time values with the new format (includes milliseconds)
        self.assertEqual(self.result_widget._format_time(0), "00:00.00")
        self.assertEqual(self.result_widget._format_time(61.5), "01:01.50")
        self.assertEqual(self.result_widget._format_time(3661.75), "61:01.75")


class TestSearchIntegration(unittest.TestCase):
    """Integration tests for search functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        # Create QApplication instance if not already created
        cls.app = QApplication.instance()
        if not cls.app:
            cls.app = QApplication(sys.argv)
    
    @patch('src.ui.main_window.MainWindow', autospec=True)
    def test_search_integration(self, mock_main_window):
        """Test integration between search panel and main window."""
        # Create mock main window with mocked methods
        mock_main_window.search_panel = SearchPanel()
        mock_main_window.on_search_requested = MagicMock()
        mock_main_window.on_search_result_selected = MagicMock()
        
        # Connect signals
        mock_main_window.search_panel.searchRequested.connect(mock_main_window.on_search_requested)
        mock_main_window.search_panel.resultSelected.connect(mock_main_window.on_search_result_selected)
        
        # Perform search
        query = "test query"
        mock_main_window.search_panel.search_input.setText(query)
        mock_main_window.search_panel.search()
        
        # Check that on_search_requested was called with query and default options
        expected_options = {'case_sensitive': False, 'whole_word': False, 'regex': False}
        mock_main_window.on_search_requested.assert_called_once_with(query, expected_options)
        
        # Create dummy result and simulate selection
        dummy_result = {"text": "test", "start": 0, "end": 1}
        mock_main_window.search_panel.resultSelected.emit(dummy_result)
        
        # Check that on_search_result_selected was called
        mock_main_window.on_search_result_selected.assert_called_once_with(dummy_result)
    
    @patch('src.ui.main_window.MainWindow')
    def test_advanced_search_options(self, mock_main_window_class):
        """Test that advanced search options work correctly."""
        # Create a mock instance
        mock_main_window = mock_main_window_class.return_value
        
        # Create transcript segments with test data
        segments = [
            {
                "text": "This is a test transcript with Test words.",
                "speaker": "Speaker 1",
                "start": 0,
                "end": 5,
                "words": [
                    {"text": "This", "start": 0, "end": 0.5},
                    {"text": "is", "start": 0.5, "end": 0.7},
                    {"text": "a", "start": 0.7, "end": 0.8},
                    {"text": "test", "start": 0.8, "end": 1.2},
                    {"text": "transcript", "start": 1.2, "end": 2},
                    {"text": "with", "start": 2, "end": 2.5},
                    {"text": "Test", "start": 2.5, "end": 3},
                    {"text": "words.", "start": 3, "end": 5}
                ]
            }
        ]
        
        # Set up the mock
        mock_main_window.segments = segments
        
        # Create a real search panel to test with
        search_panel = SearchPanel()
        mock_main_window.search_panel = search_panel
        
        # Patch the main window's methods to use real implementation but with our mock
        with patch('src.ui.main_window.MainWindow.on_search_requested', 
                  lambda self, q, opts: MainWindow.on_search_requested(mock_main_window, q, opts)):
            
            # Test case insensitive search (default)
            query = "test"
            search_panel.search_input.setText(query)
            search_panel.search()
            
            # Check that we found both "test" and "Test"
            self.assertEqual(len(search_panel.search_results), 2)
            
            # Test case sensitive search
            search_panel.clear_results()
            search_panel.case_sensitive_check.setChecked(True)
            search_panel.search()
            
            # Should only find lowercase "test"
            self.assertEqual(len(search_panel.search_results), 1)
            self.assertEqual(search_panel.search_results[0]["text"], "test")
            
            # Test whole word search
            search_panel.clear_results()
            search_panel.case_sensitive_check.setChecked(False)
            search_panel.whole_word_check.setChecked(True)
            
            # Set query to "test"
            search_panel.search_input.setText("test")
            search_panel.search()
            
            # Should find both "test" and "Test" as whole words
            self.assertEqual(len(search_panel.search_results), 2)
            
            # Test regex search
            search_panel.clear_results()
            search_panel.whole_word_check.setChecked(False)
            search_panel.regex_check.setChecked(True)
            
            # Set regex query to find words ending with "st"
            search_panel.search_input.setText(r"\w+st\b")
            search_panel.search()
            
            # Should find both "test" and "Test"
            self.assertEqual(len(search_panel.search_results), 2)


if __name__ == "__main__":
    unittest.main() 