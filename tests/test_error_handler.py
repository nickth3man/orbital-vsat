"""
Unit tests for the error handling framework.
"""

import unittest
import logging
from unittest.mock import Mock, patch

from src.utils.error_handler import (
    ErrorSeverity,
    VSATError,
    FileError,
    AudioError,
    ProcessingError,
    ExportError,
    ErrorHandler
)

class TestErrorHandler(unittest.TestCase):
    """Test case for the error handling framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure logging to avoid actual logging during tests
        logging.basicConfig(level=logging.CRITICAL)
    
    def test_vsat_error_creation(self):
        """Test creating VSATError instances."""
        # Create a basic error
        error = VSATError("Test error")
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.severity, ErrorSeverity.ERROR)
        self.assertEqual(error.details, {})
        
        # Create an error with severity and details
        error = VSATError(
            "Detailed error", 
            ErrorSeverity.WARNING, 
            {"key1": "value1", "key2": 123}
        )
        self.assertEqual(str(error), "Detailed error")
        self.assertEqual(error.severity, ErrorSeverity.WARNING)
        self.assertEqual(error.details, {"key1": "value1", "key2": 123})
    
    def test_specialized_errors(self):
        """Test creating specialized error types."""
        # Test FileError
        file_error = FileError("File not found")
        self.assertEqual(str(file_error), "File not found")
        self.assertIsInstance(file_error, VSATError)
        
        # Test AudioError
        audio_error = AudioError("Audio processing failed")
        self.assertEqual(str(audio_error), "Audio processing failed")
        self.assertIsInstance(audio_error, VSATError)
        
        # Test ExportError
        export_error = ExportError(
            "Export failed", 
            ErrorSeverity.CRITICAL, 
            {"file": "test.wav"}
        )
        self.assertEqual(str(export_error), "Export failed")
        self.assertEqual(export_error.severity, ErrorSeverity.CRITICAL)
        self.assertEqual(export_error.details, {"file": "test.wav"})
        self.assertIsInstance(export_error, VSATError)
    
    @patch('vsat.src.utils.error_handler.QMessageBox')
    def test_error_handler_show_dialog(self, mock_qmessagebox):
        """Test showing error dialog."""
        # Mock QMessageBox
        mock_dialog = Mock()
        mock_qmessagebox.return_value = mock_dialog
        
        # Call show_error_dialog
        ErrorHandler.show_error_dialog("Error Title", "Error Message", "Error Details")
        
        # Check that QMessageBox was created with correct parameters
        mock_qmessagebox.assert_called_once()
        mock_dialog.setWindowTitle.assert_called_once_with("Error Title")
        mock_dialog.setText.assert_called_once_with("Error Message")
        mock_dialog.setDetailedText.assert_called_once_with("Error Details")
        mock_dialog.exec.assert_called_once()
    
    @patch('vsat.src.utils.error_handler.ErrorHandler.show_error_dialog')
    def test_handle_exception(self, mock_show_dialog):
        """Test handling exceptions."""
        # Create a test exception
        test_exception = ExportError("Test export error")
        
        # Handle exception with dialog
        result = ErrorHandler.handle_exception(test_exception, show_dialog=True)
        self.assertTrue(result)
        mock_show_dialog.assert_called_once()
        
        # Reset mock
        mock_show_dialog.reset_mock()
        
        # Handle exception without dialog
        result = ErrorHandler.handle_exception(test_exception, show_dialog=False)
        self.assertTrue(result)
        mock_show_dialog.assert_not_called()
        
        # Test with callback
        callback_mock = Mock()
        result = ErrorHandler.handle_exception(
            test_exception, 
            show_dialog=False, 
            callback=callback_mock
        )
        self.assertTrue(result)
        callback_mock.assert_called_once_with(test_exception)
    
    @patch('vsat.src.utils.error_handler.traceback.format_exc')
    @patch('vsat.src.utils.error_handler.ErrorHandler.show_error_dialog')
    def test_handle_standard_exception(self, mock_show_dialog, mock_format_exc):
        """Test handling standard Python exceptions."""
        # Mock traceback.format_exc
        mock_format_exc.return_value = "Traceback details"
        
        # Create a standard exception
        test_exception = ValueError("Invalid value")
        
        # Handle exception
        result = ErrorHandler.handle_exception(test_exception, show_dialog=True)
        self.assertTrue(result)
        
        # Check that dialog was shown with correct parameters
        mock_show_dialog.assert_called_once()
        args = mock_show_dialog.call_args[0]
        self.assertEqual(args[0], "Unexpected Error")
        self.assertEqual(args[1], "Invalid value")
        self.assertEqual(args[2], "Traceback details")


if __name__ == "__main__":
    unittest.main() 