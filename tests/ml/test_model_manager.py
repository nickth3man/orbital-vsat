"""
Tests for the ModelManager class.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.ml.model_manager import ModelManager, ModelInfo
from src.ml.error_handling import ModelLoadError, ResourceExhaustionError


class TestModelManager(unittest.TestCase):
    """Test cases for the ModelManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for models
        self.test_model_dir = tempfile.mkdtemp()
        
        # Determine the test device based on availability
        self.test_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create a test instance with shorter cache timeout for testing
        self.model_manager = ModelManager(
            model_dir=self.test_model_dir,
            cache_timeout=2,  # Short timeout for testing
            device=self.test_device,
            hf_auth_token=None,
            optimize_for_production=True
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up
        self.model_manager.shutdown()
        
        # Remove test directory
        if os.path.exists(self.test_model_dir):
            import shutil
            shutil.rmtree(self.test_model_dir)
    
    @patch('src.transcription.whisper_transcriber.WhisperTranscriber')
    def test_get_whisper_transcriber(self, mock_whisper):
        """Test getting a Whisper transcriber from the manager."""
        # Setup mock
        mock_instance = MagicMock()
        mock_whisper.return_value = mock_instance
        
        # Get the transcriber
        transcriber = self.model_manager.get_whisper_transcriber(
            model_size="tiny",  # Use tiny model for faster testing
            device="cpu"
        )
        
        # Verify the mock was called correctly
        mock_whisper.assert_called_once()
        
        # Verify the transcriber was returned
        self.assertEqual(transcriber, mock_instance)
        
        # Verify it's now in the cache
        self.assertEqual(len(self.model_manager.models), 1)
        
        # Get it again, should use cached version
        mock_whisper.reset_mock()
        transcriber2 = self.model_manager.get_whisper_transcriber(
            model_size="tiny",
            device="cpu"
        )
        
        # Verify the mock was NOT called again
        mock_whisper.assert_not_called()
        
        # Verify the same instance was returned
        self.assertEqual(transcriber2, transcriber)
    
    @patch('src.ml.diarization.Diarizer')
    def test_get_diarizer(self, mock_diarizer):
        """Test getting a Diarizer from the manager."""
        # Setup mock
        mock_instance = MagicMock()
        mock_diarizer.return_value = mock_instance
        
        # Get the diarizer
        diarizer = self.model_manager.get_diarizer(device="cpu")
        
        # Verify the mock was called correctly
        mock_diarizer.assert_called_once()
        
        # Verify the diarizer was returned
        self.assertEqual(diarizer, mock_instance)
        
        # Verify it's now in the cache
        self.assertTrue(any("diarization" in k for k in self.model_manager.models.keys()))
    
    @patch('src.ml.speaker_identification.SpeakerIdentifier')
    def test_get_speaker_identifier(self, mock_identifier):
        """Test getting a SpeakerIdentifier from the manager."""
        # Setup mock
        mock_instance = MagicMock()
        mock_identifier.return_value = mock_instance
        
        # Get the speaker identifier
        identifier = self.model_manager.get_speaker_identifier(
            device="cpu",
            similarity_threshold=0.8
        )
        
        # Verify the mock was called correctly
        mock_identifier.assert_called_once()
        
        # Verify the identifier was returned
        self.assertEqual(identifier, mock_instance)
        
        # Verify it's in the cache
        self.assertTrue(any("speaker_id" in k for k in self.model_manager.models.keys()))
    
    @patch('src.ml.voice_activity_detection.VoiceActivityDetector')
    def test_get_voice_activity_detector(self, mock_vad):
        """Test getting a VoiceActivityDetector from the manager."""
        # Setup mock
        mock_instance = MagicMock()
        mock_vad.return_value = mock_instance
        
        # Get the VAD
        vad = self.model_manager.get_voice_activity_detector(device="cpu")
        
        # Verify the mock was called correctly
        mock_vad.assert_called_once()
        
        # Verify the VAD was returned
        self.assertEqual(vad, mock_instance)
    
    @patch('src.transcription.whisper_transcriber.WhisperTranscriber')
    def test_model_caching_and_cleanup(self, mock_whisper):
        """Test model caching and automatic cleanup."""
        # Setup mock
        mock_instance = MagicMock()
        mock_whisper.return_value = mock_instance
        
        # Get a model
        self.model_manager.get_whisper_transcriber(
            model_size="tiny",
            device="cpu"
        )
        
        # Verify it's in the cache
        self.assertEqual(len(self.model_manager.models), 1)
        
        # Wait for cache timeout
        import time
        time.sleep(3)  # Longer than cache_timeout
        
        # Force a cleanup
        self.model_manager._perform_cleanup()
        
        # Verify the model was removed from cache
        self.assertEqual(len(self.model_manager.models), 0)
    
    @patch('src.transcription.whisper_transcriber.WhisperTranscriber')
    def test_model_error_handling(self, mock_whisper):
        """Test error handling during model loading."""
        # Setup mock to raise an exception
        mock_whisper.side_effect = RuntimeError("Test error")
        
        # Try to get a model, should raise ModelLoadError
        with self.assertRaises(ModelLoadError):
            self.model_manager.get_whisper_transcriber(
                model_size="tiny",
                device="cpu"
            )
        
        # Verify no models are in the cache
        self.assertEqual(len(self.model_manager.models), 0)
    
    @patch('src.ml.model_manager.ModelManager._check_resources')
    @patch('src.transcription.whisper_transcriber.WhisperTranscriber')
    def test_resource_exhaustion_handling(self, mock_whisper, mock_check_resources):
        """Test handling of resource exhaustion."""
        # Setup mocks
        mock_instance = MagicMock()
        mock_whisper.return_value = mock_instance
        
        # First call to _check_resources returns False (not enough resources)
        # Second call returns True (after cleanup)
        mock_check_resources.side_effect = [False, False]
        
        # Try to get a model on CPU, should work as fallback
        with self.assertRaises(ResourceExhaustionError):
            self.model_manager.get_whisper_transcriber(
                model_size="tiny",
                device="cpu"  # Already CPU, so can't fall back further
            )
    
    @patch('src.transcription.whisper_transcriber.WhisperTranscriber')
    def test_async_model_loading(self, mock_whisper):
        """Test asynchronous model loading."""
        # Setup mock with delay to simulate loading time
        def delayed_return(*args, **kwargs):
            import time
            time.sleep(0.5)
            return MagicMock()
        
        mock_whisper.side_effect = delayed_return
        
        # Request async loading
        model_info = self.model_manager.get_whisper_transcriber(
            model_size="tiny",
            device="cpu",
            async_load=True
        )
        
        # Verify we got a ModelInfo object
        self.assertIsInstance(model_info, ModelInfo)
        
        # Verify the model is marked as loading
        self.assertTrue(model_info.is_loading)
        
        # Wait for the model to load
        model = self.model_manager.wait_for_model(model_info)
        
        # Verify we now have a loaded model
        self.assertFalse(model_info.is_loading)
        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main() 