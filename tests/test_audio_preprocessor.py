"""
Tests for the AudioPreprocessor class.
"""

import os
import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.audio.audio_preprocessor import AudioPreprocessor
from src.utils.error_handler import AudioError

class TestAudioPreprocessor(unittest.TestCase):
    """Test cases for the AudioPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = AudioPreprocessor()
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test audio data
        self.sample_rate = 16000
        self.duration = 3  # seconds
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        
        # Create a clean sine wave
        self.clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Create a noisy sine wave
        noise = 0.1 * np.random.randn(len(t))
        self.noisy_audio = self.clean_audio + noise
        
        # Create a stereo version
        self.stereo_audio = np.column_stack((self.noisy_audio, self.noisy_audio * 0.8))
        
        # Create test files
        self.create_test_audio_files()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def create_test_audio_files(self):
        """Create test audio files."""
        try:
            import soundfile as sf
            
            # Create a mono file
            self.mono_file = os.path.join(self.temp_dir, "mono_test.wav")
            sf.write(self.mono_file, self.noisy_audio, self.sample_rate)
            
            # Create a stereo file
            self.stereo_file = os.path.join(self.temp_dir, "stereo_test.wav")
            sf.write(self.stereo_file, self.stereo_audio, self.sample_rate)
            
        except ImportError:
            self.skipTest("soundfile not available")
    
    def test_init(self):
        """Test initialization of AudioPreprocessor."""
        self.assertIsInstance(self.preprocessor, AudioPreprocessor)
        self.assertIsNotNone(self.preprocessor.PRESETS)
        self.assertIsNotNone(self.preprocessor.EQ_PROFILES)
    
    def test_presets_and_profiles(self):
        """Test getting presets and profiles."""
        presets = self.preprocessor.get_available_presets()
        self.assertIsInstance(presets, dict)
        self.assertIn("default", presets)
        
        profiles = self.preprocessor.get_available_eq_profiles()
        self.assertIsInstance(profiles, dict)
        self.assertIn("flat", profiles)
    
    def test_apply_noise_reduction(self):
        """Test noise reduction functionality."""
        # Apply noise reduction
        reduced_audio = self.preprocessor.apply_noise_reduction(
            self.noisy_audio,
            self.sample_rate,
            threshold=0.01,
            reduction_factor=0.7
        )
        
        # Check that output has same shape as input
        self.assertEqual(reduced_audio.shape, self.noisy_audio.shape)
        
        # Check that noise has been reduced (lower standard deviation)
        self.assertLess(np.std(reduced_audio), np.std(self.noisy_audio))
    
    def test_apply_noise_reduction_stereo(self):
        """Test noise reduction on stereo audio."""
        # Apply noise reduction to stereo audio
        reduced_audio = self.preprocessor.apply_noise_reduction(
            self.stereo_audio,
            self.sample_rate,
            threshold=0.01,
            reduction_factor=0.7
        )
        
        # Check that output has same shape as input
        self.assertEqual(reduced_audio.shape, self.stereo_audio.shape)
        
        # Check that noise has been reduced in both channels
        self.assertLess(np.std(reduced_audio[:, 0]), np.std(self.stereo_audio[:, 0]))
        self.assertLess(np.std(reduced_audio[:, 1]), np.std(self.stereo_audio[:, 1]))
    
    def test_apply_normalization(self):
        """Test audio normalization."""
        # Create quiet audio
        quiet_audio = self.clean_audio * 0.1
        
        # Apply normalization
        normalized_audio = self.preprocessor.apply_normalization(
            quiet_audio,
            target_level=-20.0,
            headroom=3.0
        )
        
        # Check that output has same shape as input
        self.assertEqual(normalized_audio.shape, quiet_audio.shape)
        
        # Check that volume has been increased
        self.assertGreater(np.max(np.abs(normalized_audio)), np.max(np.abs(quiet_audio)))
        
        # Check that it doesn't exceed 0dB (with headroom)
        self.assertLessEqual(np.max(np.abs(normalized_audio)), 1.0)
    
    def test_apply_equalization(self):
        """Test equalization functionality."""
        # Apply equalization with speech_enhance profile
        equalized_audio = self.preprocessor.apply_equalization(
            self.clean_audio,
            self.sample_rate,
            profile="speech_enhance"
        )
        
        # Check that output has same shape as input
        self.assertEqual(equalized_audio.shape, self.clean_audio.shape)
        
        # Check that equalization has changed the audio (different frequency content)
        self.assertNotEqual(np.sum(np.abs(equalized_audio - self.clean_audio)), 0)
    
    def test_apply_equalization_invalid_profile(self):
        """Test equalization with invalid profile."""
        with self.assertRaises(AudioError):
            self.preprocessor.apply_equalization(
                self.clean_audio,
                self.sample_rate,
                profile="nonexistent_profile"
            )
    
    def test_preprocess_audio_default(self):
        """Test preprocessing with default preset."""
        # Apply preprocessing with default preset
        processed_audio = self.preprocessor.preprocess_audio(
            self.noisy_audio,
            self.sample_rate
        )
        
        # Check that output has same shape as input
        self.assertEqual(processed_audio.shape, self.noisy_audio.shape)
    
    def test_preprocess_audio_custom(self):
        """Test preprocessing with custom settings."""
        # Define custom settings
        custom_settings = {
            "noise_reduction": {"threshold": 0.02, "reduction_factor": 0.8},
            "normalization": {"target_level": -18.0, "headroom": 2.0},
            "equalization": {"profile": "voice"}
        }
        
        # Apply preprocessing with custom settings
        processed_audio = self.preprocessor.preprocess_audio(
            self.noisy_audio,
            self.sample_rate,
            preset="custom",
            custom_settings=custom_settings
        )
        
        # Check that output has same shape as input
        self.assertEqual(processed_audio.shape, self.noisy_audio.shape)
    
    def test_preprocess_audio_invalid_preset(self):
        """Test preprocessing with invalid preset."""
        with self.assertRaises(AudioError):
            self.preprocessor.preprocess_audio(
                self.noisy_audio,
                self.sample_rate,
                preset="nonexistent_preset"
            )
    
    def test_preprocess_audio_with_progress_callback(self):
        """Test preprocessing with progress callback."""
        # Create a mock progress callback
        mock_callback = MagicMock()
        
        # Apply preprocessing with progress callback
        processed_audio = self.preprocessor.preprocess_audio(
            self.noisy_audio,
            self.sample_rate,
            progress_callback=mock_callback
        )
        
        # Check that callback was called
        self.assertTrue(mock_callback.called)
    
    @patch('src.audio.file_handler.AudioFileHandler')
    def test_batch_preprocess(self, mock_file_handler_class):
        """Test batch preprocessing."""
        # Create mock file handler
        mock_file_handler = MagicMock()
        mock_file_handler_class.return_value = mock_file_handler
        
        # Mock load_audio to return our test data
        mock_file_handler.load_audio.return_value = (self.noisy_audio, self.sample_rate, {})
        
        # Create test file paths
        file_paths = [
            os.path.join(self.temp_dir, "file1.wav"),
            os.path.join(self.temp_dir, "file2.wav")
        ]
        
        # Create output directory
        output_dir = os.path.join(self.temp_dir, "output")
        
        # Apply batch preprocessing
        output_paths = self.preprocessor.batch_preprocess(
            file_paths,
            output_dir
        )
        
        # Check that output paths were returned
        self.assertEqual(len(output_paths), len(file_paths))
        
        # Check that load_audio was called for each file
        self.assertEqual(mock_file_handler.load_audio.call_count, len(file_paths))
        
        # Check that save_audio was called for each file
        self.assertEqual(mock_file_handler.save_audio.call_count, len(file_paths))

if __name__ == '__main__':
    unittest.main() 