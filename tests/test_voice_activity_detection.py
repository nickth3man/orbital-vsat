"""
Tests for the Voice Activity Detection module.
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
import soundfile as sf
import torch
from unittest.mock import patch, MagicMock

from src.ml.voice_activity_detection import VoiceActivityDetector
from src.ml.error_handling import ModelLoadError, InferenceError

class TestVoiceActivityDetector(unittest.TestCase):
    """Test cases for the VoiceActivityDetector class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test audio data
        self.sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        # Create a sine wave with silence gaps
        # 0-0.5s: silence
        # 0.5-1.5s: speech (sine wave)
        # 1.5-2.0s: silence
        # 2.0-2.5s: speech (sine wave)
        # 2.5-3.0s: silence
        
        # Initialize with zeros (silence)
        self.audio_data = np.zeros_like(t)
        
        # Add sine wave for speech segments
        speech_mask = ((t >= 0.5) & (t < 1.5)) | ((t >= 2.0) & (t < 2.5))
        self.audio_data[speech_mask] = 0.5 * np.sin(2 * np.pi * 440 * t[speech_mask])
        
        # Add some noise
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 0.01, self.audio_data.shape)
        self.audio_data_noisy = self.audio_data + noise
        
        # Create stereo version
        self.audio_data_stereo = np.vstack((self.audio_data_noisy, self.audio_data_noisy)).T
        
        # Save test audio files
        self.test_mono_file = os.path.join(self.temp_dir, "test_mono.wav")
        self.test_stereo_file = os.path.join(self.temp_dir, "test_stereo.wav")
        
        sf.write(self.test_mono_file, self.audio_data_noisy, self.sample_rate)
        sf.write(self.test_stereo_file, self.audio_data_stereo, self.sample_rate)
        
        # Create a VAD instance with energy-based approach (no ML model)
        self.vad = VoiceActivityDetector(settings={"use_model": False})
        
        # Create ground truth segments
        self.expected_segments = [
            {"start": 0.5, "end": 1.5, "duration": 1.0},
            {"start": 2.0, "end": 2.5, "duration": 0.5}
        ]
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of VoiceActivityDetector."""
        # Test with default settings
        vad = VoiceActivityDetector(settings={"use_model": False})
        self.assertEqual(vad.device, "cpu")
        self.assertEqual(vad.settings["energy_threshold"], 0.05)
        self.assertEqual(vad.settings["use_model"], False)
        
        # Test with custom settings
        custom_settings = {
            "energy_threshold": 0.1,
            "min_speech_duration_ms": 300,
            "use_model": False
        }
        vad = VoiceActivityDetector(settings=custom_settings)
        self.assertEqual(vad.settings["energy_threshold"], 0.1)
        self.assertEqual(vad.settings["min_speech_duration_ms"], 300)
    
    @patch('src.ml.voice_activity_detection.Pipeline')
    def test_init_with_model(self, mock_pipeline):
        """Test initialization with ML model."""
        # Mock the Pipeline.from_pretrained method
        mock_pipeline_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance
        
        # Initialize with model
        vad = VoiceActivityDetector(auth_token="test_token", settings={"use_model": True})
        
        # Check if model was loaded
        mock_pipeline.from_pretrained.assert_called_once_with(
            "pyannote/voice-activity-detection", 
            use_auth_token="test_token"
        )
        mock_pipeline_instance.to.assert_called_once()
    
    @patch('src.ml.voice_activity_detection.Pipeline')
    def test_init_model_error(self, mock_pipeline):
        """Test handling of model loading errors."""
        # Make the model loading fail
        mock_pipeline.from_pretrained.side_effect = Exception("Model loading failed")
        
        # Check if ModelLoadError is raised
        with self.assertRaises(ModelLoadError):
            VoiceActivityDetector(settings={"use_model": True})
    
    def test_sensitivity_presets(self):
        """Test sensitivity presets."""
        # Get available presets
        presets = self.vad.get_available_presets()
        self.assertIn("high", presets)
        self.assertIn("medium", presets)
        self.assertIn("low", presets)
        
        # Apply a preset
        self.vad.apply_sensitivity_preset("high")
        self.assertEqual(self.vad.settings["energy_threshold"], 0.03)
        
        # Test invalid preset
        with self.assertRaises(ValueError):
            self.vad.apply_sensitivity_preset("invalid_preset")
    
    def test_detect_speech_energy_based(self):
        """Test energy-based speech detection."""
        # Detect speech in mono audio
        segments = self.vad.detect_speech(self.audio_data_noisy, self.sample_rate)
        
        # Check if segments were detected
        self.assertGreaterEqual(len(segments), 1)
        
        # Check segment properties
        for segment in segments:
            self.assertIn("start", segment)
            self.assertIn("end", segment)
            self.assertIn("duration", segment)
            self.assertIn("confidence", segment)
            
            # Check if confidence is between 0 and 1
            self.assertGreaterEqual(segment["confidence"], 0.0)
            self.assertLessEqual(segment["confidence"], 1.0)
            
            # Check if duration is positive
            self.assertGreater(segment["duration"], 0.0)
            
            # Check if end is after start
            self.assertGreater(segment["end"], segment["start"])
    
    def test_detect_speech_from_file(self):
        """Test speech detection from audio file."""
        # Detect speech from mono file
        segments = self.vad.detect_speech(self.test_mono_file)
        
        # Check if segments were detected
        self.assertGreaterEqual(len(segments), 1)
        
        # Detect speech from stereo file
        segments_stereo = self.vad.detect_speech(self.test_stereo_file)
        
        # Check if segments were detected
        self.assertGreaterEqual(len(segments_stereo), 1)
    
    def test_detect_speech_with_progress_callback(self):
        """Test speech detection with progress callback."""
        # Create a mock progress callback
        progress_values = []
        def progress_callback(progress):
            progress_values.append(progress)
        
        # Detect speech with progress callback
        segments = self.vad.detect_speech(
            self.audio_data_noisy, 
            self.sample_rate,
            progress_callback=progress_callback
        )
        
        # Check if progress callback was called
        self.assertGreaterEqual(len(progress_values), 3)  # At least 25%, 50%, 75%, 100%
        self.assertIn(0.25, progress_values)
        self.assertIn(0.5, progress_values)
        self.assertIn(0.75, progress_values)
        self.assertIn(1.0, progress_values)
    
    def test_merge_overlapping_segments(self):
        """Test merging of overlapping segments."""
        # Create overlapping segments
        segments = [
            {"start": 1.0, "end": 2.0, "confidence": 0.8, "duration": 1.0},
            {"start": 1.5, "end": 2.5, "confidence": 0.9, "duration": 1.0},
            {"start": 3.0, "end": 4.0, "confidence": 0.7, "duration": 1.0}
        ]
        
        # Merge segments
        merged = self.vad._merge_overlapping_segments(segments)
        
        # Check result
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["start"], 1.0)
        self.assertEqual(merged[0]["end"], 2.5)
        self.assertEqual(merged[0]["confidence"], 0.9)  # Max confidence
        self.assertEqual(merged[1]["start"], 3.0)
        self.assertEqual(merged[1]["end"], 4.0)
    
    def test_merge_close_segments(self):
        """Test merging of close segments."""
        # Create close segments
        segments = [
            {"start": 1.0, "end": 2.0, "confidence": 0.8, "duration": 1.0},
            {"start": 2.1, "end": 3.0, "confidence": 0.9, "duration": 0.9},  # Gap of 0.1s
            {"start": 4.0, "end": 5.0, "confidence": 0.7, "duration": 1.0}   # Gap of 1.0s
        ]
        
        # Merge segments with max gap of 0.2s
        merged = self.vad._merge_close_segments(segments, 0.2)
        
        # Check result
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["start"], 1.0)
        self.assertEqual(merged[0]["end"], 3.0)
        self.assertEqual(merged[0]["confidence"], 0.85)  # Average confidence
        self.assertEqual(merged[1]["start"], 4.0)
        self.assertEqual(merged[1]["end"], 5.0)
    
    def test_get_speech_mask(self):
        """Test generation of speech mask."""
        # Create segments
        segments = [
            {"start": 1.0, "end": 2.0},
            {"start": 3.0, "end": 4.0}
        ]
        
        # Generate mask
        audio_length = 5 * self.sample_rate  # 5 seconds
        audio_data = np.zeros(audio_length)
        mask = self.vad.get_speech_mask(audio_data, self.sample_rate, segments)
        
        # Check mask properties
        self.assertEqual(len(mask), len(audio_data))
        
        # Check if mask is 1 in speech regions
        self.assertEqual(mask[int(1.5 * self.sample_rate)], 1)
        self.assertEqual(mask[int(3.5 * self.sample_rate)], 1)
        
        # Check if mask is 0 in non-speech regions
        self.assertEqual(mask[int(0.5 * self.sample_rate)], 0)
        self.assertEqual(mask[int(2.5 * self.sample_rate)], 0)
        self.assertEqual(mask[int(4.5 * self.sample_rate)], 0)
    
    def test_calculate_speech_statistics(self):
        """Test calculation of speech statistics."""
        # Create segments
        segments = [
            {"start": 1.0, "end": 2.0, "duration": 1.0, "confidence": 0.8},
            {"start": 3.0, "end": 4.0, "duration": 1.0, "confidence": 0.9}
        ]
        
        # Calculate statistics
        total_duration = 5.0  # 5 seconds
        stats = self.vad.calculate_speech_statistics(segments, total_duration)
        
        # Check statistics
        self.assertEqual(stats["speech_count"], 2)
        self.assertEqual(stats["total_speech_duration"], 2.0)
        self.assertEqual(stats["speech_percentage"], 40.0)
        self.assertEqual(stats["avg_speech_duration"], 1.0)
        self.assertEqual(stats["max_speech_duration"], 1.0)
        self.assertEqual(stats["min_speech_duration"], 1.0)
        self.assertEqual(stats["avg_confidence"], 0.85)
        
        # Test with empty segments
        empty_stats = self.vad.calculate_speech_statistics([], total_duration)
        self.assertEqual(empty_stats["speech_count"], 0)
        self.assertEqual(empty_stats["speech_percentage"], 0.0)
    
    @patch('matplotlib.figure.Figure')
    def test_visualize_speech_segments(self, mock_figure):
        """Test visualization of speech segments."""
        # Create segments
        segments = [
            {"start": 1.0, "end": 2.0, "confidence": 0.8},
            {"start": 3.0, "end": 4.0, "confidence": 0.9}
        ]
        
        # Mock the figure and canvas
        mock_canvas = MagicMock()
        mock_figure.return_value.add_subplot.return_value = MagicMock()
        
        # Test visualization
        with patch('src.ml.voice_activity_detection.FigureCanvas', return_value=mock_canvas):
            with patch('io.BytesIO', return_value=MagicMock()):
                with patch('matplotlib.image.imread', return_value=np.zeros((100, 100, 3))):
                    img = self.vad.visualize_speech_segments(
                        self.audio_data_noisy, 
                        self.sample_rate,
                        segments
                    )
                    self.assertIsInstance(img, np.ndarray)
    
    def test_error_handling(self):
        """Test error handling in speech detection."""
        # Test with invalid audio type
        with self.assertRaises(TypeError):
            self.vad.detect_speech(123)  # Not a valid audio input
        
        # Test with numpy array but no sample rate
        with self.assertRaises(ValueError):
            self.vad.detect_speech(self.audio_data_noisy)  # Missing sample rate
        
        # Test with non-existent file
        with self.assertRaises(InferenceError):
            self.vad.detect_speech("non_existent_file.wav")

if __name__ == '__main__':
    unittest.main() 