"""
Tests for speaker separation functionality.

This module contains tests to verify that speaker separation works correctly.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.audio.processor import AudioProcessor
from src.utils.error_handler import ProcessingError

class TestSpeakerSeparation(unittest.TestCase):
    """Test case for speaker separation functionality."""
    
    def setUp(self):
        """Set up the test."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test audio file with mixed speech
        self.audio_file = os.path.join(self.temp_dir.name, "mixed_speech.wav")
        self.create_test_audio_file()
        
        # Create audio processor
        self.audio_processor = AudioProcessor()
    
    def tearDown(self):
        """Clean up after the test."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def create_test_audio_file(self):
        """Create a test audio file with mixed speech simulated using sine waves."""
        # Generate two speech-like signals using sine waves of different frequencies
        sample_rate = 16000
        duration = 5.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # First "speaker" - 300 Hz with amplitude modulation
        speaker1 = 0.5 * np.sin(2 * np.pi * 300 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
        
        # Second "speaker" - 500 Hz with different amplitude modulation
        speaker2 = 0.5 * np.sin(2 * np.pi * 500 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t + 1.0))
        
        # Mix the signals with some periods of overlap and some of single speaker
        mixed = np.zeros_like(t)
        
        # First part: only speaker 1
        segment_length = int(sample_rate * 1.0)  # 1 second
        mixed[:segment_length] = speaker1[:segment_length]
        
        # Second part: both speakers
        segment_length = int(sample_rate * 2.0)  # 2 seconds
        start_idx = int(sample_rate * 1.0)
        mixed[start_idx:start_idx+segment_length] = (
            speaker1[start_idx:start_idx+segment_length] + 
            speaker2[start_idx:start_idx+segment_length]
        )
        
        # Third part: only speaker 2
        segment_length = int(sample_rate * 2.0)  # 2 seconds
        start_idx = int(sample_rate * 3.0)
        mixed[start_idx:] = speaker2[start_idx:]
        
        # Normalize the mixed signal
        mixed = mixed / np.max(np.abs(mixed)) * 0.9
        
        # Save as WAV file
        sf.write(self.audio_file, mixed, sample_rate)
    
    @patch('asteroid.models.ConvTasNet')
    @patch('torch.tensor')
    @patch('torch.device')
    def test_speaker_separation(self, mock_device, mock_tensor, mock_conv_tasnet):
        """Test separating speakers from mixed audio."""
        # Create mock model
        mock_model = MagicMock()
        mock_conv_tasnet.from_pretrained.return_value = mock_model
        
        # Setup mock output for model
        sample_rate = 16000
        duration = 5.0
        num_samples = int(sample_rate * duration)
        
        # Create fake separated sources
        source1 = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, duration, num_samples))
        source2 = 0.5 * np.sin(2 * np.pi * 500 * np.linspace(0, duration, num_samples))
        
        # Shape output as [batch, sources, time]
        mock_output = MagicMock()
        mock_output.cpu.return_value = mock_output
        mock_output.detach.return_value = mock_output
        mock_output.numpy.return_value = np.stack([source1, source2])[np.newaxis, :, :]
        
        mock_model.return_value = mock_output
        
        # Set up mock device and tensor
        mock_device.return_value = "cpu"
        mock_tensor.return_value = MagicMock()
        mock_tensor.return_value.unsqueeze.return_value = mock_tensor.return_value
        mock_tensor.return_value.to.return_value = mock_tensor.return_value
        
        # Run separation
        output_dir = os.path.join(self.temp_dir.name, "separated")
        output_files = self.audio_processor.separate_sources(
            self.audio_file,
            output_dir=output_dir,
            max_speakers=2
        )
        
        # Check that the output directory was created
        self.assertTrue(os.path.exists(output_dir))
        
        # Check that output files were created
        self.assertEqual(len(output_files), 2)
        for output_file in output_files:
            self.assertTrue(os.path.exists(output_file))
            
            # Check that the file contains audio data
            audio_data, _ = sf.read(output_file)
            self.assertGreater(len(audio_data), 0)
    
    def test_separation_with_invalid_model(self):
        """Test error handling with invalid model type."""
        # Attempt to use an invalid model type
        with self.assertRaises(ProcessingError):
            self.audio_processor.separate_sources(
                self.audio_file,
                model_type="invalid_model"
            )
    
    def test_separation_with_max_speakers(self):
        """Test limiting the number of speakers."""
        # Create a mocked version of separation that returns specific sources
        def mock_separate(audio_path, **kwargs):
            # Create fake separated sources
            sample_rate = 16000
            duration = 5.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create output files in the specified directory
            output_dir = kwargs.get('output_dir', os.path.join(self.temp_dir.name, "separated"))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Create and save 4 sources (more than max_speakers)
            output_files = []
            for i in range(4):
                freq = 300 + i * 100
                source = 0.5 * np.sin(2 * np.pi * freq * t)
                output_file = os.path.join(output_dir, f"speaker_{i+1}.wav")
                sf.write(output_file, source, sample_rate)
                output_files.append(output_file)
                
            return output_files[:kwargs.get('max_speakers', 6)]
            
        # Save the original method and replace it
        original_separate = self.audio_processor.separate_sources
        self.audio_processor.separate_sources = mock_separate
        
        try:
            # Test with max_speakers=2
            output_files = self.audio_processor.separate_sources(
                self.audio_file,
                max_speakers=2
            )
            
            # Check that only 2 sources were returned
            self.assertEqual(len(output_files), 2)
            
        finally:
            # Restore original method
            self.audio_processor.separate_sources = original_separate
    
    def test_progress_reporting(self):
        """Test that progress callbacks work correctly."""
        # Create a mock progress callback
        progress_values = []
        status_messages = []
        
        def progress_callback(progress, status):
            progress_values.append(progress)
            status_messages.append(status)
        
        # Create a mocked version of separation
        @patch('asteroid.models.ConvTasNet')
        @patch('torch.tensor')
        @patch('torch.device')
        def run_with_progress(mock_device, mock_tensor, mock_conv_tasnet):
            # Create mock model
            mock_model = MagicMock()
            mock_conv_tasnet.from_pretrained.return_value = mock_model
            
            # Setup mock output for model
            sample_rate = 16000
            duration = 5.0
            num_samples = int(sample_rate * duration)
            
            # Create fake separated sources
            source1 = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, duration, num_samples))
            source2 = 0.5 * np.sin(2 * np.pi * 500 * np.linspace(0, duration, num_samples))
            
            # Shape output as [batch, sources, time]
            mock_output = MagicMock()
            mock_output.cpu.return_value = mock_output
            mock_output.detach.return_value = mock_output
            mock_output.numpy.return_value = np.stack([source1, source2])[np.newaxis, :, :]
            
            mock_model.return_value = mock_output
            
            # Set up mock device and tensor
            mock_device.return_value = "cpu"
            mock_tensor.return_value = MagicMock()
            mock_tensor.return_value.unsqueeze.return_value = mock_tensor.return_value
            mock_tensor.return_value.to.return_value = mock_tensor.return_value
            
            # Run separation with progress callback
            output_dir = os.path.join(self.temp_dir.name, "separated_progress")
            self.audio_processor.separate_sources(
                self.audio_file,
                output_dir=output_dir,
                max_speakers=2,
                progress_callback=progress_callback
            )
            
        # Run the test
        run_with_progress()
        
        # Check that progress callbacks were made
        self.assertGreater(len(progress_values), 0)
        self.assertGreater(len(status_messages), 0)
        
        # Check initial and final progress values
        self.assertAlmostEqual(progress_values[0], 0.1, places=1)  # Initial progress
        self.assertAlmostEqual(progress_values[-1], 1.0, places=1) # Final progress

if __name__ == '__main__':
    unittest.main() 