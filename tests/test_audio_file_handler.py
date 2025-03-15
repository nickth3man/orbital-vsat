"""
Unit tests for the AudioFileHandler class.
"""

import os
import unittest
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.audio.file_handler import AudioFileHandler
from src.utils.error_handler import AudioError, FileError, ErrorSeverity

class TestAudioFileHandler(unittest.TestCase):
    """Test case for the AudioFileHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test audio files
        self.sample_rate = 44100
        self.duration = 2.0  # seconds
        self.channels = 1
        
        # Create mono audio data (sine wave)
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        self.audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Create paths for test files
        self.valid_wav_path = os.path.join(self.temp_dir.name, "test_valid.wav")
        self.valid_flac_path = os.path.join(self.temp_dir.name, "test_valid.flac")
        self.nonexistent_path = os.path.join(self.temp_dir.name, "nonexistent.wav")
        self.invalid_format_path = os.path.join(self.temp_dir.name, "test_invalid.xyz")
        self.corrupt_wav_path = os.path.join(self.temp_dir.name, "test_corrupt.wav")
        
        # Create valid WAV file
        sf.write(self.valid_wav_path, self.audio_data, self.sample_rate)
        
        # Create valid FLAC file
        sf.write(self.valid_flac_path, self.audio_data, self.sample_rate)
        
        # Create corrupt WAV file (just write some random bytes)
        with open(self.corrupt_wav_path, 'wb') as f:
            f.write(b'This is not a valid WAV file')
        
        # Create invalid format file
        with open(self.invalid_format_path, 'wb') as f:
            f.write(b'This is not a valid audio file')
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_is_supported_format(self):
        """Test is_supported_format method."""
        # Test supported formats
        self.assertTrue(AudioFileHandler.is_supported_format('file.wav'))
        self.assertTrue(AudioFileHandler.is_supported_format('file.mp3'))
        self.assertTrue(AudioFileHandler.is_supported_format('file.flac'))
        
        # Test unsupported formats
        self.assertFalse(AudioFileHandler.is_supported_format('file.ogg'))
        self.assertFalse(AudioFileHandler.is_supported_format('file.xyz'))
        self.assertFalse(AudioFileHandler.is_supported_format('file'))
    
    def test_load_valid_audio(self):
        """Test loading a valid audio file."""
        # Load valid WAV file
        audio_data, sample_rate, metadata = AudioFileHandler.load_audio(self.valid_wav_path)
        
        # Check that audio data was loaded correctly
        self.assertEqual(sample_rate, self.sample_rate)
        self.assertAlmostEqual(len(audio_data) / sample_rate, self.duration)
        
        # Check metadata
        self.assertEqual(metadata['filename'], os.path.basename(self.valid_wav_path))
        self.assertEqual(metadata['format'], 'wav')
        self.assertEqual(metadata['channels'], self.channels)
    
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        # Should raise FileError
        with self.assertRaises(FileError) as context:
            AudioFileHandler.load_audio(self.nonexistent_path)
        
        # Check error message and details
        self.assertIn("File not found", str(context.exception))
        self.assertEqual(context.exception.severity, ErrorSeverity.ERROR)
        self.assertEqual(context.exception.details['file_path'], self.nonexistent_path)
    
    def test_load_unsupported_format(self):
        """Test loading a file with unsupported format."""
        # Should raise AudioError
        with self.assertRaises(AudioError) as context:
            AudioFileHandler.load_audio(self.invalid_format_path)
        
        # Check error message and details
        self.assertIn("Unsupported file format", str(context.exception))
        self.assertEqual(context.exception.severity, ErrorSeverity.ERROR)
        self.assertEqual(context.exception.details['file_path'], self.invalid_format_path)
        self.assertEqual(context.exception.details['extension'], '.xyz')
    
    def test_load_corrupt_file(self):
        """Test loading a corrupt audio file."""
        # Should raise AudioError
        with self.assertRaises(AudioError) as context:
            AudioFileHandler.load_audio(self.corrupt_wav_path)
        
        # Check error message and details
        self.assertIn("Error loading audio file", str(context.exception))
        self.assertEqual(context.exception.severity, ErrorSeverity.ERROR)
        self.assertEqual(context.exception.details['file_path'], self.corrupt_wav_path)
    
    def test_save_audio(self):
        """Test saving audio data to a file."""
        # Create output path
        output_path = os.path.join(self.temp_dir.name, "output.wav")
        
        # Save audio data
        result = AudioFileHandler.save_audio(output_path, self.audio_data, self.sample_rate)
        
        # Check that file was saved successfully
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))
        
        # Load saved file to verify contents
        saved_data, saved_rate = sf.read(output_path)
        self.assertEqual(saved_rate, self.sample_rate)
        self.assertEqual(len(saved_data), len(self.audio_data))
    
    def test_save_audio_empty_data(self):
        """Test saving empty audio data."""
        # Create output path
        output_path = os.path.join(self.temp_dir.name, "empty.wav")
        
        # Should raise AudioError
        with self.assertRaises(AudioError) as context:
            AudioFileHandler.save_audio(output_path, np.array([]), self.sample_rate)
        
        # Check error message and details
        self.assertIn("No audio data to save", str(context.exception))
        self.assertEqual(context.exception.severity, ErrorSeverity.ERROR)
        self.assertEqual(context.exception.details['file_path'], output_path)
    
    def test_save_audio_unsupported_format(self):
        """Test saving audio to an unsupported format."""
        # Create output path with unsupported extension
        output_path = os.path.join(self.temp_dir.name, "output.xyz")
        
        # Should raise AudioError
        with self.assertRaises(AudioError) as context:
            AudioFileHandler.save_audio(output_path, self.audio_data, self.sample_rate)
        
        # Check error message and details
        self.assertIn("Unsupported output format", str(context.exception))
        self.assertEqual(context.exception.severity, ErrorSeverity.ERROR)
        self.assertEqual(context.exception.details['file_path'], output_path)
        self.assertEqual(context.exception.details['extension'], '.xyz')
    
    @patch('os.makedirs')
    def test_save_audio_permission_error(self, mock_makedirs):
        """Test saving audio when permission is denied."""
        # Mock os.makedirs to raise PermissionError
        mock_makedirs.side_effect = PermissionError("Permission denied")
        
        # Create output path
        output_path = os.path.join(self.temp_dir.name, "protected", "output.wav")
        
        # Should raise FileError
        with self.assertRaises(FileError) as context:
            AudioFileHandler.save_audio(output_path, self.audio_data, self.sample_rate)
        
        # Check error message and details
        self.assertIn("Permission denied", str(context.exception))
        self.assertEqual(context.exception.severity, ErrorSeverity.ERROR)
        self.assertEqual(context.exception.details['file_path'], output_path)
    
    def test_get_audio_info(self):
        """Test getting information about an audio file."""
        # Get info about valid WAV file
        info = AudioFileHandler.get_audio_info(self.valid_wav_path)
        
        # Check info
        self.assertEqual(info['filename'], os.path.basename(self.valid_wav_path))
        self.assertEqual(info['format'], 'wav')
        self.assertEqual(info['sample_rate'], self.sample_rate)
        self.assertEqual(info['channels'], self.channels)
        self.assertAlmostEqual(info['duration'], self.duration, places=2)
    
    def test_get_audio_info_nonexistent_file(self):
        """Test getting info about a nonexistent file."""
        # Should raise FileError
        with self.assertRaises(FileError) as context:
            AudioFileHandler.get_audio_info(self.nonexistent_path)
        
        # Check error message and details
        self.assertIn("File not found", str(context.exception))
        self.assertEqual(context.exception.severity, ErrorSeverity.ERROR)
        self.assertEqual(context.exception.details['file_path'], self.nonexistent_path)
    
    def test_split_audio(self):
        """Test splitting an audio file into segments."""
        # Create longer audio data for splitting
        t = np.linspace(0, 10.0, int(self.sample_rate * 10.0), endpoint=False)
        long_audio = np.sin(2 * np.pi * 440 * t)  # 10-second 440 Hz sine wave
        
        # Create long audio file
        long_audio_path = os.path.join(self.temp_dir.name, "long_audio.wav")
        sf.write(long_audio_path, long_audio, self.sample_rate)
        
        # Create output directory
        output_dir = os.path.join(self.temp_dir.name, "segments")
        
        # Split audio into 2-second segments with 0.5-second overlap
        segment_paths = AudioFileHandler.split_audio(
            long_audio_path, output_dir, segment_length=2.0, overlap=0.5
        )
        
        # Should create 6 segments: 0-2s, 1.5-3.5s, 3-5s, 4.5-6.5s, 6-8s, 7.5-9.5s
        self.assertEqual(len(segment_paths), 6)
        
        # Check that segment files exist
        for path in segment_paths:
            self.assertTrue(os.path.exists(path))
            
            # Load segment and check duration
            segment_data, _ = sf.read(path)
            segment_duration = len(segment_data) / self.sample_rate
            self.assertAlmostEqual(segment_duration, 2.0, places=1)
    
    def test_split_audio_invalid_segment_length(self):
        """Test splitting audio with invalid segment length."""
        # Should raise AudioError
        with self.assertRaises(AudioError) as context:
            AudioFileHandler.split_audio(
                self.valid_wav_path,
                os.path.join(self.temp_dir.name, "segments"),
                segment_length=0.0
            )
        
        # Check error message and details
        self.assertIn("Invalid segment length", str(context.exception))
        self.assertEqual(context.exception.severity, ErrorSeverity.ERROR)
    
    def test_split_audio_invalid_overlap(self):
        """Test splitting audio with invalid overlap."""
        # Should raise AudioError
        with self.assertRaises(AudioError) as context:
            AudioFileHandler.split_audio(
                self.valid_wav_path,
                os.path.join(self.temp_dir.name, "segments"),
                segment_length=1.0,
                overlap=1.5
            )
        
        # Check error message and details
        self.assertIn("Overlap is greater than or equal to segment length", str(context.exception))
        self.assertEqual(context.exception.severity, ErrorSeverity.ERROR)


if __name__ == "__main__":
    unittest.main() 