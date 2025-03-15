"""
Tests for audio player functionality.

This module contains tests to verify that the audio player works correctly.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

from src.audio.audio_player import AudioPlayer


class TestAudioPlayer(unittest.TestCase):
    """Test case for audio player functionality."""
    
    def setUp(self):
        """Set up the test."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test audio file
        self.audio_file = os.path.join(self.temp_dir.name, "test_audio.wav")
        self.create_test_audio_file()
        
        # Create audio player
        self.audio_player = AudioPlayer()
    
    def tearDown(self):
        """Clean up after the test."""
        # Stop playback
        self.audio_player.stop()
        
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
    
    def test_load_file(self):
        """Test loading an audio file."""
        # Load the file
        result = self.audio_player.load_file(self.audio_file)
        
        # Check that the file was loaded successfully
        self.assertTrue(result)
        
        # Check that the duration is correct
        self.assertAlmostEqual(self.audio_player.get_duration(), 5.0, delta=0.1)
    
    def test_play_pause_stop(self):
        """Test play, pause, and stop functionality."""
        # Load the file
        self.audio_player.load_file(self.audio_file)
        
        # Test play
        self.audio_player.play(self.audio_file)
        self.assertTrue(self.audio_player.is_playing())
        
        # Test pause
        self.audio_player.pause()
        self.assertFalse(self.audio_player.is_playing())
        
        # Test resume
        self.audio_player.play(self.audio_file)
        self.assertTrue(self.audio_player.is_playing())
        
        # Test stop
        self.audio_player.stop()
        self.assertFalse(self.audio_player.is_playing())
        
        # Check that position is reset
        self.assertAlmostEqual(self.audio_player.get_position(), 0.0, delta=0.1)
    
    def test_set_get_position(self):
        """Test setting and getting the playback position."""
        # Load the file
        self.audio_player.load_file(self.audio_file)
        
        # Set position
        self.audio_player.set_position(2.5)
        
        # Check that position was set correctly
        self.assertAlmostEqual(self.audio_player.get_position(), 2.5, delta=0.1)
    
    def test_set_get_volume(self):
        """Test setting and getting the volume."""
        # Set volume
        self.audio_player.set_volume(0.75)
        
        # Check that volume was set correctly
        self.assertAlmostEqual(self.audio_player.get_volume(), 0.75, delta=0.01)
    
    def test_play_segment(self):
        """Test playing a segment of the audio file."""
        # Load the file
        self.audio_player.load_file(self.audio_file)
        
        # Play segment
        self.audio_player.play(self.audio_file, 1.0, 3.0)
        
        # Check that playback started
        self.assertTrue(self.audio_player.is_playing())
        
        # Check that position is at the start of the segment
        self.assertAlmostEqual(self.audio_player.get_position(), 1.0, delta=0.1)
    
    def test_position_changed_signal(self):
        """Test that the position_changed signal is emitted."""
        # Create a flag to track signal emission
        position_changed = False
        position_value = 0.0
        
        # Connect to the signal
        def on_position_changed(position):
            nonlocal position_changed, position_value
            position_changed = True
            position_value = position
        
        self.audio_player.position_changed.connect(on_position_changed)
        
        # Load the file
        self.audio_player.load_file(self.audio_file)
        
        # Set position
        self.audio_player.set_position(2.5)
        
        # Check that the signal was emitted
        self.assertTrue(position_changed)
        self.assertAlmostEqual(position_value, 2.5, delta=0.1)
    
    def test_playback_state_changed_signal(self):
        """Test that the playback_state_changed signal is emitted."""
        # Create a flag to track signal emission
        state_changed = False
        is_playing_value = False
        
        # Connect to the signal
        def on_state_changed(is_playing):
            nonlocal state_changed, is_playing_value
            state_changed = True
            is_playing_value = is_playing
        
        self.audio_player.playback_state_changed.connect(on_state_changed)
        
        # Load the file
        self.audio_player.load_file(self.audio_file)
        
        # Play
        self.audio_player.play(self.audio_file)
        
        # Check that the signal was emitted
        self.assertTrue(state_changed)
        self.assertTrue(is_playing_value)
        
        # Reset flag
        state_changed = False
        
        # Pause
        self.audio_player.pause()
        
        # Check that the signal was emitted
        self.assertTrue(state_changed)
        self.assertFalse(is_playing_value)


if __name__ == "__main__":
    unittest.main() 