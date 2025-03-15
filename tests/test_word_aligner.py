"""
Tests for the word alignment module.
"""

import os
import unittest
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

from src.transcription.word_aligner import WordAligner

class TestWordAligner(unittest.TestCase):
    """Test cases for the WordAligner class."""
    
    def setUp(self):
        """Set up test resources."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test audio files
        self.create_test_audio_files()
        
        # Initialize the word aligner
        self.aligner = WordAligner(device="cpu")
    
    def tearDown(self):
        """Clean up resources after tests."""
        self.temp_dir.cleanup()
    
    def create_test_audio_files(self):
        """Create test audio files."""
        # Create directory for test files
        test_files_dir = os.path.join(self.temp_dir.name, "audio")
        os.makedirs(test_files_dir, exist_ok=True)
        
        # Create a sine wave audio file
        self.test_file = os.path.join(test_files_dir, "test_audio.wav")
        self.create_sine_wave_file(self.test_file, frequency=440, duration=3)
        
        # Load the audio data for testing
        self.audio_data, self.sample_rate = sf.read(self.test_file)
    
    @staticmethod
    def create_sine_wave_file(file_path, frequency=440, duration=3, sample_rate=16000):
        """Create a sine wave audio file."""
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Save to file
        sf.write(file_path, audio, sample_rate)
    
    def test_init(self):
        """Test initialization of WordAligner."""
        self.assertIsNotNone(self.aligner)
        self.assertEqual(self.aligner.device, "cpu")
    
    def test_refine_word_timestamps(self):
        """Test refining word timestamps."""
        # Create test words
        words = [
            {'text': 'hello', 'start': 0.0, 'end': 0.5, 'confidence': 0.9},
            {'text': 'world', 'start': 0.5, 'end': 1.0, 'confidence': 0.8}
        ]
        
        # Refine timestamps
        refined_words = self.aligner.refine_word_timestamps(
            self.audio_data, self.sample_rate, words
        )
        
        # Check that the refined words have the expected structure
        self.assertEqual(len(refined_words), 2)
        self.assertIn('text', refined_words[0])
        self.assertIn('start', refined_words[0])
        self.assertIn('end', refined_words[0])
        self.assertIn('confidence', refined_words[0])
        self.assertIn('boundary_confidence', refined_words[0])
        
        # Check that the timestamps are within the expected range
        self.assertGreaterEqual(refined_words[0]['start'], 0.0)
        self.assertLessEqual(refined_words[0]['end'], 1.0)
        self.assertGreaterEqual(refined_words[1]['start'], 0.0)
        self.assertLessEqual(refined_words[1]['end'], 1.0)
        
        # Check that the words don't overlap
        self.assertLessEqual(refined_words[0]['end'], refined_words[1]['start'])
    
    def test_fix_overlapping_words(self):
        """Test fixing overlapping word timestamps."""
        # Create overlapping words
        words = [
            {'text': 'hello', 'start': 0.0, 'end': 0.6, 'confidence': 0.9},
            {'text': 'world', 'start': 0.5, 'end': 1.0, 'confidence': 0.8}
        ]
        
        # Fix overlaps
        fixed_words = self.aligner._fix_overlapping_words(words)
        
        # Check that the words don't overlap
        self.assertLessEqual(fixed_words[0]['end'], fixed_words[1]['start'])
        
        # Check that the midpoint was used
        self.assertEqual(fixed_words[0]['end'], 0.55)
        self.assertEqual(fixed_words[1]['start'], 0.55)
    
    def test_align_transcript_with_audio(self):
        """Test aligning a transcript with audio."""
        # Create a test transcript
        transcript = "hello world this is a test"
        
        # Align the transcript
        aligned_words = self.aligner.align_transcript_with_audio(
            self.audio_data, self.sample_rate, transcript
        )
        
        # Check that the aligned words have the expected structure
        self.assertEqual(len(aligned_words), 6)  # 6 words in the transcript
        self.assertIn('text', aligned_words[0])
        self.assertIn('start', aligned_words[0])
        self.assertIn('end', aligned_words[0])
        self.assertIn('confidence', aligned_words[0])
        
        # Check that the words are in order
        self.assertEqual(aligned_words[0]['text'], 'hello')
        self.assertEqual(aligned_words[1]['text'], 'world')
        self.assertEqual(aligned_words[2]['text'], 'this')
        self.assertEqual(aligned_words[3]['text'], 'is')
        self.assertEqual(aligned_words[4]['text'], 'a')
        self.assertEqual(aligned_words[5]['text'], 'test')
        
        # Check that the timestamps span the entire audio
        self.assertGreaterEqual(aligned_words[0]['start'], 0.0)
        self.assertLessEqual(aligned_words[-1]['end'], len(self.audio_data) / self.sample_rate)
        
        # Check that the words don't overlap
        for i in range(1, len(aligned_words)):
            self.assertLessEqual(aligned_words[i-1]['end'], aligned_words[i]['start'])

if __name__ == '__main__':
    unittest.main() 