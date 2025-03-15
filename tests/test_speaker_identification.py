"""
Tests for the speaker identification module.
"""

import os
import unittest
import tempfile
import numpy as np
import pickle
import torch
from pathlib import Path

from src.ml.speaker_identification import SpeakerIdentifier
from src.database.models import Speaker

class MockSpeaker:
    """Mock Speaker class for testing."""
    def __init__(self, id, name, voice_print=None):
        self.id = id
        self.name = name
        self.voice_print = voice_print

class TestSpeakerIdentification(unittest.TestCase):
    """Test cases for the SpeakerIdentifier class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources once for all tests."""
        # Skip tests if no HuggingFace token is available
        cls.hf_token = os.environ.get('HF_TOKEN')
        if not cls.hf_token:
            print("Warning: HF_TOKEN not set, some tests will be skipped")
        
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test audio files
        cls.create_test_audio_files()
        
        # Initialize the speaker identifier if token is available
        if cls.hf_token:
            cls.identifier = SpeakerIdentifier(
                auth_token=cls.hf_token,
                device="cpu",
                download_root=os.path.join(cls.temp_dir.name, "models"),
                similarity_threshold=0.7
            )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests."""
        cls.temp_dir.cleanup()
    
    @classmethod
    def create_test_audio_files(cls):
        """Create test audio files with different speakers."""
        # Create directory for test files
        test_files_dir = os.path.join(cls.temp_dir.name, "audio")
        os.makedirs(test_files_dir, exist_ok=True)
        
        # Create a sine wave audio file (speaker 1)
        cls.speaker1_file = os.path.join(test_files_dir, "speaker1.wav")
        cls.create_sine_wave_file(cls.speaker1_file, frequency=440, duration=3)
        
        # Create another sine wave audio file (speaker 2)
        cls.speaker2_file = os.path.join(test_files_dir, "speaker2.wav")
        cls.create_sine_wave_file(cls.speaker2_file, frequency=220, duration=3)
    
    @staticmethod
    def create_sine_wave_file(file_path, frequency=440, duration=3, sample_rate=16000):
        """Create a sine wave audio file."""
        import soundfile as sf
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Save to file
        sf.write(file_path, audio, sample_rate)
    
    def test_init(self):
        """Test initialization of SpeakerIdentifier."""
        if not self.hf_token:
            self.skipTest("HF_TOKEN not set")
        
        self.assertIsNotNone(self.identifier)
        self.assertEqual(self.identifier.device, "cpu")
        self.assertEqual(self.identifier.similarity_threshold, 0.7)
    
    def test_generate_voice_print(self):
        """Test generating voice prints from audio."""
        if not self.hf_token:
            self.skipTest("HF_TOKEN not set")
        
        # Generate voice print from file
        embedding = self.identifier.generate_voice_print(self.speaker1_file)
        
        # Check that the embedding is a numpy array with the expected shape
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.ndim, 1)  # Should be a 1D array
        
        # Generate voice print from numpy array
        audio_data = np.random.randn(16000)  # 1 second of random noise
        embedding = self.identifier.generate_voice_print(audio_data, sample_rate=16000)
        
        # Check that the embedding is a numpy array with the expected shape
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.ndim, 1)
    
    def test_compare_voice_prints(self):
        """Test comparing voice prints."""
        if not self.hf_token:
            self.skipTest("HF_TOKEN not set")
        
        # Generate voice prints for two different speakers
        embedding1 = self.identifier.generate_voice_print(self.speaker1_file)
        embedding2 = self.identifier.generate_voice_print(self.speaker2_file)
        
        # Compare the same embedding with itself (should be 1.0)
        similarity = self.identifier.compare_voice_prints(embedding1, embedding1)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Compare different embeddings (should be less than 1.0)
        similarity = self.identifier.compare_voice_prints(embedding1, embedding2)
        self.assertLess(similarity, 1.0)
    
    def test_find_matching_speaker(self):
        """Test finding matching speakers."""
        if not self.hf_token:
            self.skipTest("HF_TOKEN not set")
        
        # Generate voice prints
        embedding1 = self.identifier.generate_voice_print(self.speaker1_file)
        embedding2 = self.identifier.generate_voice_print(self.speaker2_file)
        
        # Create mock speakers
        speakers = [
            MockSpeaker(1, "Speaker 1", pickle.dumps(embedding1)),
            MockSpeaker(2, "Speaker 2", pickle.dumps(embedding2))
        ]
        
        # Find matching speaker for embedding1
        match, score = self.identifier.find_matching_speaker(embedding1, speakers)
        self.assertIsNotNone(match)
        self.assertEqual(match.id, 1)
        self.assertAlmostEqual(score, 1.0, places=5)
        
        # Find matching speaker for embedding2
        match, score = self.identifier.find_matching_speaker(embedding2, speakers)
        self.assertIsNotNone(match)
        self.assertEqual(match.id, 2)
        self.assertAlmostEqual(score, 1.0, places=5)
        
        # Test with empty speaker list
        match, score = self.identifier.find_matching_speaker(embedding1, [])
        self.assertIsNone(match)
        self.assertEqual(score, 0.0)
    
    def test_update_speaker_voice_print(self):
        """Test updating speaker voice prints."""
        if not self.hf_token:
            self.skipTest("HF_TOKEN not set")
        
        # Generate voice prints
        embedding1 = self.identifier.generate_voice_print(self.speaker1_file)
        embedding2 = self.identifier.generate_voice_print(self.speaker2_file)
        
        # Create a mock speaker with no voice print
        speaker = MockSpeaker(1, "Speaker 1")
        
        # Update with initial voice print
        updated = self.identifier.update_speaker_voice_print(speaker, embedding1)
        self.assertIsInstance(updated, np.ndarray)
        np.testing.assert_array_almost_equal(updated, embedding1)
        
        # Create a mock speaker with existing voice print
        speaker = MockSpeaker(1, "Speaker 1", pickle.dumps(embedding1))
        
        # Update with new voice print
        updated = self.identifier.update_speaker_voice_print(speaker, embedding2, learning_rate=0.5)
        self.assertIsInstance(updated, np.ndarray)
        
        # Check that the updated embedding is between the two original embeddings
        similarity1 = self.identifier.compare_voice_prints(updated, embedding1)
        similarity2 = self.identifier.compare_voice_prints(updated, embedding2)
        self.assertGreater(similarity1, 0.0)
        self.assertGreater(similarity2, 0.0)

if __name__ == '__main__':
    unittest.main() 