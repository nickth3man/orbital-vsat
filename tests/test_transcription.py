"""
Tests for the transcription module.

This module contains tests for the WhisperTranscriber and WordAligner classes.
"""

import os
import unittest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

# Adjust imports to work with the project structure
from src.transcription.whisper_transcriber import WhisperTranscriber
from src.transcription.word_aligner import WordAligner


class TestWhisperTranscriber(unittest.TestCase):
    """Test case for WhisperTranscriber."""

    @patch('src.transcription.whisper_transcriber.WhisperModel')
    def setUp(self, mock_whisper_model):
        """Set up test environment with mocked dependencies."""
        # Mock WhisperModel
        self.mock_model = mock_whisper_model.return_value
        
        # Setup mock transcribe method
        self.mock_model.transcribe.return_value = (
            [
                MagicMock(
                    start=0.0,
                    end=5.0,
                    text="This is a test sentence.",
                    words=[
                        MagicMock(start=0.0, end=0.5, word="This"),
                        MagicMock(start=0.6, end=0.9, word="is"),
                        MagicMock(start=1.0, end=1.2, word="a"),
                        MagicMock(start=1.3, end=2.0, word="test"),
                        MagicMock(start=2.1, end=3.0, word="sentence.")
                    ]
                )
            ],
            MagicMock(language="en", language_probability=0.99)
        )
        
        # Initialize the transcriber with the mocked model
        self.transcriber = WhisperTranscriber(
            model_size="tiny",
            device="cpu",
            use_word_aligner=False
        )
        
        # Create a temporary audio file for testing
        self.temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.temp_audio.close()
        
        # Create sample audio data
        self.sample_rate = 16000
        self.audio_data = np.zeros(self.sample_rate * 5, dtype=np.float32)  # 5 seconds of silence
        
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_audio.name)
    
    def test_init(self):
        """Test WhisperTranscriber initialization."""
        self.assertEqual(self.transcriber.model_size, "tiny")
        self.assertEqual(self.transcriber.device, "cpu")
        self.assertEqual(self.transcriber.compute_type, "float32")
        self.assertFalse(self.transcriber.use_word_aligner)
    
    def test_init_with_invalid_model_size(self):
        """Test initialization with invalid model size."""
        with self.assertRaises(ValueError):
            WhisperTranscriber(model_size="invalid_size")
    
    @patch('src.transcription.whisper_transcriber.WhisperModel')
    def test_init_with_word_aligner(self, mock_whisper_model):
        """Test initialization with word aligner."""
        with patch('src.transcription.whisper_transcriber.WordAligner') as mock_word_aligner:
            transcriber = WhisperTranscriber(
                model_size="tiny",
                device="cpu",
                use_word_aligner=True
            )
            
            self.assertTrue(transcriber.use_word_aligner)
            self.assertIsNotNone(transcriber.word_aligner)
            mock_word_aligner.assert_called_once_with(device="cpu")
    
    def test_transcribe_file(self):
        """Test transcribing an audio file."""
        result = self.transcriber.transcribe_file(self.temp_audio.name)
        
        # Verify the model was called correctly
        self.mock_model.transcribe.assert_called_once()
        args, kwargs = self.mock_model.transcribe.call_args
        self.assertEqual(args[0], self.temp_audio.name)
        
        # Check the result structure
        self.assertIn("segments", result)
        self.assertIn("language", result)
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["language_probability"], 0.99)
        
        # Check segment contents
        self.assertEqual(len(result["segments"]), 1)
        segment = result["segments"][0]
        self.assertEqual(segment["text"], "This is a test sentence.")
        self.assertEqual(segment["start"], 0.0)
        self.assertEqual(segment["end"], 5.0)
        
        # Check words
        self.assertEqual(len(segment["words"]), 5)
        self.assertEqual(segment["words"][0]["text"], "This")
        self.assertEqual(segment["words"][0]["start"], 0.0)
        self.assertEqual(segment["words"][0]["end"], 0.5)
    
    def test_transcribe_numpy_array(self):
        """Test transcribing a numpy array."""
        result = self.transcriber.transcribe(self.audio_data, sample_rate=self.sample_rate)
        
        # Verify the model was called correctly
        self.mock_model.transcribe.assert_called_once()
        args, kwargs = self.mock_model.transcribe.call_args
        self.assertTrue(isinstance(args[0], np.ndarray))
        self.assertEqual(kwargs["sr"], self.sample_rate)
        
        # Check the result structure (should be the same as with file)
        self.assertIn("segments", result)
        self.assertIn("language", result)
    
    def test_transcribe_without_sample_rate(self):
        """Test transcribing a numpy array without providing sample rate."""
        with self.assertRaises(ValueError):
            self.transcriber.transcribe(self.audio_data)
    
    @patch('src.transcription.whisper_transcriber.WhisperModel')
    def test_transcribe_with_word_aligner(self, mock_whisper_model):
        """Test transcribing with word timestamp refinement."""
        # Setup mock
        mock_whisper_model.return_value.transcribe.return_value = (
            [
                MagicMock(
                    start=0.0,
                    end=5.0,
                    text="This is a test sentence.",
                    words=[
                        MagicMock(start=0.0, end=0.5, word="This"),
                        MagicMock(start=0.6, end=0.9, word="is"),
                        MagicMock(start=1.0, end=1.2, word="a"),
                        MagicMock(start=1.3, end=2.0, word="test"),
                        MagicMock(start=2.1, end=3.0, word="sentence.")
                    ]
                )
            ],
            MagicMock(language="en", language_probability=0.99)
        )
        
        # Create transcriber with word aligner
        with patch('src.transcription.whisper_transcriber.WordAligner') as mock_word_aligner:
            # Setup the mock word aligner
            mock_aligner_instance = mock_word_aligner.return_value
            
            # Make refine_word_timestamps return modified timestamps
            def mock_refine(audio_data, sample_rate, words, segment_start):
                refined = []
                for word in words:
                    refined_word = word.copy()
                    # Add small offset to make it clear refinement happened
                    refined_word["start"] += 0.01
                    refined_word["end"] += 0.01
                    refined.append(refined_word)
                return refined
                
            mock_aligner_instance.refine_word_timestamps.side_effect = mock_refine
            
            # Create transcriber with word aligner
            transcriber = WhisperTranscriber(
                model_size="tiny",
                device="cpu",
                use_word_aligner=True
            )
            
            # Run transcription
            result = transcriber.transcribe(
                self.audio_data, 
                sample_rate=self.sample_rate,
                refine_word_timestamps=True
            )
            
            # Check that word timestamps were refined
            words = result["segments"][0]["words"]
            self.assertAlmostEqual(words[0]["start"], 0.01)
            self.assertAlmostEqual(words[0]["end"], 0.51)
            
            # Check that the aligner was called
            mock_aligner_instance.refine_word_timestamps.assert_called()
    
    def test_align_transcript(self):
        """Test aligning a transcript with audio."""
        with patch.object(self.transcriber, 'word_aligner') as mock_aligner:
            # Setup the mock aligner
            mock_aligner.align_transcript_with_audio.return_value = [
                {"text": "This", "start": 0.1, "end": 0.5},
                {"text": "is", "start": 0.6, "end": 0.9},
                {"text": "a", "start": 1.0, "end": 1.2},
                {"text": "test", "start": 1.3, "end": 2.0}
            ]
            
            # Set the word aligner to our mocked one
            self.transcriber.use_word_aligner = True
            self.transcriber.word_aligner = mock_aligner
            
            # Run the alignment
            transcript = "This is a test"
            result = self.transcriber.align_transcript(
                self.audio_data,
                self.sample_rate,
                transcript
            )
            
            # Check the result
            self.assertEqual(len(result), 4)
            self.assertEqual(result[0]["text"], "This")
            self.assertEqual(result[0]["start"], 0.1)
            self.assertEqual(result[0]["end"], 0.5)
            
            # Check that the aligner was called correctly
            mock_aligner.align_transcript_with_audio.assert_called_once_with(
                self.audio_data,
                self.sample_rate,
                transcript,
                0.0,
                None
            )


class TestWordAligner(unittest.TestCase):
    """Test case for WordAligner."""
    
    def setUp(self):
        """Set up test environment."""
        self.aligner = WordAligner(device="cpu")
        
        # Create sample audio data (1 second)
        self.sample_rate = 16000
        duration = 5  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        # Create a simple audio signal with some clear patterns
        # A sine wave that gets louder in the middle of each word
        self.audio_data = np.zeros_like(t)
        
        # Word 1: "Hello" (0.0 - 1.0s)
        word1_mask = (t >= 0.0) & (t < 1.0)
        self.audio_data[word1_mask] = 0.5 * np.sin(2 * np.pi * 440 * t[word1_mask])
        
        # Silence (1.0 - 1.2s)
        
        # Word 2: "world" (1.2 - 2.0s)
        word2_mask = (t >= 1.2) & (t < 2.0)
        self.audio_data[word2_mask] = 0.7 * np.sin(2 * np.pi * 330 * t[word2_mask])
        
        # Silence (2.0 - 2.3s)
        
        # Word 3: "test" (2.3 - 3.0s)
        word3_mask = (t >= 2.3) & (t < 3.0)
        self.audio_data[word3_mask] = 0.6 * np.sin(2 * np.pi * 550 * t[word3_mask])
        
        # Create test words with deliberately imprecise timestamps
        self.words = [
            {"text": "Hello", "start": 0.1, "end": 0.9, "score": 0.8},
            {"text": "world", "start": 1.3, "end": 1.9, "score": 0.7},
            {"text": "test", "start": 2.4, "end": 2.9, "score": 0.9}
        ]
    
    def test_refine_word_timestamps(self):
        """Test refining word timestamps."""
        refined_words = self.aligner.refine_word_timestamps(
            self.audio_data,
            self.sample_rate,
            self.words
        )
        
        # Should have the same number of words
        self.assertEqual(len(refined_words), len(self.words))
        
        # Each word should have timestamps
        for word in refined_words:
            self.assertIn("start", word)
            self.assertIn("end", word)
            self.assertIsInstance(word["start"], float)
            self.assertIsInstance(word["end"], float)
        
        # Timestamps should be adjusted
        for i in range(len(self.words)):
            # The refined timestamps could be different (better)
            # We just check that they make sense
            self.assertLessEqual(refined_words[i]["start"], refined_words[i]["end"])
            
            # Start and end should be within reasonable bounds of original
            # (allowing for the refinement to improve them)
            original_duration = self.words[i]["end"] - self.words[i]["start"]
            refined_duration = refined_words[i]["end"] - refined_words[i]["start"]
            
            # Duration shouldn't change drastically
            # (This is a fuzzy test as the exact behavior depends on the audio)
            self.assertLess(abs(refined_duration - original_duration), original_duration)
    
    def test_fix_overlapping_words(self):
        """Test fixing overlapping word timestamps."""
        # Create words with overlapping timestamps
        overlapping_words = [
            {"text": "First", "start": 0.0, "end": 1.2},
            {"text": "second", "start": 1.0, "end": 2.0},  # Overlaps with first and third
            {"text": "third", "start": 1.8, "end": 2.5}
        ]
        
        fixed_words = self.aligner._fix_overlapping_words(overlapping_words)
        
        # Should have the same number of words
        self.assertEqual(len(fixed_words), len(overlapping_words))
        
        # Check that overlaps were fixed
        for i in range(len(fixed_words) - 1):
            self.assertLessEqual(fixed_words[i]["end"], fixed_words[i+1]["start"])
    
    def test_align_transcript_with_audio(self):
        """Test aligning a full transcript with audio."""
        transcript = "Hello world test"
        
        aligned_words = self.aligner.align_transcript_with_audio(
            self.audio_data,
            self.sample_rate,
            transcript
        )
        
        # Should have three words
        self.assertEqual(len(aligned_words), 3)
        
        # Check word content
        self.assertEqual(aligned_words[0]["text"], "Hello")
        self.assertEqual(aligned_words[1]["text"], "world")
        self.assertEqual(aligned_words[2]["text"], "test")
        
        # Check timestamps
        for word in aligned_words:
            self.assertIn("start", word)
            self.assertIn("end", word)
            self.assertIsInstance(word["start"], float)
            self.assertIsInstance(word["end"], float)
            self.assertLessEqual(word["start"], word["end"])
    
    def test_align_empty_transcript(self):
        """Test aligning an empty transcript."""
        aligned_words = self.aligner.align_transcript_with_audio(
            self.audio_data,
            self.sample_rate,
            ""
        )
        
        # Should be empty
        self.assertEqual(len(aligned_words), 0)
    
    def test_align_with_segment_bounds(self):
        """Test aligning with explicit segment bounds."""
        transcript = "world"  # Just the middle word
        
        aligned_words = self.aligner.align_transcript_with_audio(
            self.audio_data,
            self.sample_rate,
            transcript,
            segment_start=1.0,
            segment_end=2.5
        )
        
        # Should have one word
        self.assertEqual(len(aligned_words), 1)
        
        # Check word content
        self.assertEqual(aligned_words[0]["text"], "world")
        
        # Check timestamp is within segment bounds
        self.assertGreaterEqual(aligned_words[0]["start"], 1.0)
        self.assertLessEqual(aligned_words[0]["end"], 2.5)


if __name__ == "__main__":
    unittest.main() 