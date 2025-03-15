"""
Unit tests for the chunked audio processor.
"""

import unittest
import os
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import Mock, patch, MagicMock

from src.audio.chunked_processor import ChunkedProcessor, ChunkingError
from src.ml.diarization import Diarizer

class TestChunkedProcessor(unittest.TestCase):
    """Test cases for the ChunkedProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test audio file
        self.sample_rate = 16000
        duration = 10  # seconds
        self.audio_data = np.random.rand(duration * self.sample_rate).astype(np.float32)
        self.audio_file = os.path.join(self.temp_dir, "test_audio.wav")
        sf.write(self.audio_file, self.audio_data, self.sample_rate)
        
        # Create a mock processor function
        self.mock_processor = MagicMock()
        self.mock_processor.return_value = {"result": "processed"}
        
        # Initialize the chunked processor
        self.chunked_processor = ChunkedProcessor(
            chunk_size=2.0,  # 2 seconds per chunk
            overlap=0.5,     # 0.5 seconds overlap
            processor_func=self.mock_processor
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test files
        if os.path.exists(self.audio_file):
            os.remove(self.audio_file)
        
        # Remove any chunk files that might have been created
        for file in os.listdir(self.temp_dir):
            if file.startswith("chunk_"):
                os.remove(os.path.join(self.temp_dir, file))
        
        # Remove the temporary directory
        os.rmdir(self.temp_dir)
    
    def test_init(self):
        """Test initialization of ChunkedProcessor."""
        self.assertEqual(self.chunked_processor.chunk_size, 2.0)
        self.assertEqual(self.chunked_processor.overlap, 0.5)
        self.assertEqual(self.chunked_processor.processor_func, self.mock_processor)
    
    def test_process_file(self):
        """Test processing an audio file in chunks."""
        # Process the test file
        result = self.chunked_processor.process_file(self.audio_file)
        
        # Check that the processor function was called multiple times
        # For a 10-second file with 2-second chunks and 0.5-second overlap,
        # we expect 7 chunks: 0-2, 1.5-3.5, 3-5, 4.5-6.5, 6-8, 7.5-9.5, 9-10
        self.assertEqual(self.mock_processor.call_count, 7)
        
        # Check that the result contains the merged results
        self.assertIn("merged_result", result)
    
    def test_process_data(self):
        """Test processing audio data in chunks."""
        # Process the audio data
        result = self.chunked_processor.process_data(
            self.audio_data, self.sample_rate
        )
        
        # Check that the processor function was called multiple times
        self.assertEqual(self.mock_processor.call_count, 7)
        
        # Check that the result contains the merged results
        self.assertIn("merged_result", result)
    
    def test_save_chunks_to_disk(self):
        """Test saving audio chunks to disk."""
        # Save chunks to disk
        chunk_files = self.chunked_processor._save_chunks_to_disk(
            self.audio_data, self.sample_rate, self.temp_dir
        )
        
        # Check that the expected number of chunk files were created
        self.assertEqual(len(chunk_files), 7)
        
        # Check that all chunk files exist
        for chunk_file in chunk_files:
            self.assertTrue(os.path.exists(chunk_file))
    
    def test_cleanup_chunks(self):
        """Test cleaning up chunk files."""
        # Save chunks to disk
        chunk_files = self.chunked_processor._save_chunks_to_disk(
            self.audio_data, self.sample_rate, self.temp_dir
        )
        
        # Verify files exist
        for chunk_file in chunk_files:
            self.assertTrue(os.path.exists(chunk_file))
        
        # Clean up chunks
        self.chunked_processor._cleanup_chunks(chunk_files)
        
        # Verify files no longer exist
        for chunk_file in chunk_files:
            self.assertFalse(os.path.exists(chunk_file))
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        # Try to process a non-existent file
        with self.assertRaises(FileNotFoundError):
            self.chunked_processor.process_file("non_existent_file.wav")
    
    def test_chunking_error(self):
        """Test handling of errors during chunk processing."""
        # Create a processor function that raises an exception
        def failing_processor(*args, **kwargs):
            raise ValueError("Test error")
        
        # Create a chunked processor with the failing processor
        chunked_processor = ChunkedProcessor(
            chunk_size=2.0,
            overlap=0.5,
            processor_func=failing_processor
        )
        
        # Try to process the file
        with self.assertRaises(ChunkingError):
            chunked_processor.process_file(self.audio_file)

if __name__ == "__main__":
    unittest.main() 