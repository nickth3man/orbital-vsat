#!/usr/bin/env python3
"""
Integration tests for the VSAT application.

These tests verify that the entire processing pipeline works correctly
from audio input to final output, including all intermediate steps.
"""

import os
import unittest
import tempfile
import shutil
from pathlib import Path
import logging

import numpy as np
import soundfile as sf

from src.audio.file_handler import AudioFileHandler
from src.audio.processor import AudioProcessor
from src.ml.diarization import Diarizer
from src.ml.speaker_identification import SpeakerIdentifier
from src.transcription.whisper_transcriber import WhisperTranscriber
from src.database.db_manager import DatabaseManager
from src.database.data_manager import DataManager

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationTest(unittest.TestCase):
    """Test the entire processing pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_db_path = os.path.join(cls.temp_dir, "test_integration.db")
        
        # Create a test audio file
        cls.test_audio_path = os.path.join(cls.temp_dir, "test_audio.wav")
        cls._create_test_audio_file(cls.test_audio_path)
        
        # Initialize database
        cls.db_manager = DatabaseManager(cls.test_db_path)
        cls.db_manager.initialize_database()
        
        # Initialize data manager
        cls.data_manager = DataManager(cls.db_manager)
        
        logger.info("Integration test setup complete")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run."""
        # Close database connection
        cls.db_manager.close()
        
        # Remove temporary directory and all files
        shutil.rmtree(cls.temp_dir)
        
        logger.info("Integration test cleanup complete")

    @staticmethod
    def _create_test_audio_file(file_path, duration=5.0, sample_rate=16000):
        """Create a test audio file with synthetic speech-like content."""
        # Generate a simple sine wave with some amplitude modulation to simulate speech
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        
        # Create two different "speakers" with different frequencies
        speaker1 = 0.5 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
        speaker2 = 0.5 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t + 1.5))
        
        # Alternate between speakers
        audio = np.zeros_like(t)
        segment_length = int(sample_rate * 1.0)  # 1-second segments
        
        for i in range(0, len(t), segment_length):
            end = min(i + segment_length, len(t))
            if (i // segment_length) % 2 == 0:
                audio[i:end] = speaker1[i:end]
            else:
                audio[i:end] = speaker2[i:end]
        
        # Write to file
        sf.write(file_path, audio, sample_rate)
        
        logger.info(f"Created test audio file: {file_path}")
        return file_path

    def test_end_to_end_processing(self):
        """Test the entire processing pipeline from audio file to database."""
        logger.info("Starting end-to-end processing test")
        
        # 1. Load audio file
        file_handler = AudioFileHandler()
        audio_data = file_handler.load_file(self.test_audio_path)
        self.assertIsNotNone(audio_data, "Failed to load audio file")
        
        # 2. Initialize audio processor
        processor = AudioProcessor()
        
        # 3. Process audio
        try:
            result = processor.process_audio(
                audio_path=self.test_audio_path,
                audio_data=audio_data,
                perform_diarization=True,
                perform_transcription=True,
                perform_speaker_identification=True
            )
            self.assertIsNotNone(result, "Processing result should not be None")
            
            # 4. Verify result structure
            self.assertIn("diarization", result, "Result should contain diarization data")
            self.assertIn("transcription", result, "Result should contain transcription data")
            
            # 5. Store results in database
            recording_id = self.data_manager.add_recording(
                file_path=self.test_audio_path,
                file_name=os.path.basename(self.test_audio_path),
                duration=audio_data.duration,
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels
            )
            
            self.assertIsNotNone(recording_id, "Recording ID should not be None")
            
            # 6. Add transcript segments
            if "segments" in result["transcription"]:
                for segment in result["transcription"]["segments"]:
                    segment_id = self.data_manager.add_transcript_segment(
                        recording_id=recording_id,
                        start_time=segment["start"],
                        end_time=segment["end"],
                        text=segment["text"],
                        speaker_id=segment.get("speaker_id", None)
                    )
                    self.assertIsNotNone(segment_id, "Segment ID should not be None")
            
            # 7. Verify data was stored correctly
            recording = self.data_manager.get_recording(recording_id)
            self.assertIsNotNone(recording, "Should be able to retrieve recording")
            
            segments = self.data_manager.get_transcript_segments(recording_id)
            self.assertGreater(len(segments), 0, "Should have stored transcript segments")
            
            logger.info("End-to-end processing test completed successfully")
            
        except Exception as e:
            self.fail(f"Processing pipeline raised an exception: {str(e)}")

    def test_chunked_processing(self):
        """Test processing a longer file with chunked processing."""
        logger.info("Starting chunked processing test")
        
        # Create a longer test file
        long_audio_path = os.path.join(self.temp_dir, "long_test_audio.wav")
        self._create_test_audio_file(long_audio_path, duration=15.0)
        
        # Load audio file
        file_handler = AudioFileHandler()
        audio_data = file_handler.load_file(long_audio_path)
        
        # Initialize audio processor with chunked processing
        processor = AudioProcessor(enable_chunked_processing=True, chunk_size_seconds=5.0)
        
        try:
            # Process audio
            result = processor.process_audio(
                audio_path=long_audio_path,
                audio_data=audio_data,
                perform_diarization=True,
                perform_transcription=True
            )
            
            self.assertIsNotNone(result, "Processing result should not be None")
            self.assertIn("diarization", result, "Result should contain diarization data")
            self.assertIn("transcription", result, "Result should contain transcription data")
            
            # Verify chunked processing worked correctly by checking segment boundaries
            if "segments" in result["transcription"]:
                segments = result["transcription"]["segments"]
                self.assertGreater(len(segments), 0, "Should have transcript segments")
                
                # Check that segments span the entire duration
                all_times = [seg["start"] for seg in segments] + [seg["end"] for seg in segments]
                if all_times:
                    min_time = min(all_times)
                    max_time = max(all_times)
                    
                    # Allow some tolerance at the beginning and end
                    self.assertLess(min_time, 1.0, "First segment should start near the beginning")
                    self.assertGreater(max_time, 14.0, "Last segment should end near the end")
            
            logger.info("Chunked processing test completed successfully")
            
        except Exception as e:
            self.fail(f"Chunked processing raised an exception: {str(e)}")

    def test_error_recovery(self):
        """Test error recovery mechanisms in the processing pipeline."""
        logger.info("Starting error recovery test")
        
        # Create an invalid audio file to test error handling
        invalid_audio_path = os.path.join(self.temp_dir, "invalid_audio.wav")
        with open(invalid_audio_path, 'w') as f:
            f.write("This is not a valid audio file")
        
        # Initialize audio processor with error recovery enabled
        processor = AudioProcessor()
        
        try:
            # Attempt to process invalid file
            result = processor.process_audio(
                audio_path=invalid_audio_path,
                audio_data=None,  # Force file loading in processor
                perform_diarization=True,
                perform_transcription=True
            )
            
            # Should return None or raise a handled exception
            self.assertIsNone(result, "Processing invalid file should return None")
            
        except Exception as e:
            # If an exception is raised, it should be a specific VSAT error type
            error_name = type(e).__name__
            self.assertTrue(
                error_name.endswith("Error") and not error_name == "Exception",
                f"Expected a specific error type, got {error_name}"
            )
            
            # The error should contain useful context
            self.assertIn(invalid_audio_path, str(e), "Error should mention the file path")
            
            logger.info(f"Error recovery test completed with expected error: {error_name}")


if __name__ == "__main__":
    unittest.main() 