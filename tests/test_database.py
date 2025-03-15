"""
Tests for the database module.

This module contains tests for the database models and manager.
"""

import os
import unittest
import tempfile
import datetime
import json
from typing import Dict, Any
import time
from unittest.mock import patch
import sys
import os

# Add the parent directory to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

# Adjust imports to work with the project structure
from src.database.models import Base, Speaker, Recording, TranscriptSegment, TranscriptWord
from src.database.db_manager import DatabaseManager
from src.utils.error_handler import DatabaseError, ErrorSeverity
from src.database.data_manager import DataManager, DataManagerError


class TestDatabaseModels(unittest.TestCase):
    """Test case for database models."""
    
    def setUp(self):
        """Set up test database."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Create engine and create all tables
        self.engine = create_engine(f"sqlite:///{self.temp_db.name}")
        Base.metadata.create_all(self.engine)
        
        # Create session
        self.session = Session(self.engine)
    
    def tearDown(self):
        """Clean up after tests."""
        self.session.close()
        self.engine.dispose()  # Ensure all connections are closed
        
        # Add a small delay to ensure file is released
        time.sleep(0.1)
        
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            print(f"Warning: Could not delete temporary database file: {self.temp_db.name}")
            # Continue with the test, this is just cleanup
    
    def test_speaker_model(self):
        """Test Speaker model creation and relationships."""
        # Create a speaker
        speaker = Speaker(name="Test Speaker", 
                         meta_data={"gender": "male", "accent": "american"})
        self.session.add(speaker)
        self.session.commit()
        
        # Test created speaker
        self.assertIsNotNone(speaker.id)
        self.assertEqual(speaker.name, "Test Speaker")
        self.assertEqual(speaker.meta_data["gender"], "male")
        self.assertIsNotNone(speaker.created_at)
        self.assertIsNotNone(speaker.last_seen)
        
        # Test representation
        self.assertIn("Test Speaker", repr(speaker))
        
        # Test to_dict method
        speaker_dict = speaker.to_dict()
        self.assertEqual(speaker_dict["name"], "Test Speaker")
        self.assertEqual(speaker_dict["meta_data"]["gender"], "male")
        
        # Test total_speaking_time with no segments
        self.assertEqual(speaker.total_speaking_time, 0)
    
    def test_recording_model(self):
        """Test Recording model creation and relationships."""
        # Create a recording
        recording = Recording(
            filename="test.wav",
            path="/path/to/test.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2,
            meta_data={"bitrate": 320000}
        )
        self.session.add(recording)
        self.session.commit()
        
        # Test created recording
        self.assertIsNotNone(recording.id)
        self.assertEqual(recording.filename, "test.wav")
        self.assertEqual(recording.duration, 120.5)
        self.assertEqual(recording.meta_data["bitrate"], 320000)
        
        # Test representation
        self.assertIn("test.wav", repr(recording))
        
        # Test to_dict method
        recording_dict = recording.to_dict()
        self.assertEqual(recording_dict["filename"], "test.wav")
        self.assertEqual(recording_dict["duration"], 120.5)
        
        # Test speakers property with no segments
        self.assertEqual(len(recording.speakers), 0)
    
    def test_transcript_segment_model(self):
        """Test TranscriptSegment model creation and relationships."""
        # Create a recording and speaker
        recording = Recording(
            filename="test.wav",
            path="/path/to/test.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2
        )
        speaker = Speaker(name="Test Speaker")
        self.session.add_all([recording, speaker])
        self.session.commit()
        
        # Create a transcript segment
        segment = TranscriptSegment(
            recording_id=recording.id,
            speaker_id=speaker.id,
            start_time=10.5,
            end_time=15.2,
            text="This is a test segment",
            confidence=0.95
        )
        self.session.add(segment)
        self.session.commit()
        
        # Test created segment
        self.assertIsNotNone(segment.id)
        self.assertEqual(segment.text, "This is a test segment")
        self.assertEqual(segment.start_time, 10.5)
        self.assertEqual(segment.end_time, 15.2)
        self.assertEqual(segment.confidence, 0.95)
        
        # Test relationships
        self.assertEqual(segment.recording.filename, "test.wav")
        self.assertEqual(segment.speaker.name, "Test Speaker")
        
        # Test duration property
        self.assertAlmostEqual(segment.duration, 4.7, places=6)
        
        # Test representation
        self.assertIn("start=10.50", repr(segment))
        
        # Test to_dict method
        segment_dict = segment.to_dict()
        self.assertEqual(segment_dict["text"], "This is a test segment")
        self.assertEqual(segment_dict["speaker_name"], "Test Speaker")
    
    def test_transcript_word_model(self):
        """Test TranscriptWord model creation and relationships."""
        # Create dependencies
        recording = Recording(
            filename="test.wav",
            path="/path/to/test.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2
        )
        speaker = Speaker(name="Test Speaker")
        self.session.add_all([recording, speaker])
        self.session.commit()
        
        segment = TranscriptSegment(
            recording_id=recording.id,
            speaker_id=speaker.id,
            start_time=10.5,
            end_time=15.2,
            text="This is a test segment"
        )
        self.session.add(segment)
        self.session.commit()
        
        # Create a word
        word = TranscriptWord(
            segment_id=segment.id,
            text="test",
            start_time=12.3,
            end_time=12.8,
            confidence=0.98
        )
        self.session.add(word)
        self.session.commit()
        
        # Test created word
        self.assertIsNotNone(word.id)
        self.assertEqual(word.text, "test")
        self.assertEqual(word.start_time, 12.3)
        self.assertEqual(word.end_time, 12.8)
        self.assertEqual(word.confidence, 0.98)
        
        # Test relationships
        self.assertEqual(word.segment.text, "This is a test segment")
        
        # Test duration property
        self.assertEqual(word.duration, 0.5)
        
        # Test representation
        self.assertIn("test", repr(word))
        
        # Test to_dict method
        word_dict = word.to_dict()
        self.assertEqual(word_dict["text"], "test")
        self.assertEqual(word_dict["duration"], 0.5)
    
    def test_relationships(self):
        """Test relationships between models."""
        # Create dependencies
        recording = Recording(
            filename="test.wav",
            path="/path/to/test.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2
        )
        speaker = Speaker(name="Test Speaker")
        self.session.add_all([recording, speaker])
        self.session.commit()
        
        # Create transcript segments
        segment1 = TranscriptSegment(
            recording_id=recording.id,
            speaker_id=speaker.id,
            start_time=10.5,
            end_time=15.2,
            text="This is the first segment"
        )
        segment2 = TranscriptSegment(
            recording_id=recording.id,
            speaker_id=speaker.id,
            start_time=20.1,
            end_time=25.7,
            text="This is the second segment"
        )
        self.session.add_all([segment1, segment2])
        self.session.commit()
        
        # Create some words
        word1 = TranscriptWord(
            segment_id=segment1.id,
            text="first",
            start_time=12.3,
            end_time=12.8
        )
        word2 = TranscriptWord(
            segment_id=segment2.id,
            text="second",
            start_time=22.5,
            end_time=23.1
        )
        self.session.add_all([word1, word2])
        self.session.commit()
        
        # Test recording relationships
        self.assertEqual(len(recording.segments), 2)
        self.assertEqual(recording.segments[0].text, "This is the first segment")
        self.assertEqual(recording.segments[1].text, "This is the second segment")
        
        # Test speaker relationships
        self.assertEqual(len(speaker.segments), 2)
        self.assertIn(segment1, speaker.segments)
        self.assertIn(segment2, speaker.segments)
        
        # Test speaker total_speaking_time
        self.assertEqual(speaker.total_speaking_time, (15.2 - 10.5) + (25.7 - 20.1))
        
        # Test recording speakers property
        self.assertEqual(len(recording.speakers), 1)
        self.assertEqual(recording.speakers[0].name, "Test Speaker")
        
        # Test segment word relationships
        self.assertEqual(len(segment1.words), 1)
        self.assertEqual(segment1.words[0].text, "first")
        self.assertEqual(len(segment2.words), 1)
        self.assertEqual(segment2.words[0].text, "second")
        
        # Test word segment relationship
        self.assertEqual(word1.segment.text, "This is the first segment")
        self.assertEqual(word2.segment.text, "This is the second segment")


class TestDatabaseManager(unittest.TestCase):
    """Test case for database manager."""
    
    def setUp(self):
        """Set up test database."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Create database manager
        self.db_manager = DatabaseManager(db_path=self.temp_db.name)
        
        # Get a session
        self.session = self.db_manager.get_session()
    
    def tearDown(self):
        """Clean up after tests."""
        self.db_manager.close_session(self.session)
        self.db_manager.engine.dispose()  # Ensure all connections are closed
        
        # Add a small delay to ensure file is released
        time.sleep(0.1)
        
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            print(f"Warning: Could not delete temporary database file: {self.temp_db.name}")
            # Continue with the test, this is just cleanup
    
    def test_add_recording(self):
        """Test adding a recording."""
        # Add a recording
        recording = self.db_manager.add_recording(
            self.session,
            filename="test.wav",
            path="/path/to/test.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2,
            meta_data={"bitrate": 320000}
        )
        
        # Test created recording
        self.assertIsNotNone(recording.id)
        self.assertEqual(recording.filename, "test.wav")
        self.assertEqual(recording.duration, 120.5)
        self.assertEqual(recording.meta_data["bitrate"], 320000)
    
    def test_get_recording(self):
        """Test getting a recording."""
        # Add a recording
        recording = self.db_manager.add_recording(
            self.session,
            filename="test.wav",
            path="/path/to/test.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2
        )
        
        # Get the recording
        fetched_recording = self.db_manager.get_recording(self.session, recording.id)
        
        # Test fetched recording
        self.assertIsNotNone(fetched_recording)
        self.assertEqual(fetched_recording.id, recording.id)
        self.assertEqual(fetched_recording.filename, "test.wav")
        
        # Test with non-existent id
        non_existent = self.db_manager.get_recording(self.session, 9999)
        self.assertIsNone(non_existent)
    
    def test_get_all_recordings(self):
        """Test getting all recordings."""
        # Add multiple recordings
        recording1 = self.db_manager.add_recording(
            self.session,
            filename="test1.wav",
            path="/path/to/test1.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2
        )
        recording2 = self.db_manager.add_recording(
            self.session,
            filename="test2.wav",
            path="/path/to/test2.wav",
            duration=90.2,
            sample_rate=48000,
            channels=1
        )
        
        # Get all recordings
        recordings = self.db_manager.get_all_recordings(self.session)
        
        # Test fetched recordings
        self.assertEqual(len(recordings), 2)
        self.assertIn(recording1.filename, [r.filename for r in recordings])
        self.assertIn(recording2.filename, [r.filename for r in recordings])
    
    def test_add_speaker(self):
        """Test adding a speaker."""
        # Add a speaker
        speaker = self.db_manager.add_speaker(
            self.session,
            name="Test Speaker",
            voice_print=b"dummy_voice_print_data",
            meta_data={"gender": "female"}
        )
        
        # Test created speaker
        self.assertIsNotNone(speaker.id)
        self.assertEqual(speaker.name, "Test Speaker")
        self.assertEqual(speaker.voice_print, b"dummy_voice_print_data")
        self.assertEqual(speaker.meta_data["gender"], "female")
    
    def test_get_speaker(self):
        """Test getting a speaker."""
        # Add a speaker
        speaker = self.db_manager.add_speaker(
            self.session,
            name="Test Speaker"
        )
        
        # Get the speaker
        fetched_speaker = self.db_manager.get_speaker(self.session, speaker.id)
        
        # Test fetched speaker
        self.assertIsNotNone(fetched_speaker)
        self.assertEqual(fetched_speaker.id, speaker.id)
        self.assertEqual(fetched_speaker.name, "Test Speaker")
        
        # Test with non-existent id
        non_existent = self.db_manager.get_speaker(self.session, 9999)
        self.assertIsNone(non_existent)
    
    def test_get_all_speakers(self):
        """Test getting all speakers."""
        # Add multiple speakers
        speaker1 = self.db_manager.add_speaker(
            self.session,
            name="Speaker 1"
        )
        speaker2 = self.db_manager.add_speaker(
            self.session,
            name="Speaker 2"
        )
        
        # Get all speakers
        speakers = self.db_manager.get_all_speakers(self.session)
        
        # Test fetched speakers
        self.assertEqual(len(speakers), 2)
        self.assertIn(speaker1.name, [s.name for s in speakers])
        self.assertIn(speaker2.name, [s.name for s in speakers])
    
    def test_add_transcript_segment(self):
        """Test adding a transcript segment."""
        # Add dependencies
        recording = self.db_manager.add_recording(
            self.session,
            filename="test.wav",
            path="/path/to/test.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2
        )
        speaker = self.db_manager.add_speaker(
            self.session,
            name="Test Speaker"
        )
        
        # Add a transcript segment
        segment = self.db_manager.add_transcript_segment(
            self.session,
            recording_id=recording.id,
            speaker_id=speaker.id,
            start_time=10.5,
            end_time=15.2,
            text="This is a test segment",
            confidence=0.95,
            meta_data={"background_noise": "low"}
        )
        
        # Test created segment
        self.assertIsNotNone(segment.id)
        self.assertEqual(segment.text, "This is a test segment")
        self.assertEqual(segment.start_time, 10.5)
        self.assertEqual(segment.end_time, 15.2)
        self.assertEqual(segment.confidence, 0.95)
        self.assertEqual(segment.meta_data["background_noise"], "low")
        
        # Test relationships
        self.assertEqual(segment.recording.id, recording.id)
        self.assertEqual(segment.speaker.id, speaker.id)
    
    def test_add_transcript_word(self):
        """Test adding a transcript word."""
        # Add dependencies
        recording = self.db_manager.add_recording(
            self.session,
            filename="test.wav",
            path="/path/to/test.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2
        )
        speaker = self.db_manager.add_speaker(
            self.session,
            name="Test Speaker"
        )
        segment = self.db_manager.add_transcript_segment(
            self.session,
            recording_id=recording.id,
            speaker_id=speaker.id,
            start_time=10.5,
            end_time=15.2,
            text="This is a test segment"
        )
        
        # Add a transcript word
        word = self.db_manager.add_transcript_word(
            self.session,
            segment_id=segment.id,
            text="test",
            start_time=12.3,
            end_time=12.8,
            confidence=0.98,
            meta_data={"emphasis": "medium"}
        )
        
        # Test created word
        self.assertIsNotNone(word.id)
        self.assertEqual(word.text, "test")
        self.assertEqual(word.start_time, 12.3)
        self.assertEqual(word.end_time, 12.8)
        self.assertEqual(word.confidence, 0.98)
        self.assertEqual(word.meta_data["emphasis"], "medium")
        
        # Test relationships
        self.assertEqual(word.segment.id, segment.id)
    
    def test_search_transcript(self):
        """Test searching transcript text."""
        # Add dependencies
        recording = self.db_manager.add_recording(
            self.session,
            filename="test.wav",
            path="/path/to/test.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2
        )
        speaker = self.db_manager.add_speaker(
            self.session,
            name="Test Speaker"
        )
        
        # Add transcript segments with searchable content
        segment1 = self.db_manager.add_transcript_segment(
            self.session,
            recording_id=recording.id,
            speaker_id=speaker.id,
            start_time=10.5,
            end_time=15.2,
            text="This is a test about searching for apple"
        )
        segment2 = self.db_manager.add_transcript_segment(
            self.session,
            recording_id=recording.id,
            speaker_id=speaker.id,
            start_time=20.1,
            end_time=25.7,
            text="Another test about an apple pie recipe"
        )
        segment3 = self.db_manager.add_transcript_segment(
            self.session,
            recording_id=recording.id,
            speaker_id=speaker.id,
            start_time=30.0,
            end_time=40.0,
            text="This has nothing relevant"
        )
        
        # Search for "apple"
        results = self.db_manager.search_transcript(self.session, "apple")
        
        # Test search results
        self.assertEqual(len(results), 2)
        result_texts = [r["segment"]["text"] for r in results]
        self.assertIn(segment1.text, result_texts)
        self.assertIn(segment2.text, result_texts)
        self.assertNotIn(segment3.text, result_texts)
        
        # Search for something not in any segment
        empty_results = self.db_manager.search_transcript(self.session, "banana")
        self.assertEqual(len(empty_results), 0)
    
    def test_get_speaker_statistics(self):
        """Test getting speaker statistics."""
        # Add dependencies
        recording = self.db_manager.add_recording(
            self.session,
            filename="test.wav",
            path="/path/to/test.wav",
            duration=120.5,
            sample_rate=44100,
            channels=2
        )
        speaker = self.db_manager.add_speaker(
            self.session,
            name="Test Speaker"
        )
        
        # Add transcript segments for the speaker
        segment1 = self.db_manager.add_transcript_segment(
            self.session,
            recording_id=recording.id,
            speaker_id=speaker.id,
            start_time=10.5,
            end_time=15.2,
            text="This is the first segment"
        )
        segment2 = self.db_manager.add_transcript_segment(
            self.session,
            recording_id=recording.id,
            speaker_id=speaker.id,
            start_time=20.1,
            end_time=25.7,
            text="This is the second segment"
        )
        
        # Get speaker statistics
        stats = self.db_manager.get_speaker_statistics(self.session, speaker.id)
        
        # Test statistics
        self.assertEqual(stats["recording_count"], 1)
        self.assertAlmostEqual(stats["total_speaking_time"], (15.2 - 10.5) + (25.7 - 20.1), places=6)
        self.assertTrue("speaker" in stats)
        self.assertTrue("word_count" in stats)
        self.assertTrue("average_speaking_time" in stats)
        
        # Test with non-existent speaker
        empty_stats = self.db_manager.get_speaker_statistics(self.session, 9999)
        self.assertEqual(empty_stats, {})


class TestDatabaseErrorHandling(unittest.TestCase):
    """Test case for database error handling."""
    
    def setUp(self):
        """Set up the test environment."""
        print("\nRunning database error handling tests...")
        # Use in-memory database for testing
        self.db_manager = DatabaseManager(":memory:")
        self.data_manager = DataManager(self.db_manager)
        self.session = self.db_manager.get_session()
    
    def tearDown(self):
        """Clean up after tests."""
        self.db_manager.close_session(self.session)
        print("Database error handling tests completed")
    
    @patch('sqlalchemy.orm.query.Query.first')
    def test_get_database_statistics_error(self, mock_first):
        """Test error handling when getting database statistics fails."""
        print("Testing database statistics error handling...")
        # Make the query.first() method raise an exception
        mock_first.side_effect = Exception("Test error")
        
        # Attempt to get database statistics
        with self.assertRaises(DataManagerError) as context:
            self.data_manager.get_database_statistics(self.session)
        
        # Verify the exception details
        self.assertEqual(str(context.exception), "Failed to retrieve database statistics: Test error")
        self.assertEqual(context.exception.details["error"], "Test error")
        print("Database statistics error test passed!")
    
    def test_create_backup_in_memory_error(self):
        """Test error handling when trying to backup an in-memory database."""
        print("Testing in-memory database backup error handling...")
        # The setup uses an in-memory database, so backup should fail
        with self.assertRaises(DataManagerError) as context:
            self.data_manager.create_backup()
        
        # Verify the exception details
        self.assertEqual(str(context.exception), "Unexpected error during database backup: Cannot backup in-memory database")
        self.assertEqual(context.exception.details["source"], ":memory:")
        print("In-memory backup error test passed!")


if __name__ == "__main__":
    unittest.main() 