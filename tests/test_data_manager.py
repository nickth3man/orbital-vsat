"""
Tests for the data management module.
"""

import os
import json
import tempfile
import datetime
import zipfile
import unittest
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, Recording, Speaker, TranscriptSegment, TranscriptWord
from src.database.db_manager import DatabaseManager
from src.database.data_manager import DataManager, DataManagerError

class TestDataManager(unittest.TestCase):
    """Test cases for the DataManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test database
        self.db_path = os.path.join(self.temp_dir.name, "test_vsat.db")
        self.db_manager = DatabaseManager(db_path=self.db_path)
        
        # Create the data manager
        self.data_manager = DataManager(self.db_manager)
        
        # Create a session
        self.session = self.db_manager.get_session()
        
        # Create test data
        self.create_test_data()
    
    def tearDown(self):
        """Clean up after tests."""
        # Close session
        self.session.close()
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def create_test_data(self):
        """Create test data in the database."""
        # Create speakers
        speaker1 = Speaker(name="Speaker 1", voice_print=b'test_voice_print_1')
        speaker2 = Speaker(name="Speaker 2", voice_print=b'test_voice_print_2')
        self.session.add_all([speaker1, speaker2])
        self.session.flush()
        
        # Create recordings
        recording1 = Recording(
            filename="test_recording_1.wav",
            path=os.path.join(self.temp_dir.name, "test_recording_1.wav"),
            duration=120.0,
            sample_rate=16000,
            channels=1,
            processed=True,
            meta_data={"test": "metadata1"}
        )
        recording2 = Recording(
            filename="test_recording_2.wav",
            path=os.path.join(self.temp_dir.name, "test_recording_2.wav"),
            duration=180.0,
            sample_rate=16000,
            channels=1,
            processed=False,
            meta_data={"test": "metadata2"}
        )
        self.session.add_all([recording1, recording2])
        self.session.flush()
        
        # Create segments
        segment1 = TranscriptSegment(
            recording_id=recording1.id,
            speaker_id=speaker1.id,
            start_time=0.0,
            end_time=10.0,
            text="This is a test segment one.",
            confidence=0.95
        )
        segment2 = TranscriptSegment(
            recording_id=recording1.id,
            speaker_id=speaker2.id,
            start_time=10.0,
            end_time=20.0,
            text="This is a test segment two.",
            confidence=0.90
        )
        segment3 = TranscriptSegment(
            recording_id=recording2.id,
            speaker_id=speaker1.id,
            start_time=0.0,
            end_time=15.0,
            text="This is a test segment three.",
            confidence=0.85
        )
        self.session.add_all([segment1, segment2, segment3])
        self.session.flush()
        
        # Create words
        words1 = [
            TranscriptWord(segment_id=segment1.id, text="This", start_time=0.0, end_time=0.5, confidence=0.95),
            TranscriptWord(segment_id=segment1.id, text="is", start_time=0.5, end_time=1.0, confidence=0.95),
            TranscriptWord(segment_id=segment1.id, text="a", start_time=1.0, end_time=1.2, confidence=0.95),
            TranscriptWord(segment_id=segment1.id, text="test", start_time=1.2, end_time=1.7, confidence=0.95),
            TranscriptWord(segment_id=segment1.id, text="segment", start_time=1.7, end_time=2.2, confidence=0.95),
            TranscriptWord(segment_id=segment1.id, text="one", start_time=2.2, end_time=2.5, confidence=0.95)
        ]
        words2 = [
            TranscriptWord(segment_id=segment2.id, text="This", start_time=10.0, end_time=10.5, confidence=0.90),
            TranscriptWord(segment_id=segment2.id, text="is", start_time=10.5, end_time=11.0, confidence=0.90),
            TranscriptWord(segment_id=segment2.id, text="a", start_time=11.0, end_time=11.2, confidence=0.90),
            TranscriptWord(segment_id=segment2.id, text="test", start_time=11.2, end_time=11.7, confidence=0.90),
            TranscriptWord(segment_id=segment2.id, text="segment", start_time=11.7, end_time=12.2, confidence=0.90),
            TranscriptWord(segment_id=segment2.id, text="two", start_time=12.2, end_time=12.5, confidence=0.90)
        ]
        words3 = [
            TranscriptWord(segment_id=segment3.id, text="This", start_time=0.0, end_time=0.5, confidence=0.85),
            TranscriptWord(segment_id=segment3.id, text="is", start_time=0.5, end_time=1.0, confidence=0.85),
            TranscriptWord(segment_id=segment3.id, text="a", start_time=1.0, end_time=1.2, confidence=0.85),
            TranscriptWord(segment_id=segment3.id, text="test", start_time=1.2, end_time=1.7, confidence=0.85),
            TranscriptWord(segment_id=segment3.id, text="segment", start_time=1.7, end_time=2.2, confidence=0.85),
            TranscriptWord(segment_id=segment3.id, text="three", start_time=2.2, end_time=2.5, confidence=0.85)
        ]
        
        self.session.add_all(words1 + words2 + words3)
        
        # Create a test audio file
        with open(recording1.path, 'wb') as f:
            f.write(b'test_audio_data')
        
        # Commit the session
        self.session.commit()
        
        # Store IDs for later use
        self.speaker1_id = speaker1.id
        self.speaker2_id = speaker2.id
        self.recording1_id = recording1.id
        self.recording2_id = recording2.id
        self.segment1_id = segment1.id
        self.segment2_id = segment2.id
        self.segment3_id = segment3.id
    
    def test_init(self):
        """Test initialization of DataManager."""
        self.assertIsNotNone(self.data_manager)
        self.assertEqual(self.data_manager.db_manager, self.db_manager)
        self.assertEqual(self.data_manager.engine, self.db_manager.engine)
    
    def test_get_database_statistics(self):
        """Test collecting database statistics."""
        # Get statistics
        stats = self.data_manager.get_database_statistics(self.session)
        
        # Check that the statistics contain the expected keys
        self.assertIn('timestamp', stats)
        self.assertIn('database', stats)
        self.assertIn('recordings', stats)
        self.assertIn('speakers', stats)
        self.assertIn('transcripts', stats)
        
        # Check database stats
        db_stats = stats['database']
        self.assertEqual(db_stats['path'], self.db_path)
        self.assertGreater(db_stats['size_bytes'], 0)
        self.assertIn('size_formatted', db_stats)
        self.assertIn('tables', db_stats)
        self.assertIn('integrity_check', db_stats)
        
        # Check recording stats
        recording_stats = stats['recordings']
        self.assertEqual(recording_stats['count'], 2)
        self.assertEqual(recording_stats['total_duration'], 300.0)  # 120 + 180
        self.assertIn('total_duration_formatted', recording_stats)
        self.assertEqual(recording_stats['avg_duration'], 150.0)  # (120 + 180) / 2
        
        # Check speaker stats
        speaker_stats = stats['speakers']
        self.assertEqual(speaker_stats['count'], 2)
        self.assertEqual(speaker_stats['with_voice_prints'], 2)
        
        # Check transcript stats
        transcript_stats = stats['transcripts']
        self.assertEqual(transcript_stats['segment_count'], 3)
        self.assertEqual(transcript_stats['word_count'], 18)  # 6 + 6 + 6
        self.assertEqual(transcript_stats['avg_words_per_segment'], 6.0)  # 18 / 3
    
    @patch('src.database.data_manager.sqlite3.connect')
    def test_create_backup(self, mock_sqlite3_connect):
        """Test creating a database backup."""
        # Mock connections
        mock_src_conn = MagicMock()
        mock_dst_conn = MagicMock()
        mock_src_conn.__enter__.return_value = mock_src_conn
        mock_dst_conn.__enter__.return_value = mock_dst_conn
        mock_sqlite3_connect.side_effect = [mock_src_conn, mock_dst_conn]
        
        # Create backup
        backup_path = self.data_manager.create_backup(output_path=os.path.join(self.temp_dir.name, "test_backup.zip"))
        
        # Check that the backup file was created
        self.assertTrue(os.path.exists(backup_path))
        
        # Check that the backup is a valid ZIP file
        self.assertTrue(zipfile.is_zipfile(backup_path))
        
        # Check backup contents
        with zipfile.ZipFile(backup_path, 'r') as backup_zip:
            files = backup_zip.namelist()
            self.assertIn("vsat.db", files)
            self.assertIn("backup_metadata.json", files)
            
            # Check metadata
            with backup_zip.open("backup_metadata.json") as f:
                metadata = json.load(f)
                self.assertIn('version', metadata)
                self.assertIn('timestamp', metadata)
                self.assertIn('database_path', metadata)
                self.assertIn('tables', metadata)
    
    @patch('src.database.data_manager.shutil.copy2')
    @patch('src.database.data_manager.os.path.exists')
    def test_restore_backup(self, mock_exists, mock_copy2):
        """Test restoring from a backup."""
        # Create a mock backup file
        backup_dir = os.path.join(self.temp_dir.name, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, "test_backup.zip")
        
        # Create a temporary directory for the test backup contents
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock database file
            db_file = os.path.join(temp_dir, "vsat.db")
            with open(db_file, 'wb') as f:
                f.write(b'mock_database_data')
            
            # Create metadata
            metadata = {
                'version': '1.0',
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'database_path': self.db_path,
                'tables': ['speakers', 'recordings', 'transcript_segments', 'transcript_words']
            }
            
            metadata_file = os.path.join(temp_dir, "backup_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Create the ZIP file
            with zipfile.ZipFile(backup_path, 'w') as backup_zip:
                backup_zip.write(db_file, arcname="vsat.db")
                backup_zip.write(metadata_file, arcname="backup_metadata.json")
        
        # Mock existence checks
        mock_exists.side_effect = lambda path: path == backup_path or path.endswith("vsat.db") or path.endswith("backup_metadata.json")
        
        # Test restore
        self.data_manager.Session = MagicMock()  # Mock Session to avoid actual removal
        result = self.data_manager.restore_backup(backup_path)
        
        # Check result
        self.assertTrue(result)
        
        # Check that the database file was copied
        mock_copy2.assert_called()
    
    def test_archive_recording(self):
        """Test archiving a recording."""
        # Create an archive directory
        archive_dir = os.path.join(self.temp_dir.name, "archives")
        os.makedirs(archive_dir, exist_ok=True)
        
        # Archive the recording
        archive_path = self.data_manager.archive_recording(
            self.session, self.recording1_id, archive_dir=archive_dir
        )
        
        # Check that the archive file was created
        self.assertTrue(os.path.exists(archive_path))
        
        # Check that the archive is a valid ZIP file
        self.assertTrue(zipfile.is_zipfile(archive_path))
        
        # Check archive contents
        with zipfile.ZipFile(archive_path, 'r') as archive_zip:
            files = archive_zip.namelist()
            self.assertIn("recording_data.json", files)
            
            # Check data
            with archive_zip.open("recording_data.json") as f:
                data = json.load(f)
                self.assertIn('recording', data)
                self.assertIn('segments', data)
                self.assertIn('speakers', data)
                self.assertIn('timestamp', data)
                
                # Check recording
                recording = data['recording']
                self.assertEqual(recording['id'], self.recording1_id)
                self.assertEqual(recording['filename'], "test_recording_1.wav")
                
                # Check segments
                segments = data['segments']
                self.assertEqual(len(segments), 2)  # Two segments for recording1
                
                # Check speakers
                speakers = data['speakers']
                self.assertEqual(len(speakers), 2)  # Two speakers in recording1
    
    def test_restore_archive(self):
        """Test restoring a recording from an archive."""
        # First create an archive
        archive_dir = os.path.join(self.temp_dir.name, "archives")
        os.makedirs(archive_dir, exist_ok=True)
        
        archive_path = self.data_manager.archive_recording(
            self.session, self.recording1_id, archive_dir=archive_dir
        )
        
        # Delete the original recording from the database
        recording = self.session.query(Recording).get(self.recording1_id)
        self.session.delete(recording)
        self.session.commit()
        
        # Restore the recording
        restored_id = self.data_manager.restore_archive(self.session, archive_path)
        
        # Check that a new recording was created
        self.assertIsNotNone(restored_id)
        self.assertNotEqual(restored_id, self.recording1_id)  # Should be a new ID
        
        # Check that the restored recording has the same data
        restored = self.session.query(Recording).get(restored_id)
        self.assertEqual(restored.filename, "test_recording_1.wav")
        self.assertEqual(restored.duration, 120.0)
        
        # Check that the segments were restored
        segments = self.session.query(TranscriptSegment).filter(TranscriptSegment.recording_id == restored_id).all()
        self.assertEqual(len(segments), 2)
        
        # Check that the words were restored
        words = self.session.query(TranscriptWord).join(TranscriptSegment).filter(TranscriptSegment.recording_id == restored_id).all()
        self.assertEqual(len(words), 12)  # 6 words per segment, 2 segments
    
    def test_prune_data(self):
        """Test pruning data based on rules."""
        # Define pruning rules
        rules = {
            'older_than_days': 30,  # Not applicable in our test
            'remove_unprocessed': True,  # Should remove recording2
            'min_duration': 150.0,  # Should remove recording1
            'max_duration': 200.0,  # Not applicable
            'speakers': [self.speaker2_id],  # Should affect speaker2
            'remove_orphaned_speakers': True  # Should remove any speakers with no segments
        }
        
        # Execute pruning
        results = self.data_manager.prune_data(self.session, rules)
        
        # Check pruning results
        self.assertEqual(results['recordings_removed'], 2)  # Both recordings should be removed
        self.assertEqual(results['segments_removed'], 3)  # All segments
        self.assertEqual(results['words_removed'], 18)  # All words
        self.assertEqual(results['speakers_removed'], 2)  # Both speakers (speaker2 explicitly, speaker1 as orphaned)
        self.assertGreater(results['bytes_freed'], 0)
        
        # Verify database state
        recordings = self.session.query(Recording).all()
        self.assertEqual(len(recordings), 0)
        
        segments = self.session.query(TranscriptSegment).all()
        self.assertEqual(len(segments), 0)
        
        words = self.session.query(TranscriptWord).all()
        self.assertEqual(len(words), 0)
        
        speakers = self.session.query(Speaker).all()
        self.assertEqual(len(speakers), 0)
    
    def test_format_size(self):
        """Test formatting size in bytes."""
        self.assertEqual(self.data_manager._format_size(500), "500.00 B")
        self.assertEqual(self.data_manager._format_size(1500), "1.46 KB")
        self.assertEqual(self.data_manager._format_size(1500000), "1.43 MB")
        self.assertEqual(self.data_manager._format_size(1500000000), "1.40 GB")
    
    def test_format_duration(self):
        """Test formatting duration in seconds."""
        self.assertEqual(self.data_manager._format_duration(30), "30.0s")
        self.assertEqual(self.data_manager._format_duration(90), "1m 30s")
        self.assertEqual(self.data_manager._format_duration(3600), "1h 0m 0s")
        self.assertEqual(self.data_manager._format_duration(3661), "1h 1m 1s")

class TestDataManagerErrorHandling(unittest.TestCase):
    """Test case for data manager error handling."""
    
    def setUp(self):
        """Set up the test environment."""
        # Use in-memory database for testing
        self.db_manager = DatabaseManager(":memory:")
        self.data_manager = DataManager(self.db_manager)
        self.session = self.db_manager.get_session()
    
    def tearDown(self):
        """Clean up after tests."""
        self.db_manager.close_session(self.session)
    
    @patch('sqlalchemy.orm.query.Query.first')
    def test_get_database_statistics_error(self, mock_first):
        """Test error handling when getting database statistics fails."""
        # Make the query.first() method raise an exception
        mock_first.side_effect = Exception("Test error")
        
        # Attempt to get database statistics
        with self.assertRaises(DataManagerError) as context:
            self.data_manager.get_database_statistics(self.session)
        
        # Verify the exception details
        self.assertEqual(str(context.exception), "Failed to retrieve database statistics: Test error")
        self.assertEqual(context.exception.details["error"], "Test error")
    
    def test_create_backup_in_memory_error(self):
        """Test error handling when trying to backup an in-memory database."""
        # The setup uses an in-memory database, so backup should fail
        with self.assertRaises(DataManagerError) as context:
            self.data_manager.create_backup()
        
        # Verify the exception details
        self.assertEqual(str(context.exception), "Cannot backup in-memory database")
        self.assertEqual(context.exception.details["db_path"], ":memory:")
    
    @patch('sqlite3.connect')
    def test_create_backup_sqlite_error(self, mock_connect):
        """Test error handling when SQLite operations fail during backup."""
        # Mock the database path to be a real path instead of :memory:
        with patch.object(self.db_manager.engine.url, 'database', 'test.db'):
            # Make sqlite3.connect raise an exception
            mock_connect.side_effect = sqlite3.Error("Database is locked")
            
            # Attempt to create a backup
            with self.assertRaises(DataManagerError) as context:
                self.data_manager.create_backup("test_backup.db")
            
            # Verify the exception details
            self.assertEqual(str(context.exception), "Database backup failed: Database is locked")
            self.assertEqual(context.exception.details["error"], "Database is locked")
    
    def test_restore_backup_file_not_found(self):
        """Test error handling when backup file doesn't exist."""
        # Attempt to restore from a non-existent file
        with self.assertRaises(DataManagerError) as context:
            self.data_manager.restore_backup("nonexistent_backup.db")
        
        # Verify the exception details
        self.assertEqual(str(context.exception), "Backup file not found: nonexistent_backup.db")
        self.assertEqual(context.exception.details["backup_path"], "nonexistent_backup.db")
    
    def test_restore_backup_in_memory_error(self):
        """Test error handling when trying to restore to an in-memory database."""
        # Create a temporary file to use as a fake backup
        with tempfile.NamedTemporaryFile(suffix='.db') as temp_file:
            # Attempt to restore to an in-memory database
            with self.assertRaises(DataManagerError) as context:
                self.data_manager.restore_backup(temp_file.name)
            
            # Verify the exception details
            self.assertEqual(str(context.exception), "Cannot restore to in-memory database")
            self.assertEqual(context.exception.details["db_path"], ":memory:")

if __name__ == '__main__':
    unittest.main() 