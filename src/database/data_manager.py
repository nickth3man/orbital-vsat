"""
Data Management module for VSAT.

This module provides functionality for data archiving, database statistics,
backup/restore operations, and data pruning.
"""

import os
import json
import shutil
import logging
import datetime
import zipfile
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import time

from sqlalchemy import func, text, inspect
from sqlalchemy.orm import Session

from src.database.models import Base, Recording, Speaker, TranscriptSegment, TranscriptWord
from src.database.db_manager import DatabaseManager
from src.utils.error_handler import VSATError, DatabaseError

logger = logging.getLogger(__name__)

class DataManagerError(DatabaseError):
    """Exception for data management operations."""
    pass

class DataManager:
    """Class for managing database archiving, backup, statistics, and pruning operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the data manager.
        
        Args:
            db_manager: The database manager instance
        """
        self.db_manager = db_manager
        self.engine = db_manager.engine
        
    def get_database_statistics(self, session: Session) -> Dict[str, Any]:
        """Collect statistics about the database.
        
        Args:
            session: Database session
            
        Returns:
            Dict[str, Any]: Statistics about the database
        """
        try:
            # Get record counts
            recording_count = session.query(func.count(Recording.id)).scalar() or 0
            speaker_count = session.query(func.count(Speaker.id)).scalar() or 0
            segment_count = session.query(func.count(TranscriptSegment.id)).scalar() or 0
            word_count = session.query(func.count(TranscriptWord.id)).scalar() or 0
            
            # Get total audio duration
            total_duration = session.query(func.sum(Recording.duration)).scalar() or 0
            
            # Get database file size
            db_path = self.engine.url.database
            db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
            
            # Get table sizes
            inspector = inspect(self.engine)
            table_stats = {}
            
            with self.engine.connect() as conn:
                for table_name in inspector.get_table_names():
                    # Get row count
                    row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0
                    
                    # Get table size (SQLite specific)
                    # This uses SQLite page count and page size to estimate table size
                    page_count = conn.execute(text(f"PRAGMA page_count")).scalar() or 0
                    page_size = conn.execute(text(f"PRAGMA page_size")).scalar() or 0
                    table_size = (page_count * page_size) / len(inspector.get_table_names())  # Rough estimate
                    
                    table_stats[table_name] = {
                        'row_count': row_count,
                        'size_bytes': int(table_size)
                    }
            
            # Collect statistics on speakers
            speakers_with_voice_prints = session.query(func.count(Speaker.id)) \
                .filter(Speaker.voice_print.isnot(None)) \
                .scalar() or 0
            
            avg_speaking_time = session.query(
                func.avg(TranscriptSegment.end_time - TranscriptSegment.start_time)
            ).scalar() or 0
            
            # Database health metrics
            integrity_check = "ok"
            try:
                with self.engine.connect() as conn:
                    integrity_result = conn.execute(text("PRAGMA integrity_check")).scalar()
                    integrity_check = integrity_result
            except Exception as e:
                integrity_check = f"Error: {str(e)}"
            
            # Compile all statistics
            stats = {
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'database': {
                    'path': db_path,
                    'size_bytes': db_size,
                    'size_formatted': self._format_size(db_size),
                    'tables': table_stats,
                    'integrity_check': integrity_check
                },
                'recordings': {
                    'count': recording_count,
                    'total_duration': total_duration,
                    'total_duration_formatted': self._format_duration(total_duration),
                    'avg_duration': (total_duration / recording_count) if recording_count > 0 else 0
                },
                'speakers': {
                    'count': speaker_count,
                    'with_voice_prints': speakers_with_voice_prints,
                    'avg_speaking_time': avg_speaking_time
                },
                'transcripts': {
                    'segment_count': segment_count,
                    'word_count': word_count,
                    'avg_words_per_segment': (word_count / segment_count) if segment_count > 0 else 0
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error collecting database statistics: {e}")
            raise DataManagerError(f"Failed to collect database statistics: {str(e)}")
    
    def create_backup(self, output_path: Optional[str] = None) -> str:
        """Create a backup of the database.
        
        Args:
            output_path: Optional path for the backup file. If None, uses a default location.
            
        Returns:
            str: Path to the created backup file
        """
        try:
            # Determine backup path
            if output_path is None:
                db_dir = Path.home() / '.vsat' / 'backups'
                db_dir.mkdir(exist_ok=True, parents=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = str(db_dir / f"vsat_backup_{timestamp}.zip")
            
            # Create a temporary directory for backup files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Get the database file path
                db_path = self.engine.url.database
                
                if not os.path.exists(db_path):
                    raise DataManagerError(f"Database file not found at {db_path}")
                
                # Create a copy of the database file to avoid locking issues
                db_copy_path = os.path.join(temp_dir, "vsat.db")
                
                # Use the SQLite backup API for a consistent backup
                with sqlite3.connect(db_path) as src_conn:
                    with sqlite3.connect(db_copy_path) as dst_conn:
                        src_conn.backup(dst_conn)
                
                # Create a metadata file with backup information
                metadata = {
                    'version': '1.0',
                    'timestamp': datetime.datetime.utcnow().isoformat(),
                    'database_path': db_path,
                    'tables': [table for table in inspect(self.engine).get_table_names()]
                }
                
                metadata_path = os.path.join(temp_dir, "backup_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Create the ZIP archive containing the database copy and metadata
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
                    backup_zip.write(db_copy_path, arcname="vsat.db")
                    backup_zip.write(metadata_path, arcname="backup_metadata.json")
            
            logger.info(f"Created database backup at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            raise DataManagerError(f"Failed to create database backup: {str(e)}")
    
    def restore_backup(self, backup_path: str) -> bool:
        """Restore the database from a backup file.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            bool: True if the restore was successful
        """
        try:
            if not os.path.exists(backup_path):
                raise DataManagerError(f"Backup file not found at {backup_path}")
            
            # Create a temporary directory for extracting backup files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the backup archive
                with zipfile.ZipFile(backup_path, 'r') as backup_zip:
                    backup_zip.extractall(temp_dir)
                
                # Verify backup metadata
                metadata_path = os.path.join(temp_dir, "backup_metadata.json")
                if not os.path.exists(metadata_path):
                    raise DataManagerError("Invalid backup: missing metadata file")
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Verify database file
                db_backup_path = os.path.join(temp_dir, "vsat.db")
                if not os.path.exists(db_backup_path):
                    raise DataManagerError("Invalid backup: missing database file")
                
                # Get the current database file path
                db_path = self.engine.url.database
                
                # Create a backup of the existing database before restoring
                if os.path.exists(db_path):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    db_backup = f"{db_path}.bak.{timestamp}"
                    shutil.copy2(db_path, db_backup)
                    logger.info(f"Created backup of existing database at {db_backup}")
                
                # Close all database connections
                self.db_manager.Session.remove()
                
                # Replace the database file with the backup
                try:
                    shutil.copy2(db_backup_path, db_path)
                except Exception as e:
                    raise DataManagerError(f"Failed to restore database: {str(e)}")
                
                logger.info(f"Restored database from backup {backup_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error restoring database from backup: {e}")
            raise DataManagerError(f"Failed to restore database: {str(e)}")
    
    def archive_recording(self, session: Session, recording_id: int, archive_dir: Optional[str] = None) -> str:
        """Archive a recording and all its associated data.
        
        Args:
            session: Database session
            recording_id: ID of the recording to archive
            archive_dir: Optional directory for the archive file. If None, uses a default location.
            
        Returns:
            str: Path to the created archive file
        """
        try:
            # Get the recording
            recording = session.query(Recording).filter(Recording.id == recording_id).first()
            if not recording:
                raise DataManagerError(f"Recording with ID {recording_id} not found")
            
            # Determine archive path
            if archive_dir is None:
                archive_dir = Path.home() / '.vsat' / 'archives'
                archive_dir.mkdir(exist_ok=True, parents=True)
            else:
                archive_dir = Path(archive_dir)
                archive_dir.mkdir(exist_ok=True, parents=True)
            
            # Create a unique filename for the archive
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = recording.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
            archive_path = str(archive_dir / f"{safe_filename}_{timestamp}.zip")
            
            # Create a temporary directory for the archive files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a JSON file with the recording metadata
                recording_data = recording.to_dict()
                
                # Get related data
                segments = [segment.to_dict() for segment in recording.segments]
                
                # Collect all words for each segment
                for i, segment in enumerate(recording.segments):
                    segments[i]['words'] = [word.to_dict() for word in segment.words]
                
                # Get speaker data for all speakers in the recording
                speakers = [speaker.to_dict() for speaker in recording.speakers]
                
                # Combine all data
                archive_data = {
                    'recording': recording_data,
                    'segments': segments,
                    'speakers': speakers,
                    'timestamp': datetime.datetime.utcnow().isoformat(),
                    'archive_version': '1.0'
                }
                
                # Save the data to a JSON file
                data_path = os.path.join(temp_dir, "recording_data.json")
                with open(data_path, 'w') as f:
                    json.dump(archive_data, f, indent=2)
                
                # Copy the audio file if it exists
                audio_file = recording.path
                audio_dest = os.path.join(temp_dir, os.path.basename(audio_file))
                if os.path.exists(audio_file):
                    shutil.copy2(audio_file, audio_dest)
                else:
                    logger.warning(f"Audio file not found: {audio_file}")
                
                # Create the ZIP archive
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive_zip:
                    archive_zip.write(data_path, arcname="recording_data.json")
                    if os.path.exists(audio_dest):
                        archive_zip.write(audio_dest, arcname=os.path.basename(audio_file))
            
            logger.info(f"Archived recording {recording.filename} to {archive_path}")
            return archive_path
            
        except Exception as e:
            logger.error(f"Error archiving recording: {e}")
            raise DataManagerError(f"Failed to archive recording: {str(e)}")
    
    def restore_archive(self, session: Session, archive_path: str) -> int:
        """Restore a recording from an archive.
        
        Args:
            session: Database session
            archive_path: Path to the archive file
            
        Returns:
            int: ID of the restored recording
        """
        try:
            if not os.path.exists(archive_path):
                raise DataManagerError(f"Archive file not found at {archive_path}")
            
            # Create a temporary directory for extracting archive files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the archive
                with zipfile.ZipFile(archive_path, 'r') as archive_zip:
                    archive_zip.extractall(temp_dir)
                
                # Read the recording data
                data_path = os.path.join(temp_dir, "recording_data.json")
                if not os.path.exists(data_path):
                    raise DataManagerError("Invalid archive: missing recording data file")
                
                with open(data_path, 'r') as f:
                    archive_data = json.load(f)
                
                # Extract recording data
                recording_data = archive_data.get('recording', {})
                segments_data = archive_data.get('segments', [])
                speakers_data = archive_data.get('speakers', [])
                
                # Create speakers if they don't exist
                speaker_id_map = {}  # Maps original speaker IDs to new speaker IDs
                for speaker_data in speakers_data:
                    # Check if speaker already exists with the same name
                    speaker = None
                    if speaker_data.get('name'):
                        speaker = session.query(Speaker).filter(Speaker.name == speaker_data['name']).first()
                    
                    if not speaker:
                        # Create a new speaker
                        speaker = Speaker(
                            name=speaker_data.get('name'),
                            meta_data=speaker_data.get('meta_data')
                        )
                        session.add(speaker)
                        session.flush()  # Generate ID without committing
                    
                    # Map the original ID to the new ID
                    speaker_id_map[speaker_data['id']] = speaker.id
                
                # Create the recording
                original_id = recording_data['id']
                recording = Recording(
                    filename=recording_data.get('filename'),
                    path=recording_data.get('path'),
                    duration=recording_data.get('duration'),
                    sample_rate=recording_data.get('sample_rate'),
                    channels=recording_data.get('channels'),
                    processed=recording_data.get('processed', False),
                    meta_data=recording_data.get('meta_data')
                )
                session.add(recording)
                session.flush()  # Generate ID without committing
                
                # Create segments and words
                for segment_data in segments_data:
                    # Map the speaker ID if it exists
                    original_speaker_id = segment_data.get('speaker_id')
                    new_speaker_id = speaker_id_map.get(original_speaker_id) if original_speaker_id else None
                    
                    segment = TranscriptSegment(
                        recording_id=recording.id,
                        speaker_id=new_speaker_id,
                        start_time=segment_data.get('start_time'),
                        end_time=segment_data.get('end_time'),
                        text=segment_data.get('text'),
                        confidence=segment_data.get('confidence'),
                        meta_data=segment_data.get('meta_data')
                    )
                    session.add(segment)
                    session.flush()  # Generate ID without committing
                    
                    # Create words if they exist
                    words_data = segment_data.get('words', [])
                    for word_data in words_data:
                        word = TranscriptWord(
                            segment_id=segment.id,
                            text=word_data.get('text'),
                            start_time=word_data.get('start_time'),
                            end_time=word_data.get('end_time'),
                            confidence=word_data.get('confidence'),
                            meta_data=word_data.get('meta_data')
                        )
                        session.add(word)
                
                # Commit all changes
                session.commit()
                
                # Copy the audio file to the correct location if needed
                audio_files = [f for f in os.listdir(temp_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
                if audio_files and not os.path.exists(recording.path):
                    audio_src = os.path.join(temp_dir, audio_files[0])
                    audio_dir = os.path.dirname(recording.path)
                    os.makedirs(audio_dir, exist_ok=True)
                    shutil.copy2(audio_src, recording.path)
                
                logger.info(f"Restored recording {recording.filename} from archive {archive_path}")
                return recording.id
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error restoring recording from archive: {e}")
            raise DataManagerError(f"Failed to restore recording: {str(e)}")
    
    def prune_data(self, session: Session, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Prune data from the database based on configured rules.
        
        Args:
            session: Database session
            rules: Dictionary of pruning rules
                {
                    'older_than_days': int,  # Remove recordings older than X days
                    'remove_unprocessed': bool,  # Remove recordings that haven't been processed
                    'min_duration': float,  # Remove recordings shorter than X seconds
                    'max_duration': float,  # Remove recordings longer than X seconds
                    'speakers': List[int],  # Speaker IDs to remove (with their segments)
                    'remove_orphaned_speakers': bool,  # Remove speakers with no segments
                }
            
        Returns:
            Dict[str, Any]: Statistics about the pruning operation
        """
        try:
            results = {
                'recordings_removed': 0,
                'segments_removed': 0,
                'words_removed': 0,
                'speakers_removed': 0,
                'bytes_freed': 0
            }
            
            # Process recordings to remove
            recordings_to_remove = []
            
            # Add recordings based on age
            if rules.get('older_than_days'):
                cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=rules['older_than_days'])
                old_recordings = session.query(Recording) \
                    .filter(Recording.created_at < cutoff_date) \
                    .all()
                recordings_to_remove.extend(old_recordings)
            
            # Add unprocessed recordings
            if rules.get('remove_unprocessed'):
                unprocessed = session.query(Recording) \
                    .filter(Recording.processed == False) \
                    .all()
                recordings_to_remove.extend([r for r in unprocessed if r not in recordings_to_remove])
            
            # Add recordings based on duration
            if rules.get('min_duration'):
                short_recordings = session.query(Recording) \
                    .filter(Recording.duration < rules['min_duration']) \
                    .all()
                recordings_to_remove.extend([r for r in short_recordings if r not in recordings_to_remove])
            
            if rules.get('max_duration'):
                long_recordings = session.query(Recording) \
                    .filter(Recording.duration > rules['max_duration']) \
                    .all()
                recordings_to_remove.extend([r for r in long_recordings if r not in recordings_to_remove])
            
            # Process speakers to remove
            speakers_to_remove = []
            
            if rules.get('speakers'):
                speakers = session.query(Speaker) \
                    .filter(Speaker.id.in_(rules['speakers'])) \
                    .all()
                speakers_to_remove.extend(speakers)
            
            # Remove speakers with no segments if requested
            if rules.get('remove_orphaned_speakers'):
                # First, find speakers with no segments
                # This subquery gets all speaker IDs that have segments
                speaker_ids_with_segments = session.query(TranscriptSegment.speaker_id) \
                    .filter(TranscriptSegment.speaker_id.isnot(None)) \
                    .distinct().subquery()
                
                # Then find speakers not in that list
                orphaned_speakers = session.query(Speaker) \
                    .filter(~Speaker.id.in_(speaker_ids_with_segments)) \
                    .all()
                
                speakers_to_remove.extend([s for s in orphaned_speakers if s not in speakers_to_remove])
            
            # Calculate total size before deletion
            db_path = self.engine.url.database
            initial_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
            
            # Remove recordings
            for recording in recordings_to_remove:
                # Count segments and words being removed
                segment_count = session.query(func.count(TranscriptSegment.id)) \
                    .filter(TranscriptSegment.recording_id == recording.id) \
                    .scalar() or 0
                
                word_count = session.query(func.count(TranscriptWord.id)) \
                    .join(TranscriptSegment) \
                    .filter(TranscriptSegment.recording_id == recording.id) \
                    .scalar() or 0
                
                results['segments_removed'] += segment_count
                results['words_removed'] += word_count
                
                # Delete the recording (cascades to segments and words)
                session.delete(recording)
                
                results['recordings_removed'] += 1
                
                # Also delete the audio file if it exists
                if recording.path and os.path.exists(recording.path):
                    try:
                        os.remove(recording.path)
                    except Exception as e:
                        logger.warning(f"Failed to delete audio file {recording.path}: {e}")
            
            # Remove speakers
            for speaker in speakers_to_remove:
                # For actual speaker deletion, we need to handle segments
                # This is safer than cascading, as we don't want to delete associated recordings
                segments = session.query(TranscriptSegment) \
                    .filter(TranscriptSegment.speaker_id == speaker.id) \
                    .all()
                
                # Update segments to have NULL speaker_id instead of deleting
                for segment in segments:
                    segment.speaker_id = None
                
                # Now delete the speaker
                session.delete(speaker)
                results['speakers_removed'] += 1
            
            # Commit all changes
            session.commit()
            
            # Calculate space freed
            if os.path.exists(db_path):
                final_size = os.path.getsize(db_path)
                results['bytes_freed'] = max(0, initial_size - final_size)
                results['size_freed_formatted'] = self._format_size(results['bytes_freed'])
            
            # Run VACUUM to reclaim space
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("VACUUM"))
            except Exception as e:
                logger.warning(f"Failed to VACUUM database: {e}")
            
            logger.info(f"Pruned database: {results}")
            return results
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error pruning database: {e}")
            raise DataManagerError(f"Failed to prune database: {str(e)}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format a size in bytes to a human-readable string.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            str: Human-readable size
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def _format_duration(self, seconds: float) -> str:
        """Format a duration in seconds to a human-readable string.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            str: Human-readable duration
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{seconds:.1f}s" 