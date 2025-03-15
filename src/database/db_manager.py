"""
Database manager for VSAT.

This module provides functionality for database connection, initialization, and operations.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool

from src.database.models import Base, Speaker, Recording, TranscriptSegment, TranscriptWord

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Class for managing database operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses default path.
        """
        if db_path is None:
            # Use default path in user's home directory
            db_dir = Path.home() / '.vsat'
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / 'vsat.db')
        
        logger.info(f"Initializing database at {db_path}")
        
        # Create database engine
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            connect_args={"check_same_thread": False}  # Allow multithreaded access
        )
        
        # Create session factory
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)
        
        # Initialize database if it doesn't exist
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session.
        
        Returns:
            Session: A new SQLAlchemy session
        """
        return self.Session()
    
    def close_session(self, session: Session):
        """Close a database session.
        
        Args:
            session: The session to close
        """
        session.close()
    
    def add_recording(self, session: Session, filename: str, path: str, duration: float,
                     sample_rate: int, channels: int, meta_data: Optional[Dict[str, Any]] = None) -> Recording:
        """Add a new recording to the database.
        
        Args:
            session: Database session
            filename: Name of the audio file
            path: Path to the audio file
            duration: Duration of the recording in seconds
            sample_rate: Sample rate of the recording
            channels: Number of audio channels
            meta_data: Optional metadata
            
        Returns:
            Recording: The created recording object
        """
        try:
            recording = Recording(
                filename=filename,
                path=path,
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                meta_data=meta_data
            )
            
            session.add(recording)
            session.commit()
            
            logger.info(f"Added recording: {filename}")
            return recording
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding recording: {e}")
            raise
    
    def get_recording(self, session: Session, recording_id: int) -> Optional[Recording]:
        """Get a recording by ID.
        
        Args:
            session: Database session
            recording_id: ID of the recording
            
        Returns:
            Optional[Recording]: The recording object, or None if not found
        """
        return session.query(Recording).filter(Recording.id == recording_id).first()
    
    def get_all_recordings(self, session: Session) -> List[Recording]:
        """Get all recordings in the database.
        
        Args:
            session: Database session
            
        Returns:
            List[Recording]: List of all recordings
        """
        return session.query(Recording).order_by(Recording.created_at.desc()).all()
    
    def add_speaker(self, session: Session, name: Optional[str] = None,
                   voice_print: Optional[bytes] = None,
                   meta_data: Optional[Dict[str, Any]] = None) -> Speaker:
        """Add a new speaker to the database.
        
        Args:
            session: Database session
            name: Name of the speaker
            voice_print: Speaker embedding as bytes
            meta_data: Optional metadata
            
        Returns:
            Speaker: The created speaker object
        """
        try:
            speaker = Speaker(
                name=name,
                voice_print=voice_print,
                meta_data=meta_data
            )
            
            session.add(speaker)
            session.commit()
            
            logger.info(f"Added speaker: {name}")
            return speaker
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding speaker: {e}")
            raise
    
    def get_speaker(self, session: Session, speaker_id: int) -> Optional[Speaker]:
        """Get a speaker by ID.
        
        Args:
            session: Database session
            speaker_id: ID of the speaker
            
        Returns:
            Optional[Speaker]: The speaker object, or None if not found
        """
        return session.query(Speaker).filter(Speaker.id == speaker_id).first()
    
    def get_all_speakers(self, session: Session) -> List[Speaker]:
        """Get all speakers in the database.
        
        Args:
            session: Database session
            
        Returns:
            List[Speaker]: List of all speakers
        """
        return session.query(Speaker).order_by(Speaker.name).all()
    
    def add_transcript_segment(self, session: Session, recording_id: int,
                              speaker_id: Optional[int], start_time: float,
                              end_time: float, text: Optional[str] = None,
                              confidence: Optional[float] = None,
                              meta_data: Optional[Dict[str, Any]] = None) -> TranscriptSegment:
        """Add a new transcript segment to the database.
        
        Args:
            session: Database session
            recording_id: ID of the recording
            speaker_id: ID of the speaker, or None if unknown
            start_time: Start time of the segment in seconds
            end_time: End time of the segment in seconds
            text: Transcribed text
            confidence: Confidence score
            meta_data: Optional metadata
            
        Returns:
            TranscriptSegment: The created segment object
        """
        try:
            segment = TranscriptSegment(
                recording_id=recording_id,
                speaker_id=speaker_id,
                start_time=start_time,
                end_time=end_time,
                text=text,
                confidence=confidence,
                meta_data=meta_data
            )
            
            session.add(segment)
            session.commit()
            
            return segment
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding transcript segment: {e}")
            raise
    
    def add_transcript_word(self, session: Session, segment_id: int,
                           text: str, start_time: float, end_time: float,
                           confidence: Optional[float] = None,
                           meta_data: Optional[Dict[str, Any]] = None) -> TranscriptWord:
        """Add a new transcript word to the database.
        
        Args:
            session: Database session
            segment_id: ID of the parent segment
            text: Word text
            start_time: Start time of the word in seconds
            end_time: End time of the word in seconds
            confidence: Confidence score
            meta_data: Optional metadata
            
        Returns:
            TranscriptWord: The created word object
        """
        try:
            word = TranscriptWord(
                segment_id=segment_id,
                text=text,
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                meta_data=meta_data
            )
            
            session.add(word)
            session.commit()
            
            return word
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding transcript word: {e}")
            raise
    
    def search_transcript(self, session: Session, query: str) -> List[Dict[str, Any]]:
        """Search for text in transcripts.
        
        Args:
            session: Database session
            query: Search query
            
        Returns:
            List[Dict[str, Any]]: List of matching segments with recording info
        """
        try:
            # Simple search implementation - can be enhanced with FTS5 in the future
            results = session.query(TranscriptSegment, Recording) \
                .join(Recording) \
                .filter(TranscriptSegment.text.like(f"%{query}%")) \
                .all()
            
            return [
                {
                    'segment': segment.to_dict(),
                    'recording': recording.to_dict()
                }
                for segment, recording in results
            ]
            
        except Exception as e:
            logger.error(f"Error searching transcript: {e}")
            raise
    
    def get_speaker_statistics(self, session: Session, speaker_id: int) -> Dict[str, Any]:
        """Get statistics for a speaker.
        
        Args:
            session: Database session
            speaker_id: ID of the speaker
            
        Returns:
            Dict[str, Any]: Statistics for the speaker
        """
        try:
            speaker = self.get_speaker(session, speaker_id)
            if not speaker:
                return {}
            
            # Get total speaking time
            total_time = session.query(func.sum(TranscriptSegment.end_time - TranscriptSegment.start_time)) \
                .filter(TranscriptSegment.speaker_id == speaker_id) \
                .scalar() or 0
            
            # Get word count
            word_count = session.query(func.count(TranscriptWord.id)) \
                .join(TranscriptSegment) \
                .filter(TranscriptSegment.speaker_id == speaker_id) \
                .scalar() or 0
            
            # Get recording count
            recording_count = session.query(func.count(func.distinct(TranscriptSegment.recording_id))) \
                .filter(TranscriptSegment.speaker_id == speaker_id) \
                .scalar() or 0
            
            return {
                'speaker': speaker.to_dict(),
                'total_speaking_time': total_time,
                'word_count': word_count,
                'recording_count': recording_count,
                'average_speaking_time': total_time / recording_count if recording_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting speaker statistics: {e}")
            raise 