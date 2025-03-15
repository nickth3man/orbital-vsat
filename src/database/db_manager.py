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
from sqlalchemy.exc import SQLAlchemyError

from src.database.models import Base, Speaker, Recording, TranscriptSegment, TranscriptWord
from src.utils.error_handler import DatabaseError, ErrorSeverity

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Class for managing database operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses default path.
            
        Raises:
            DatabaseError: If database initialization fails
        """
        try:
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
            
            # Initialize database
            self._initialize_db()
            
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Failed to initialize database: {str(e)}",
                ErrorSeverity.CRITICAL,
                {"db_path": db_path, "error": str(e)}
            ) from e
        except Exception as e:
            raise DatabaseError(
                f"Unexpected error during database initialization: {str(e)}",
                ErrorSeverity.CRITICAL,
                {"db_path": db_path}
            ) from e
    
    def _initialize_db(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            # Create all tables if they don't exist
            Base.metadata.create_all(self.engine)
            logger.info("Database schema initialized")
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Failed to initialize database schema: {str(e)}",
                ErrorSeverity.CRITICAL
            ) from e
    
    def get_session(self) -> Session:
        """Get a new database session.
        
        Returns:
            Session: A new database session
            
        Raises:
            DatabaseError: If session creation fails
        """
        try:
            return self.Session()
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Failed to create database session: {str(e)}",
                ErrorSeverity.ERROR
            ) from e
    
    def close_session(self, session: Session):
        """Close a database session.
        
        Args:
            session: The session to close
            
        Raises:
            DatabaseError: If closing the session fails
        """
        try:
            session.close()
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Failed to close database session: {str(e)}",
                ErrorSeverity.ERROR
            ) from e
    
    def add_recording(self, session: Session, filename: str, path: str, duration: float,
                     sample_rate: int, channels: int, meta_data: Optional[Dict[str, Any]] = None) -> Recording:
        """Add a new recording to the database.
        
        Args:
            session: Database session
            filename: Name of the recording file
            path: Path to the recording file
            duration: Duration of the recording in seconds
            sample_rate: Sample rate of the recording
            channels: Number of audio channels
            meta_data: Additional metadata as a dictionary
            
        Returns:
            Recording: The newly created recording object
            
        Raises:
            DatabaseError: If the recording cannot be added
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
        except SQLAlchemyError as e:
            session.rollback()
            raise DatabaseError(
                f"Failed to add recording to database: {str(e)}",
                ErrorSeverity.ERROR,
                {"filename": filename, "path": path}
            ) from e
        except Exception as e:
            session.rollback()
            raise DatabaseError(
                f"Unexpected error adding recording: {str(e)}",
                ErrorSeverity.ERROR,
                {"filename": filename, "path": path}
            ) from e
    
    def get_recording(self, session: Session, recording_id: int) -> Optional[Recording]:
        """Get a recording by ID.
        
        Args:
            session: Database session
            recording_id: ID of the recording to retrieve
            
        Returns:
            Recording: The recording object, or None if not found
            
        Raises:
            DatabaseError: If an error occurs while retrieving the recording
        """
        try:
            return session.query(Recording).filter(Recording.id == recording_id).first()
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Failed to retrieve recording: {str(e)}",
                ErrorSeverity.ERROR,
                {"recording_id": recording_id}
            ) from e
    
    def get_all_recordings(self, session: Session) -> List[Recording]:
        """Get all recordings.
        
        Args:
            session: Database session
            
        Returns:
            List[Recording]: List of all recordings
            
        Raises:
            DatabaseError: If an error occurs while retrieving recordings
        """
        try:
            return session.query(Recording).all()
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Failed to retrieve recordings: {str(e)}",
                ErrorSeverity.ERROR
            ) from e
    
    def add_speaker(self, session: Session, name: Optional[str] = None,
                   voice_print: Optional[bytes] = None,
                   meta_data: Optional[Dict[str, Any]] = None) -> Speaker:
        """Add a new speaker to the database.
        
        Args:
            session: Database session
            name: Name of the speaker (optional)
            voice_print: Voice print data for the speaker (optional)
            meta_data: Additional metadata as a dictionary (optional)
            
        Returns:
            Speaker: The newly created speaker object
            
        Raises:
            DatabaseError: If the speaker cannot be added
        """
        try:
            # Generate a default name if none provided
            if name is None:
                # Count existing speakers with default names
                count = session.query(func.count(Speaker.id)).filter(
                    Speaker.name.like("Speaker %")
                ).scalar()
                
                # Create a new default name
                name = f"Speaker {count + 1}"
            
            speaker = Speaker(
                name=name,
                voice_print=voice_print,
                meta_data=meta_data
            )
            
            session.add(speaker)
            session.commit()
            
            logger.info(f"Added speaker: {name}")
            return speaker
        except SQLAlchemyError as e:
            session.rollback()
            raise DatabaseError(
                f"Failed to add speaker to database: {str(e)}",
                ErrorSeverity.ERROR,
                {"name": name}
            ) from e
        except Exception as e:
            session.rollback()
            raise DatabaseError(
                f"Unexpected error adding speaker: {str(e)}",
                ErrorSeverity.ERROR,
                {"name": name}
            ) from e
    
    def get_speaker(self, session: Session, speaker_id: int) -> Optional[Speaker]:
        """Get a speaker by ID.
        
        Args:
            session: Database session
            speaker_id: ID of the speaker to retrieve
            
        Returns:
            Speaker: The speaker object, or None if not found
            
        Raises:
            DatabaseError: If an error occurs while retrieving the speaker
        """
        try:
            return session.query(Speaker).filter(Speaker.id == speaker_id).first()
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Failed to retrieve speaker: {str(e)}",
                ErrorSeverity.ERROR,
                {"speaker_id": speaker_id}
            ) from e
    
    def get_all_speakers(self, session: Session) -> List[Speaker]:
        """Get all speakers.
        
        Args:
            session: Database session
            
        Returns:
            List[Speaker]: List of all speakers
            
        Raises:
            DatabaseError: If an error occurs while retrieving speakers
        """
        try:
            return session.query(Speaker).all()
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Failed to retrieve speakers: {str(e)}",
                ErrorSeverity.ERROR
            ) from e
    
    def add_transcript_segment(self, session: Session, recording_id: int,
                              speaker_id: Optional[int], start_time: float,
                              end_time: float, text: Optional[str] = None,
                              confidence: Optional[float] = None,
                              meta_data: Optional[Dict[str, Any]] = None) -> TranscriptSegment:
        """Add a new transcript segment to the database.
        
        Args:
            session: Database session
            recording_id: ID of the recording
            speaker_id: ID of the speaker (optional)
            start_time: Start time of the segment in seconds
            end_time: End time of the segment in seconds
            text: Transcript text (optional)
            confidence: Confidence score for the transcript (optional)
            meta_data: Additional metadata as a dictionary (optional)
            
        Returns:
            TranscriptSegment: The newly created transcript segment object
            
        Raises:
            DatabaseError: If the segment cannot be added
        """
        try:
            # Verify that the recording exists
            recording = self.get_recording(session, recording_id)
            if recording is None:
                raise DatabaseError(
                    f"Cannot add segment: Recording with ID {recording_id} not found",
                    ErrorSeverity.ERROR,
                    {"recording_id": recording_id}
                )
            
            # Verify that the speaker exists if specified
            if speaker_id is not None:
                speaker = self.get_speaker(session, speaker_id)
                if speaker is None:
                    raise DatabaseError(
                        f"Cannot add segment: Speaker with ID {speaker_id} not found",
                        ErrorSeverity.ERROR,
                        {"speaker_id": speaker_id}
                    )
            
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
            
            logger.debug(f"Added transcript segment: {start_time}-{end_time}s")
            return segment
        except DatabaseError:
            session.rollback()
            raise
        except SQLAlchemyError as e:
            session.rollback()
            raise DatabaseError(
                f"Failed to add transcript segment to database: {str(e)}",
                ErrorSeverity.ERROR,
                {
                    "recording_id": recording_id,
                    "speaker_id": speaker_id,
                    "start_time": start_time,
                    "end_time": end_time
                }
            ) from e
        except Exception as e:
            session.rollback()
            raise DatabaseError(
                f"Unexpected error adding transcript segment: {str(e)}",
                ErrorSeverity.ERROR,
                {
                    "recording_id": recording_id,
                    "speaker_id": speaker_id,
                    "start_time": start_time,
                    "end_time": end_time
                }
            ) from e
    
    def add_transcript_word(self, session: Session, segment_id: int,
                           text: str, start_time: float, end_time: float,
                           confidence: Optional[float] = None,
                           meta_data: Optional[Dict[str, Any]] = None) -> TranscriptWord:
        """Add a new transcript word to the database.
        
        Args:
            session: Database session
            segment_id: ID of the transcript segment
            text: Word text
            start_time: Start time of the word in seconds
            end_time: End time of the word in seconds
            confidence: Confidence score for the word (optional)
            meta_data: Additional metadata as a dictionary (optional)
            
        Returns:
            TranscriptWord: The newly created transcript word object
            
        Raises:
            DatabaseError: If the word cannot be added
        """
        try:
            # Verify that the segment exists
            segment = session.query(TranscriptSegment).filter(
                TranscriptSegment.id == segment_id
            ).first()
            
            if segment is None:
                raise DatabaseError(
                    f"Cannot add word: Segment with ID {segment_id} not found",
                    ErrorSeverity.ERROR,
                    {"segment_id": segment_id}
                )
            
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
            
            logger.debug(f"Added transcript word: '{text}' at {start_time}-{end_time}s")
            return word
        except DatabaseError:
            session.rollback()
            raise
        except SQLAlchemyError as e:
            session.rollback()
            raise DatabaseError(
                f"Failed to add transcript word to database: {str(e)}",
                ErrorSeverity.ERROR,
                {
                    "segment_id": segment_id,
                    "text": text,
                    "start_time": start_time,
                    "end_time": end_time
                }
            ) from e
        except Exception as e:
            session.rollback()
            raise DatabaseError(
                f"Unexpected error adding transcript word: {str(e)}",
                ErrorSeverity.ERROR,
                {
                    "segment_id": segment_id,
                    "text": text,
                    "start_time": start_time,
                    "end_time": end_time
                }
            ) from e
    
    def search_transcript(self, session: Session, query: str) -> List[Dict[str, Any]]:
        """Search for transcript segments containing the query text.
        
        Args:
            session: Database session
            query: Text to search for
            
        Returns:
            List[Dict[str, Any]]: List of matching segments with recording and speaker info
            
        Raises:
            DatabaseError: If the search fails
        """
        try:
            # Format the query for SQLite LIKE
            search_query = f"%{query}%"
            
            # Query for segments containing the search text
            results = session.query(
                TranscriptSegment, Recording, Speaker
            ).join(
                Recording, TranscriptSegment.recording_id == Recording.id
            ).outerjoin(
                Speaker, TranscriptSegment.speaker_id == Speaker.id
            ).filter(
                TranscriptSegment.text.like(search_query)
            ).all()
            
            # Format results
            formatted_results = []
            for segment, recording, speaker in results:
                formatted_results.append({
                    "segment_id": segment.id,
                    "recording_id": recording.id,
                    "recording_name": recording.filename,
                    "speaker_id": speaker.id if speaker else None,
                    "speaker_name": speaker.name if speaker else "Unknown",
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "text": segment.text,
                    "confidence": segment.confidence
                })
            
            return formatted_results
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Failed to search transcript: {str(e)}",
                ErrorSeverity.ERROR,
                {"query": query}
            ) from e
        except Exception as e:
            raise DatabaseError(
                f"Unexpected error during transcript search: {str(e)}",
                ErrorSeverity.ERROR,
                {"query": query}
            ) from e
    
    def get_speaker_statistics(self, session: Session, speaker_id: int) -> Dict[str, Any]:
        """Get statistics for a speaker.
        
        Args:
            session: Database session
            speaker_id: ID of the speaker
            
        Returns:
            Dict[str, Any]: Dictionary containing speaker statistics
            
        Raises:
            DatabaseError: If the statistics cannot be retrieved
        """
        try:
            # Verify that the speaker exists
            speaker = self.get_speaker(session, speaker_id)
            if speaker is None:
                raise DatabaseError(
                    f"Cannot get statistics: Speaker with ID {speaker_id} not found",
                    ErrorSeverity.ERROR,
                    {"speaker_id": speaker_id}
                )
            
            # Get segments for the speaker
            segments = session.query(TranscriptSegment).filter(
                TranscriptSegment.speaker_id == speaker_id
            ).all()
            
            # Calculate statistics
            total_segments = len(segments)
            total_duration = sum(segment.end_time - segment.start_time for segment in segments)
            total_words = session.query(func.count(TranscriptWord.id)).join(
                TranscriptSegment, TranscriptWord.segment_id == TranscriptSegment.id
            ).filter(
                TranscriptSegment.speaker_id == speaker_id
            ).scalar()
            
            # Get unique recordings
            unique_recordings = set(segment.recording_id for segment in segments)
            
            return {
                "speaker_id": speaker_id,
                "speaker_name": speaker.name,
                "total_segments": total_segments,
                "total_duration": total_duration,
                "total_words": total_words,
                "recordings_count": len(unique_recordings)
            }
        except DatabaseError:
            raise
        except SQLAlchemyError as e:
            raise DatabaseError(
                f"Failed to get speaker statistics: {str(e)}",
                ErrorSeverity.ERROR,
                {"speaker_id": speaker_id}
            ) from e
        except Exception as e:
            raise DatabaseError(
                f"Unexpected error getting speaker statistics: {str(e)}",
                ErrorSeverity.ERROR,
                {"speaker_id": speaker_id}
            ) from e 