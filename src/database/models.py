"""
Database models for VSAT.

This module defines the SQLAlchemy ORM models for the application's database.
"""

import json
import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime, Text, LargeBinary, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Speaker(Base):
    """Speaker model representing a unique speaker in the database."""
    
    __tablename__ = 'speakers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow)
    voice_print = Column(LargeBinary, nullable=True)  # Store speaker embedding
    meta_data = Column(JSON, nullable=True)
    
    # Relationships
    segments = relationship("TranscriptSegment", back_populates="speaker")
    
    def __repr__(self):
        return f"<Speaker(id={self.id}, name='{self.name}')>"
    
    @property
    def total_speaking_time(self) -> float:
        """Calculate total speaking time for this speaker across all recordings."""
        return sum((segment.end_time - segment.start_time) for segment in self.segments)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the speaker to a dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'meta_data': self.meta_data,
            'total_speaking_time': self.total_speaking_time
        }


class Recording(Base):
    """Recording model representing an audio recording in the database."""
    
    __tablename__ = 'recordings'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    path = Column(String(1024), nullable=False)
    duration = Column(Float, nullable=False)
    sample_rate = Column(Integer, nullable=False)
    channels = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    processed = Column(Boolean, default=False)
    meta_data = Column(JSON, nullable=True)
    
    # Relationships
    segments = relationship("TranscriptSegment", back_populates="recording", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Recording(id={self.id}, filename='{self.filename}')>"
    
    @property
    def speakers(self) -> List[Speaker]:
        """Get all unique speakers in this recording."""
        return list(set(segment.speaker for segment in self.segments if segment.speaker))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the recording to a dictionary."""
        return {
            'id': self.id,
            'filename': self.filename,
            'path': self.path,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'processed': self.processed,
            'meta_data': self.meta_data,
            'speaker_count': len(self.speakers)
        }


class TranscriptSegment(Base):
    """TranscriptSegment model representing a segment of speech in a recording."""
    
    __tablename__ = 'transcript_segments'
    
    id = Column(Integer, primary_key=True)
    recording_id = Column(Integer, ForeignKey('recordings.id'), nullable=False)
    speaker_id = Column(Integer, ForeignKey('speakers.id'), nullable=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    meta_data = Column(JSON, nullable=True)
    
    # Relationships
    recording = relationship("Recording", back_populates="segments")
    speaker = relationship("Speaker", back_populates="segments")
    words = relationship("TranscriptWord", back_populates="segment", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<TranscriptSegment(id={self.id}, start={self.start_time:.2f}, end={self.end_time:.2f})>"
    
    @property
    def duration(self) -> float:
        """Calculate the duration of this segment."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the segment to a dictionary."""
        return {
            'id': self.id,
            'recording_id': self.recording_id,
            'speaker_id': self.speaker_id,
            'speaker_name': self.speaker.name if self.speaker else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'text': self.text,
            'confidence': self.confidence,
            'meta_data': self.meta_data,
            'word_count': len(self.words)
        }


class TranscriptWord(Base):
    """TranscriptWord model representing a single word in a transcript segment."""
    
    __tablename__ = 'transcript_words'
    
    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey('transcript_segments.id'), nullable=False)
    text = Column(String(255), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    meta_data = Column(JSON, nullable=True)
    
    # Relationships
    segment = relationship("TranscriptSegment", back_populates="words")
    
    def __repr__(self):
        return f"<TranscriptWord(id={self.id}, text='{self.text}', start={self.start_time:.2f})>"
    
    @property
    def duration(self) -> float:
        """Calculate the duration of this word."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the word to a dictionary."""
        return {
            'id': self.id,
            'segment_id': self.segment_id,
            'text': self.text,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'confidence': self.confidence,
            'meta_data': self.meta_data
        } 