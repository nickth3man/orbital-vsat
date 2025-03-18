"""
Configuration module for VSAT.

This module provides application-wide configuration settings and constants.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """Application configuration class."""

    # Application paths
    APP_DIR = Path.home() / ".vsat"
    DATA_DIR = APP_DIR / "data"
    MODELS_DIR = APP_DIR / "models"
    LOGS_DIR = APP_DIR / "logs"

    # Database settings
    DATABASE_PATH = DATA_DIR / "vsat.db"

    # Audio processing settings
    DEFAULT_SAMPLE_RATE = 16000
    AUDIO_CHUNK_SIZE = 30  # seconds
    AUDIO_CHUNK_OVERLAP = 5  # seconds

    # Speaker identification settings
    MIN_SPEAKER_CONFIDENCE = 0.7
    VOICE_PRINT_DIMENSION = 192

    # UI settings
    WAVEFORM_HEIGHT = 150
    TIMELINE_HEIGHT = 50

    @classmethod
    def initialize(cls):
        """Initialize configuration and create necessary directories."""
        # Create application directories if they don't exist
        os.makedirs(cls.APP_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)

        logger.info(f"Application directories initialized at {cls.APP_DIR}")

        return cls
