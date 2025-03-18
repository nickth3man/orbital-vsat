"""
Audio processing components for VSAT.

This package contains modules for audio file handling, processing, and playback.
"""

from .file_handler import AudioFileHandler
from .processor import AudioProcessor
from .chunked_processor import ChunkedProcessor, ChunkingError
from .audio_preprocessor import AudioPreprocessor

# Comment out problematic imports for now to allow tests to run
# from .audio_player import AudioPlayer
# from .segment_player import SegmentPlayer
# from .player_config import PlayerConfig
# from .player_events import PlayerEvent
# from .player_signals import PlayerSignals
# from .player_state import PlayerState
# from .playback_controller import PlaybackController
# from .volume_controller import VolumeController

__all__ = [
    'AudioFileHandler',
    'AudioProcessor',
    'ChunkedProcessor',
    'ChunkingError',
    'AudioPreprocessor'
    # 'AudioPlayer',
    # 'SegmentPlayer',
    # 'PlayerConfig',
    # 'PlayerEvent',
    # 'PlayerSignals',
    # 'PlayerState',
    # 'PlaybackController',
    # 'VolumeController',
]