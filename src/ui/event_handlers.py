"""
Event handling module for VSAT.

This module provides event and signal handling functionality.
"""

import logging
from typing import Dict, Any, List, Optional
from PyQt6.QtCore import QObject, pyqtSlot

logger = logging.getLogger(__name__)


class EventHandler(QObject):
    """Handles events and signals for the main window."""
    
    def __init__(self, main_window):
        """Initialize the event handler.
        
        Args:
            main_window: The parent MainWindow instance
        """
        super().__init__(main_window)
        self.main_window = main_window
        
        logger.debug("Event handler initialized")
        
    def connect_signals(self):
        """Connect signals to their handlers."""
        # Get UI components
        ui = self.main_window.ui_components
        waveform_view = ui.get_waveform_view()
        transcript_view = ui.get_transcript_view()
        search_panel = ui.get_search_panel()
        content_analysis_panel = ui.get_content_analysis_panel()
        
        # Connect waveform view signals
        waveform_view.positionClicked.connect(self.on_position_clicked)
        waveform_view.rangeSelected.connect(self.on_range_selected)
        
        # Connect transcript view signals
        transcript_view.wordClicked.connect(self.on_word_clicked)
        transcript_view.wordsSelected.connect(self.on_words_selected)
        
        # Connect search panel signals
        search_panel.searchRequested.connect(self.on_search_requested)
        search_panel.resultSelected.connect(self.on_search_result_selected)
        
        # Connect content analysis panel signals
        content_analysis_panel.important_moment_selected.connect(self.on_important_moment_selected)
        
        # Connect audio player signals
        audio_player = self.main_window.audio_player
        audio_player.position_changed.connect(self.on_playback_position_changed)
        audio_player.playback_state_changed.connect(self.on_playback_state_changed)
        
        logger.debug("Signals connected")
        
    @pyqtSlot(float)
    def on_position_clicked(self, position):
        """Handle click on a position in the waveform.
        
        Args:
            position: Position in seconds
        """
        if not self.main_window.audio_player:
            return
            
        # Set position in audio player
        self.main_window.audio_player.set_position(position)
        
        logger.debug(f"Position clicked: {position}")
        
    @pyqtSlot(float, float)
    def on_range_selected(self, start, end):
        """Handle selection of a range in the waveform.
        
        Args:
            start: Start position in seconds
            end: End position in seconds
        """
        if not self.main_window.audio_player:
            return
            
        # Play the selected range
        self.main_window.audio_player.play_segment(start, end)
        
        logger.debug(f"Range selected: {start} - {end}")
        
    @pyqtSlot(dict)
    def on_word_clicked(self, word):
        """Handle click on a word in the transcript.
        
        Args:
            word: Word data dictionary
        """
        if not self.main_window.audio_player:
            return
            
        # Play the word
        self.main_window.audio_player.play_word(word)
        
        logger.debug(f"Word clicked: {word.get('text', '')}")
        
    @pyqtSlot(list)
    def on_words_selected(self, words):
        """Handle selection of words in the transcript.
        
        Args:
            words: List of word data dictionaries
        """
        if not words or not self.main_window.audio_player:
            return
            
        # Get start and end times
        start = words[0].get('start', 0)
        end = words[-1].get('end', 0)
        
        # Play the selected range
        self.main_window.audio_player.play_segment(start, end)
        
        logger.debug(f"Words selected: {len(words)} words")
        
    @pyqtSlot(str, object)
    def on_search_requested(self, query, options=None):
        """Handle search request from the search panel.
        
        Args:
            query: Search query string
            options: Optional dictionary with search options
        """
        if not query or not self.main_window.segments:
            return
            
        results = []
        
        # Search through segments
        for segment_idx, segment in enumerate(self.main_window.segments):
            words = segment.get('words', [])
            
            for word_idx, word in enumerate(words):
                if query.lower() in word.get('text', '').lower():
                    # Get context around the word
                    context = self._get_context(segment, word_idx)
                    
                    # Add result
                    results.append({
                        'segment_index': segment_idx,
                        'word_index': word_idx,
                        'word': word,
                        'context': context
                    })
        
        # Update search panel with results
        self.main_window.ui_components.get_search_panel().set_results(results)
        
        logger.debug(f"Search requested: {query}, found {len(results)} results")
        
    def _get_context(self, segment, word_index, context_size=3):
        """Get context around a word in a segment.
        
        Args:
            segment: Segment containing the word
            word_index: Index of the word in the segment
            context_size: Number of words to include before and after
            
        Returns:
            Context string
        """
        words = segment.get('words', [])
        
        # Calculate context range
        start_idx = max(0, word_index - context_size)
        end_idx = min(len(words) - 1, word_index + context_size)
        
        # Extract words in context
        context_words = words[start_idx:end_idx + 1]
        context_text = ' '.join(word.get('text', '') for word in context_words)
        
        # Highlight the target word
        target_word_idx = word_index - start_idx
        if 0 <= target_word_idx < len(context_words):
            # Add highlighting to the target word
            context_words_list = [word.get('text', '') for word in context_words]
            context_words_list[target_word_idx] = f"**{context_words_list[target_word_idx]}**"
            context_text = ' '.join(context_words_list)
        
        return context_text
        
    @pyqtSlot(dict)
    def on_search_result_selected(self, result):
        """Handle selection of a search result.
        
        Args:
            result: Search result dictionary
        """
        if not result or not self.main_window.audio_player:
            return
            
        # Get the word
        word = result.get('word', {})
        
        # Play the word
        if word:
            self.main_window.audio_player.play_word(word)
        
        # Highlight the word in the transcript
        segment_idx = result.get('segment_index', -1)
        word_idx = result.get('word_index', -1)
        
        if segment_idx >= 0 and word_idx >= 0:
            self.main_window.ui_components.get_transcript_view().highlight_word(segment_idx, word_idx)
        
        logger.debug(f"Search result selected: {word.get('text', '')}")
        
    @pyqtSlot(dict)
    def on_important_moment_selected(self, moment):
        """Handle selection of an important moment.
        
        Args:
            moment: Moment data dictionary
        """
        if not moment or not self.main_window.audio_player:
            return
            
        # Get the timestamp
        timestamp = moment.get('timestamp', 0)
        
        # Set position in audio player
        self.main_window.audio_player.set_position(timestamp)
        
        logger.debug(f"Important moment selected: {moment.get('text', '')}")
        
    @pyqtSlot(float)
    def on_playback_position_changed(self, position):
        """Handle playback position changes.
        
        Args:
            position: Position in seconds
        """
        # Update waveform view
        self.main_window.ui_components.get_waveform_view().set_position(position)
        
        # Update transcript view
        self.main_window.ui_components.get_transcript_view().highlight_at_position(position)
        
    @pyqtSlot(bool)
    def on_playback_state_changed(self, is_playing):
        """Handle playback state changes.
        
        Args:
            is_playing: Whether audio is playing
        """
        # Update play button icon
        self.main_window.menu_manager.update_play_action_icon(is_playing)
