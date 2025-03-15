"""
Demo script for Content Analysis functionality.

This script demonstrates the content analysis capabilities of VSAT with an interactive UI.
"""

import sys
import os
import logging
from pathlib import Path
import json
import tempfile

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QPushButton, QFileDialog, QLabel, QSplitter, QTabWidget, 
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QToolBar, QStatusBar, QScrollArea, QGridLayout, QGroupBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QColor, QTextCharFormat, QFont

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.ml.content_analysis import ContentAnalyzer
from src.transcription.whisper_transcriber import WhisperTranscriber
from src.audio.processor import AudioProcessor
from src.audio.audio_player import AudioPlayer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentAnalysisDemo(QMainWindow):
    """Main window for the Content Analysis demo application."""
    
    def __init__(self):
        """Initialize the demo window."""
        super().__init__()
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.transcriber = WhisperTranscriber(model_size="base", device="cpu")
        self.content_analyzer = ContentAnalyzer()
        self.audio_player = AudioPlayer()
        
        # Current file and analysis results
        self.current_file = None
        self.transcript_segments = []
        self.analysis_results = None
        self.current_segment = None
        
        # Set up UI
        self.setWindowTitle("VSAT - Content Analysis Demo")
        self.setGeometry(100, 100, 1200, 800)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        main_layout = QVBoxLayout(central_widget)
        
        # Add title
        title_label = QLabel("Content Analysis Demo")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # Create toolbar
        self._create_toolbar()
        
        # Create main content area with splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top section: Text area and controls
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        
        # Text display
        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setPlaceholderText("Transcript will appear here")
        top_layout.addWidget(self.transcript_text)
        
        # Process controls
        process_group = QGroupBox("Processing Controls")
        process_layout = QHBoxLayout(process_group)
        
        self.transcribe_button = QPushButton("Transcribe Audio")
        self.transcribe_button.clicked.connect(self._transcribe_audio)
        self.transcribe_button.setEnabled(False)
        process_layout.addWidget(self.transcribe_button)
        
        self.analyze_button = QPushButton("Analyze Content")
        self.analyze_button.clicked.connect(self._analyze_content)
        self.analyze_button.setEnabled(False)
        process_layout.addWidget(self.analyze_button)
        
        top_layout.addWidget(process_group)
        
        # Add audio controls
        audio_group = QGroupBox("Audio Controls")
        audio_layout = QHBoxLayout(audio_group)
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_playback)
        self.play_button.setEnabled(False)
        audio_layout.addWidget(self.play_button)
        
        self.play_segment_button = QPushButton("Play Selected Segment")
        self.play_segment_button.clicked.connect(self._play_selected_segment)
        self.play_segment_button.setEnabled(False)
        audio_layout.addWidget(self.play_segment_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_playback)
        self.stop_button.setEnabled(False)
        audio_layout.addWidget(self.stop_button)
        
        top_layout.addWidget(audio_group)
        
        splitter.addWidget(top_widget)
        
        # Bottom section: Analysis results
        bottom_widget = QTabWidget()
        
        # Summary tab
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Summary will appear here")
        summary_layout.addWidget(self.summary_text)
        
        bottom_widget.addTab(self.summary_tab, "Summary")
        
        # Keywords tab
        self.keywords_tab = QWidget()
        keywords_layout = QVBoxLayout(self.keywords_tab)
        
        self.keywords_table = QTableWidget(0, 2)
        self.keywords_table.setHorizontalHeaderLabels(["Keyword", "Score"])
        self.keywords_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        keywords_layout.addWidget(self.keywords_table)
        
        bottom_widget.addTab(self.keywords_tab, "Keywords")
        
        # Topics tab
        self.topics_tab = QWidget()
        topics_layout = QVBoxLayout(self.topics_tab)
        
        self.topics_table = QTableWidget(0, 2)
        self.topics_table.setHorizontalHeaderLabels(["Topic ID", "Topic Words"])
        self.topics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        topics_layout.addWidget(self.topics_table)
        
        bottom_widget.addTab(self.topics_tab, "Topics")
        
        # Important moments tab
        self.moments_tab = QWidget()
        moments_layout = QVBoxLayout(self.moments_tab)
        
        self.moments_table = QTableWidget(0, 3)
        self.moments_table.setHorizontalHeaderLabels(["Time", "Speaker", "Text"])
        self.moments_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.moments_table.cellClicked.connect(self._moment_selected)
        moments_layout.addWidget(self.moments_table)
        
        bottom_widget.addTab(self.moments_tab, "Important Moments")
        
        # Sentiment tab
        self.sentiment_tab = QWidget()
        sentiment_layout = QVBoxLayout(self.sentiment_tab)
        
        sentiment_summary_group = QGroupBox("Overall Sentiment")
        sentiment_summary_layout = QGridLayout(sentiment_summary_group)
        
        sentiment_summary_layout.addWidget(QLabel("Label:"), 0, 0)
        self.sentiment_label = QLabel("N/A")
        sentiment_summary_layout.addWidget(self.sentiment_label, 0, 1)
        
        sentiment_summary_layout.addWidget(QLabel("Compound Score:"), 1, 0)
        self.sentiment_compound = QLabel("N/A")
        sentiment_summary_layout.addWidget(self.sentiment_compound, 1, 1)
        
        sentiment_summary_layout.addWidget(QLabel("Positive:"), 2, 0)
        self.sentiment_positive = QLabel("N/A")
        sentiment_summary_layout.addWidget(self.sentiment_positive, 2, 1)
        
        sentiment_summary_layout.addWidget(QLabel("Neutral:"), 3, 0)
        self.sentiment_neutral = QLabel("N/A")
        sentiment_summary_layout.addWidget(self.sentiment_neutral, 3, 1)
        
        sentiment_summary_layout.addWidget(QLabel("Negative:"), 4, 0)
        self.sentiment_negative = QLabel("N/A")
        sentiment_summary_layout.addWidget(self.sentiment_negative, 4, 1)
        
        sentiment_layout.addWidget(sentiment_summary_group)
        
        sentiment_detail_group = QGroupBox("Sentiment by Segment")
        sentiment_detail_layout = QVBoxLayout(sentiment_detail_group)
        
        self.sentiment_table = QTableWidget(0, 3)
        self.sentiment_table.setHorizontalHeaderLabels(["Time", "Sentiment", "Text"])
        self.sentiment_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.sentiment_table.cellClicked.connect(self._sentiment_segment_selected)
        sentiment_detail_layout.addWidget(self.sentiment_table)
        
        sentiment_layout.addWidget(sentiment_detail_group)
        
        bottom_widget.addTab(self.sentiment_tab, "Sentiment")
        
        splitter.addWidget(bottom_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 400])
        
        main_layout.addWidget(splitter)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load an audio file to begin.")
        
        # Connect audio player signals
        self.audio_player.playback_state_changed.connect(self._update_playback_ui)
    
    def _create_toolbar(self):
        """Create the application toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Load file action
        load_action = QAction("Load Audio", self)
        load_action.triggered.connect(self._load_audio_file)
        toolbar.addAction(load_action)
        
        # Export transcript action
        export_action = QAction("Export Results", self)
        export_action.triggered.connect(self._export_results)
        export_action.setEnabled(False)
        self.export_action = export_action
        toolbar.addAction(export_action)
        
        toolbar.addSeparator()
        
        # Help action
        help_action = QAction("Help", self)
        help_action.triggered.connect(self._show_help)
        toolbar.addAction(help_action)
    
    def _load_audio_file(self):
        """Load an audio file for processing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        
        if file_path:
            try:
                logger.info(f"Loading audio file: {file_path}")
                self.current_file = file_path
                
                # Clear previous data
                self.transcript_segments = []
                self.analysis_results = None
                self.transcript_text.clear()
                self._clear_analysis_displays()
                
                # Enable controls
                self.play_button.setEnabled(True)
                self.transcribe_button.setEnabled(True)
                
                # Disable analysis button until transcription is done
                self.analyze_button.setEnabled(False)
                
                # Load the file in the audio player
                self.audio_player.load_file(file_path)
                
                # Update status
                file_name = os.path.basename(file_path)
                duration = self.audio_player.get_duration()
                self.status_bar.showMessage(f"Loaded: {file_name} ({duration:.2f} seconds)")
            except Exception as e:
                logger.error(f"Error loading file: {e}")
                self.status_bar.showMessage(f"Error loading file: {str(e)}")
    
    def _transcribe_audio(self):
        """Transcribe the loaded audio file."""
        if not self.current_file:
            return
        
        self.status_bar.showMessage("Transcribing audio... This may take a while.")
        self.setCursor(Qt.CursorShape.WaitCursor)
        self.transcribe_button.setEnabled(False)
        
        try:
            # Transcribe the audio
            self.transcript_segments = self.transcriber.transcribe_file(
                self.current_file,
                word_timestamps=True
            )
            
            # Display the transcript
            self._display_transcript()
            
            # Enable the analyze button
            self.analyze_button.setEnabled(True)
            
            self.status_bar.showMessage("Transcription complete")
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            self.status_bar.showMessage(f"Error transcribing audio: {str(e)}")
        finally:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.transcribe_button.setEnabled(True)
    
    def _analyze_content(self):
        """Analyze the transcript content."""
        if not self.transcript_segments:
            return
        
        self.status_bar.showMessage("Analyzing content...")
        self.setCursor(Qt.CursorShape.WaitCursor)
        self.analyze_button.setEnabled(False)
        
        try:
            # Analyze the content
            self.analysis_results = self.content_analyzer.analyze_content(self.transcript_segments)
            
            # Display the analysis results
            self._display_analysis_results()
            
            # Enable export
            self.export_action.setEnabled(True)
            
            self.status_bar.showMessage("Content analysis complete")
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            self.status_bar.showMessage(f"Error analyzing content: {str(e)}")
        finally:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.analyze_button.setEnabled(True)
    
    def _display_transcript(self):
        """Display the transcript in the text area."""
        self.transcript_text.clear()
        for segment in self.transcript_segments:
            start_time = segment.get('start', 0)
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '')
            
            # Format: [00:01.23] Speaker: Text
            time_str = self._format_time(start_time)
            self.transcript_text.append(f"<b>[{time_str}] {speaker}:</b> {text}")
            self.transcript_text.append("")
    
    def _display_analysis_results(self):
        """Display the content analysis results."""
        if not self.analysis_results:
            return
        
        # Display summary
        summary = self.analysis_results.get('summary', '')
        self.summary_text.setText(summary)
        
        # Display keywords
        keywords = self.analysis_results.get('keywords', [])
        self.keywords_table.setRowCount(len(keywords))
        for i, (keyword, score) in enumerate(keywords):
            self.keywords_table.setItem(i, 0, QTableWidgetItem(keyword))
            self.keywords_table.setItem(i, 1, QTableWidgetItem(f"{score:.4f}"))
        
        # Display topics
        topics = self.analysis_results.get('topics', [])
        self.topics_table.setRowCount(len(topics))
        for i, topic in enumerate(topics):
            topic_id = topic.get('id', i)
            topic_words = ', '.join([word for word, _ in topic.get('words', [])])
            
            self.topics_table.setItem(i, 0, QTableWidgetItem(str(topic_id)))
            self.topics_table.setItem(i, 1, QTableWidgetItem(topic_words))
        
        # Display important moments
        moments = self.analysis_results.get('important_moments', [])
        self.moments_table.setRowCount(len(moments))
        for i, moment in enumerate(moments):
            segment = moment.get('segment', {})
            start_time = segment.get('start', 0)
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '')
            
            time_item = QTableWidgetItem(self._format_time(start_time))
            speaker_item = QTableWidgetItem(speaker)
            text_item = QTableWidgetItem(text)
            
            self.moments_table.setItem(i, 0, time_item)
            self.moments_table.setItem(i, 1, speaker_item)
            self.moments_table.setItem(i, 2, text_item)
        
        # Display sentiment
        sentiment = self.analysis_results.get('sentiment_analysis', {})
        overall = sentiment.get('overall', {})
        
        # Update overall sentiment display
        self.sentiment_label.setText(overall.get('label', 'N/A'))
        self.sentiment_compound.setText(f"{overall.get('compound', 0):.4f}")
        self.sentiment_positive.setText(f"{overall.get('pos', 0):.4f}")
        self.sentiment_neutral.setText(f"{overall.get('neu', 0):.4f}")
        self.sentiment_negative.setText(f"{overall.get('neg', 0):.4f}")
        
        # Set label color based on sentiment
        label = overall.get('label', '')
        if 'Negative' in label:
            self.sentiment_label.setStyleSheet("color: red;")
        elif 'Positive' in label:
            self.sentiment_label.setStyleSheet("color: green;")
        else:
            self.sentiment_label.setStyleSheet("")
        
        # Display segment sentiment
        by_segment = sentiment.get('by_segment', [])
        self.sentiment_table.setRowCount(len(by_segment))
        for i, segment in enumerate(by_segment):
            start_time = segment.get('start', 0)
            sentiment_label = segment.get('sentiment_label', 'N/A')
            text = segment.get('text', '')
            
            time_item = QTableWidgetItem(self._format_time(start_time))
            sentiment_item = QTableWidgetItem(sentiment_label)
            text_item = QTableWidgetItem(text)
            
            # Color based on sentiment
            if 'Negative' in sentiment_label:
                sentiment_item.setForeground(QColor(255, 0, 0))  # Red
            elif 'Positive' in sentiment_label:
                sentiment_item.setForeground(QColor(0, 128, 0))  # Green
            
            self.sentiment_table.setItem(i, 0, time_item)
            self.sentiment_table.setItem(i, 1, sentiment_item)
            self.sentiment_table.setItem(i, 2, text_item)
    
    def _clear_analysis_displays(self):
        """Clear all analysis displays."""
        self.summary_text.clear()
        self.keywords_table.setRowCount(0)
        self.topics_table.setRowCount(0)
        self.moments_table.setRowCount(0)
        self.sentiment_table.setRowCount(0)
        
        self.sentiment_label.setText("N/A")
        self.sentiment_compound.setText("N/A")
        self.sentiment_positive.setText("N/A")
        self.sentiment_neutral.setText("N/A")
        self.sentiment_negative.setText("N/A")
        self.sentiment_label.setStyleSheet("")
        
        self.export_action.setEnabled(False)
    
    def _moment_selected(self, row, column):
        """Handle selection of an important moment."""
        if row < 0 or row >= self.moments_table.rowCount():
            return
        
        time_str = self.moments_table.item(row, 0).text()
        speaker = self.moments_table.item(row, 1).text()
        text = self.moments_table.item(row, 2).text()
        
        # Find the corresponding segment
        moments = self.analysis_results.get('important_moments', [])
        if row < len(moments):
            self.current_segment = moments[row].get('segment', {})
            self.play_segment_button.setEnabled(True)
            
            # Update status
            self.status_bar.showMessage(f"Selected moment: [{time_str}] {speaker}: {text}")
    
    def _sentiment_segment_selected(self, row, column):
        """Handle selection of a sentiment segment."""
        if row < 0 or row >= self.sentiment_table.rowCount():
            return
        
        time_str = self.sentiment_table.item(row, 0).text()
        sentiment = self.sentiment_table.item(row, 1).text()
        text = self.sentiment_table.item(row, 2).text()
        
        # Find the corresponding segment
        segments = self.analysis_results.get('sentiment_analysis', {}).get('by_segment', [])
        if row < len(segments):
            self.current_segment = segments[row]
            self.play_segment_button.setEnabled(True)
            
            # Update status
            self.status_bar.showMessage(f"Selected segment: [{time_str}] {sentiment}: {text}")
    
    def _toggle_playback(self):
        """Toggle audio playback."""
        if not self.current_file:
            return
        
        if self.audio_player.is_playing():
            self.audio_player.pause()
        else:
            self.audio_player.play(self.current_file)
            self.status_bar.showMessage("Playing audio...")
    
    def _stop_playback(self):
        """Stop audio playback."""
        self.audio_player.stop()
        self.status_bar.showMessage("Playback stopped")
    
    def _play_selected_segment(self):
        """Play the currently selected segment."""
        if not self.current_segment or not self.current_file:
            return
        
        start = self.current_segment.get('start', 0)
        end = self.current_segment.get('end', 0)
        
        if end > start:
            self.audio_player.play(self.current_file, start, end)
            self.status_bar.showMessage(f"Playing segment: {self._format_time(start)} - {self._format_time(end)}")
    
    def _update_playback_ui(self, is_playing):
        """Update UI based on playback state."""
        self.play_button.setText("Pause" if is_playing else "Play")
        self.stop_button.setEnabled(is_playing)
    
    def _export_results(self):
        """Export analysis results to a file."""
        if not self.analysis_results:
            return
        
        # Ask for export file name
        export_file, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Results",
            "",
            "JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
        )
        
        if not export_file:
            return
        
        try:
            # Prepare data for export
            export_data = {
                'summary': self.analysis_results.get('summary', ''),
                'keywords': self.analysis_results.get('keywords', []),
                'topics': [{
                    'id': topic.get('id', i),
                    'words': topic.get('words', [])
                } for i, topic in enumerate(self.analysis_results.get('topics', []))],
                'important_moments': [{
                    'time': self._format_time(moment.get('segment', {}).get('start', 0)),
                    'speaker': moment.get('segment', {}).get('speaker', 'Unknown'),
                    'text': moment.get('segment', {}).get('text', ''),
                    'score': moment.get('score', 0)
                } for moment in self.analysis_results.get('important_moments', [])],
                'sentiment': {
                    'overall': self.analysis_results.get('sentiment_analysis', {}).get('overall', {}),
                    'by_segment': [{
                        'time': self._format_time(segment.get('start', 0)),
                        'text': segment.get('text', ''),
                        'sentiment': segment.get('sentiment_label', ''),
                        'scores': segment.get('sentiment', {})
                    } for segment in self.analysis_results.get('sentiment_analysis', {}).get('by_segment', [])]
                }
            }
            
            # Export as JSON
            if export_file.lower().endswith('.json'):
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=2)
            else:
                # Export as formatted text
                with open(export_file, 'w') as f:
                    f.write("=== CONTENT ANALYSIS RESULTS ===\n\n")
                    
                    f.write("== SUMMARY ==\n")
                    f.write(export_data['summary'])
                    f.write("\n\n")
                    
                    f.write("== KEYWORDS ==\n")
                    for keyword, score in export_data['keywords']:
                        f.write(f"- {keyword} ({score:.4f})\n")
                    f.write("\n")
                    
                    f.write("== TOPICS ==\n")
                    for topic in export_data['topics']:
                        f.write(f"Topic {topic['id']}:\n")
                        words_str = ', '.join([word for word, _ in topic['words']])
                        f.write(f"  {words_str}\n")
                    f.write("\n")
                    
                    f.write("== IMPORTANT MOMENTS ==\n")
                    for moment in export_data['important_moments']:
                        f.write(f"[{moment['time']}] {moment['speaker']}: {moment['text']}\n")
                        f.write(f"  Importance Score: {moment['score']:.4f}\n")
                    f.write("\n")
                    
                    f.write("== SENTIMENT ANALYSIS ==\n")
                    overall = export_data['sentiment']['overall']
                    f.write(f"Overall Sentiment: {overall.get('label', 'N/A')}\n")
                    f.write(f"  Compound: {overall.get('compound', 0):.4f}\n")
                    f.write(f"  Positive: {overall.get('pos', 0):.4f}\n")
                    f.write(f"  Neutral: {overall.get('neu', 0):.4f}\n")
                    f.write(f"  Negative: {overall.get('neg', 0):.4f}\n\n")
                    
                    f.write("Sentiment by Segment:\n")
                    for segment in export_data['sentiment']['by_segment']:
                        f.write(f"[{segment['time']}] {segment['sentiment']}: {segment['text']}\n")
            
            self.status_bar.showMessage(f"Results exported to {export_file}")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            self.status_bar.showMessage(f"Error exporting results: {str(e)}")
    
    def _show_help(self):
        """Show help information."""
        help_text = (
            "<h3>VSAT Content Analysis Demo</h3>"
            "<p>This demo shows how to analyze the content of audio transcripts.</p>"
            "<h4>Instructions:</h4>"
            "<ol>"
            "<li>Load an audio file using the toolbar</li>"
            "<li>Click 'Transcribe Audio' to generate a transcript</li>"
            "<li>Click 'Analyze Content' to analyze the transcript content</li>"
            "<li>Browse the different tabs to view analysis results:</li>"
            "<ul>"
            "<li><b>Summary:</b> Automatic text summarization</li>"
            "<li><b>Keywords:</b> Important terms extracted from the transcript</li>"
            "<li><b>Topics:</b> Main topics discussed in the audio</li>"
            "<li><b>Important Moments:</b> Key segments based on content analysis</li>"
            "<li><b>Sentiment:</b> Emotional tone analysis of the content</li>"
            "</ul>"
            "<li>Click on segments in the tables to select them for playback</li>"
            "<li>Use the audio controls to play the entire file or selected segments</li>"
            "<li>Export the analysis results using the 'Export Results' button</li>"
            "</ol>"
        )
        
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Help", help_text)
    
    def _format_time(self, seconds):
        """Format time in seconds to [MM:SS.ms] format."""
        minutes = int(seconds / 60)
        seconds_remainder = seconds % 60
        return f"{minutes:02d}:{seconds_remainder:06.3f}"
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop playback if playing
        if self.audio_player.is_playing():
            self.audio_player.stop()
        
        # Accept the close event
        event.accept()

def main():
    """Run the content analysis demo application."""
    app = QApplication(sys.argv)
    window = ContentAnalysisDemo()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 