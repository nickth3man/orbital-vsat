"""
Content analysis panel for displaying topics, keywords, summaries, and important moments.
"""

import logging
from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QProgressBar, QSplitter, QMessageBox,
    QListWidget, QListWidgetItem, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtGui import QColor, QFont, QBrush
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from src.ml.content_analysis import ContentAnalyzer, ContentAnalysisError
from src.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class ContentAnalysisPanel(QWidget):
    """Panel for displaying content analysis results."""
    
    # Signals
    analysis_completed = pyqtSignal()
    important_moment_selected = pyqtSignal(dict)  # Emitted when an important moment is selected
    
    def __init__(self, parent=None):
        """Initialize the content analysis panel."""
        super().__init__(parent)
        self.content_analyzer = ContentAnalyzer()
        self.analysis_results = None
        self.segments = []
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create summary tab
        self.summary_tab = QWidget()
        self._init_summary_tab()
        self.tab_widget.addTab(self.summary_tab, "Summary")
        
        # Create topics tab
        self.topics_tab = QWidget()
        self._init_topics_tab()
        self.tab_widget.addTab(self.topics_tab, "Topics")
        
        # Create keywords tab
        self.keywords_tab = QWidget()
        self._init_keywords_tab()
        self.tab_widget.addTab(self.keywords_tab, "Keywords")
        
        # Create important moments tab
        self.moments_tab = QWidget()
        self._init_moments_tab()
        self.tab_widget.addTab(self.moments_tab, "Important Moments")
        
        # Create sentiment tab
        self.sentiment_tab = QWidget()
        self._init_sentiment_tab()
        self.tab_widget.addTab(self.sentiment_tab, "Sentiment")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Add analyze button and progress bar
        button_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("Analyze Content")
        self.analyze_button.clicked.connect(self.analyze_content)
        button_layout.addWidget(self.analyze_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        button_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(button_layout)
    
    def _init_summary_tab(self):
        """Initialize the summary tab."""
        layout = QVBoxLayout(self.summary_tab)
        
        # Add summary label
        summary_label = QLabel("Content Summary:")
        summary_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(summary_label)
        
        # Add summary text area
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
    
    def _init_topics_tab(self):
        """Initialize the topics tab."""
        layout = QVBoxLayout(self.topics_tab)
        
        # Add topics label
        topics_label = QLabel("Identified Topics:")
        topics_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(topics_label)
        
        # Add topics table
        self.topics_table = QTableWidget(0, 2)
        self.topics_table.setHorizontalHeaderLabels(["Topic", "Key Words"])
        self.topics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.topics_table)
    
    def _init_keywords_tab(self):
        """Initialize the keywords tab."""
        layout = QVBoxLayout(self.keywords_tab)
        
        # Add keywords label
        keywords_label = QLabel("Key Terms and Phrases:")
        keywords_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(keywords_label)
        
        # Add keywords table
        self.keywords_table = QTableWidget(0, 2)
        self.keywords_table.setHorizontalHeaderLabels(["Keyword", "Relevance"])
        self.keywords_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.keywords_table)
    
    def _init_moments_tab(self):
        """Initialize the important moments tab."""
        layout = QVBoxLayout(self.moments_tab)
        
        # Add important moments label
        moments_label = QLabel("Important Moments:")
        moments_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(moments_label)
        
        # Add important moments table
        self.moments_table = QTableWidget(0, 4)
        self.moments_table.setHorizontalHeaderLabels(["Time", "Speaker", "Text", "Importance"])
        self.moments_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.moments_table.cellDoubleClicked.connect(self._on_moment_double_clicked)
        layout.addWidget(self.moments_table)
    
    def _init_sentiment_tab(self):
        """Initialize the sentiment tab."""
        layout = QVBoxLayout(self.sentiment_tab)
        
        # Overall sentiment section
        overall_frame = QFrame()
        overall_frame.setFrameShape(QFrame.Shape.StyledPanel)
        overall_layout = QVBoxLayout(overall_frame)
        
        overall_label = QLabel("Overall Sentiment:")
        overall_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        overall_layout.addWidget(overall_label)
        
        self.overall_sentiment_label = QLabel("No sentiment data available")
        overall_layout.addWidget(self.overall_sentiment_label)
        
        # Add sentiment chart
        self.sentiment_figure = plt.figure(figsize=(6, 3))
        self.sentiment_canvas = FigureCanvas(self.sentiment_figure)
        overall_layout.addWidget(self.sentiment_canvas)
        
        layout.addWidget(overall_frame)
        
        # Sentiment by segment section
        segment_label = QLabel("Sentiment by Segment:")
        segment_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(segment_label)
        
        # Create table for segment sentiment
        self.sentiment_table = QTableWidget()
        self.sentiment_table.setColumnCount(4)
        self.sentiment_table.setHorizontalHeaderLabels(["Speaker", "Text", "Sentiment", "Score"])
        
        self.sentiment_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.sentiment_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.sentiment_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.sentiment_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        
        layout.addWidget(self.sentiment_table)
    
    def set_segments(self, segments: List[Dict[str, Any]]):
        """
        Set the transcript segments for analysis.
        
        Args:
            segments: List of transcript segments with text, speaker, etc.
        """
        self.segments = segments
        self.clear_results()  # Clear any previous results
    
    def clear_results(self):
        """Clear all analysis results."""
        self.analysis_results = None
        
        # Clear summary
        self.summary_text.clear()
        
        # Clear topics table
        self.topics_table.setRowCount(0)
        
        # Clear keywords table
        self.keywords_table.setRowCount(0)
        
        # Clear important moments table
        self.moments_table.setRowCount(0)
    
    @pyqtSlot()
    def analyze_content(self):
        """Analyze the content of the segments."""
        if not self.segments:
            QMessageBox.warning(self, "No Content", "No content available for analysis.")
            return
        
        try:
            # Show progress
            self.progress_bar.setVisible(True)
            self.analyze_button.setEnabled(False)
            
            # Run analysis in a separate thread to avoid blocking UI
            class AnalysisThread(QThread):
                def __init__(self, parent, analyzer, segments):
                    super().__init__(parent)
                    self.analyzer = analyzer
                    self.segments = segments
                    self.results = None
                    self.error = None
                
                def run(self):
                    try:
                        self.results = self.analyzer.analyze_content(self.segments)
                    except Exception as e:
                        self.error = e
            
            self.analysis_thread = AnalysisThread(self, self.content_analyzer, self.segments)
            
            def on_analysis_completed():
                # Hide progress
                self.progress_bar.setVisible(False)
                self.analyze_button.setEnabled(True)
                
                if self.analysis_thread.error:
                    # Handle error
                    if isinstance(self.analysis_thread.error, ContentAnalysisError):
                        ErrorHandler.handle_exception(self.analysis_thread.error, parent=self)
                    else:
                        QMessageBox.critical(
                            self, 
                            "Analysis Error", 
                            f"An error occurred during content analysis: {str(self.analysis_thread.error)}"
                        )
                    return
                
                # Update UI with results
                self.analysis_results = self.analysis_thread.results
                self._update_ui_with_results()
                
                # Emit signal
                self.analysis_completed.emit()
            
            self.analysis_thread.finished.connect(on_analysis_completed)
            self.analysis_thread.start()
            
        except Exception as e:
            # Hide progress
            self.progress_bar.setVisible(False)
            self.analyze_button.setEnabled(True)
            
            # Handle error
            ErrorHandler.handle_exception(e, parent=self)
    
    def _update_ui_with_results(self):
        """Update UI with analysis results."""
        if not self.analysis_results:
            return
        
        # Update summary
        summary = self.analysis_results.get('summary', '')
        self.summary_text.setText(summary)
        
        # Update topics
        topics = self.analysis_results.get('topics', [])
        self.topics_table.setRowCount(len(topics))
        for i, topic in enumerate(topics):
            topic_words = topic.get('words', [])
            topic_text = ", ".join(topic_words)
            
            topic_item = QTableWidgetItem(f"Topic {i+1}")
            words_item = QTableWidgetItem(topic_text)
            
            self.topics_table.setItem(i, 0, topic_item)
            self.topics_table.setItem(i, 1, words_item)
        
        # Update keywords
        keywords = self.analysis_results.get('keywords', [])
        self.keywords_table.setRowCount(len(keywords))
        for i, (keyword, score) in enumerate(keywords):
            keyword_item = QTableWidgetItem(keyword)
            score_item = QTableWidgetItem(f"{score:.4f}")
            
            self.keywords_table.setItem(i, 0, keyword_item)
            self.keywords_table.setItem(i, 1, score_item)
        
        # Update important moments
        important_moments = self.analysis_results.get('important_moments', [])
        self.moments_table.setRowCount(len(important_moments))
        for i, moment in enumerate(important_moments):
            time_str = self._format_time(moment.get('start_time', 0))
            speaker = moment.get('speaker', 'Unknown')
            text = moment.get('text', '')
            score = moment.get('importance_score', 0)
            
            time_item = QTableWidgetItem(time_str)
            speaker_item = QTableWidgetItem(speaker)
            text_item = QTableWidgetItem(text)
            score_item = QTableWidgetItem(f"{score:.2f}")
            
            self.moments_table.setItem(i, 0, time_item)
            self.moments_table.setItem(i, 1, speaker_item)
            self.moments_table.setItem(i, 2, text_item)
            self.moments_table.setItem(i, 3, score_item)
        
        # Update sentiment information
        sentiment_data = self.analysis_results.get('sentiment_analysis', {})
        overall_sentiment = sentiment_data.get('overall', {})
        segment_sentiments = sentiment_data.get('by_segment', [])
        
        # Update overall sentiment label
        overall_label = overall_sentiment.get('label', 'Neutral')
        compound_score = overall_sentiment.get('compound', 0.0)
        self.overall_sentiment_label.setText(f"{overall_label} (Score: {compound_score:.2f})")
        
        # Set label color based on sentiment
        self._set_sentiment_label_color(self.overall_sentiment_label, compound_score)
        
        # Update sentiment chart
        self._update_sentiment_chart(overall_sentiment)
        
        # Update sentiment table
        self.sentiment_table.setRowCount(len(segment_sentiments))
        for i, segment in enumerate(segment_sentiments):
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '')
            sentiment_label = segment.get('sentiment_label', 'Neutral')
            compound_score = segment.get('sentiment', {}).get('compound', 0.0)
            
            speaker_item = QTableWidgetItem(speaker)
            text_item = QTableWidgetItem(text)
            sentiment_item = QTableWidgetItem(sentiment_label)
            score_item = QTableWidgetItem(f"{compound_score:.2f}")
            
            # Color the sentiment label based on the score
            self._set_sentiment_item_color(sentiment_item, compound_score)
            
            self.sentiment_table.setItem(i, 0, speaker_item)
            self.sentiment_table.setItem(i, 1, text_item)
            self.sentiment_table.setItem(i, 2, sentiment_item)
            self.sentiment_table.setItem(i, 3, score_item)
    
    def _set_sentiment_label_color(self, label, score):
        """Set the color of a sentiment label based on the score."""
        if score <= -0.6:
            label.setStyleSheet("color: #d62728")  # Red for very negative
        elif score <= -0.2:
            label.setStyleSheet("color: #ff7f0e")  # Orange for negative
        elif score < 0.2:
            label.setStyleSheet("color: #7f7f7f")  # Gray for neutral
        elif score < 0.6:
            label.setStyleSheet("color: #2ca02c")  # Green for positive
        else:
            label.setStyleSheet("color: #1f77b4")  # Blue for very positive
    
    def _set_sentiment_item_color(self, item, score):
        """Set the color of a sentiment table item based on the score."""
        if score <= -0.6:
            item.setBackground(QBrush(QColor(255, 200, 200)))  # Light red
        elif score <= -0.2:
            item.setBackground(QBrush(QColor(255, 225, 200)))  # Light orange
        elif score < 0.2:
            item.setBackground(QBrush(QColor(240, 240, 240)))  # Light gray
        elif score < 0.6:
            item.setBackground(QBrush(QColor(200, 255, 200)))  # Light green
        else:
            item.setBackground(QBrush(QColor(200, 225, 255)))  # Light blue
    
    def _update_sentiment_chart(self, sentiment_data):
        """Update the sentiment chart with the provided data."""
        self.sentiment_figure.clear()
        ax = self.sentiment_figure.add_subplot(111)
        
        # Extract sentiment scores
        pos = sentiment_data.get('pos', 0)
        neu = sentiment_data.get('neu', 0)
        neg = sentiment_data.get('neg', 0)
        
        # Create bar chart
        labels = ['Positive', 'Neutral', 'Negative']
        values = [pos, neu, neg]
        colors = ['#2ca02c', '#7f7f7f', '#d62728']
        
        ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 1.0)
        ax.set_title('Sentiment Distribution')
        ax.set_ylabel('Score')
        
        self.sentiment_canvas.draw()
    
    def _format_time(self, seconds: float) -> str:
        """
        Format seconds as MM:SS.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        minutes = int(seconds) // 60
        seconds_remainder = int(seconds) % 60
        return f"{minutes:02d}:{seconds_remainder:02d}"
    
    def _on_moment_double_clicked(self, row: int, col: int):
        """
        Handle double-click on an important moment.
        
        Args:
            row: Table row
            col: Table column
        """
        if not self.analysis_results:
            return
        
        # Get the moment data
        moments = self.analysis_results.get('important_moments', [])
        if 0 <= row < len(moments):
            moment = moments[row]
            self.important_moment_selected.emit(moment) 