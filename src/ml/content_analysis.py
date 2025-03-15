"""
Content analysis module for VSAT.

This module provides functionality for analyzing transcript content:
- Topic identification
- Keyword extraction
- Summarization
- Important moment detection
- Sentiment analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.summarization import summarize as gensim_summarize
from gensim.summarization import keywords as gensim_keywords
from nltk.sentiment import SentimentIntensityAnalyzer

from src.utils.error_handler import VSATError, ErrorSeverity

logger = logging.getLogger(__name__)

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')


class ContentAnalysisError(VSATError):
    """Exception raised for errors in content analysis."""
    def __init__(self, message, severity=ErrorSeverity.ERROR, details=None):
        super().__init__(message, severity, details)


class TopicModeler:
    """Class for identifying topics in transcript content."""
    
    def __init__(self, num_topics: int = 5, model_type: str = 'lda'):
        """
        Initialize the topic modeler.
        
        Args:
            num_topics: Number of topics to extract
            model_type: Type of topic model to use ('lda' or 'nmf')
        """
        self.num_topics = num_topics
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for topic modeling.
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation
        tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the topic model on the given texts.
        
        Args:
            texts: List of transcript segments or documents
        """
        try:
            # Preprocess texts
            preprocessed_texts = [self._preprocess_text(text) for text in texts]
            
            # Vectorize
            self.vectorizer = TfidfVectorizer(max_features=1000)
            X = self.vectorizer.fit_transform(preprocessed_texts)
            
            # Create and fit model
            if self.model_type == 'lda':
                self.model = LatentDirichletAllocation(
                    n_components=self.num_topics,
                    random_state=42,
                    max_iter=20
                )
            elif self.model_type == 'nmf':
                self.model = NMF(
                    n_components=self.num_topics,
                    random_state=42,
                    max_iter=200
                )
            else:
                raise ContentAnalysisError(
                    f"Unknown model type: {self.model_type}",
                    details={"valid_types": ["lda", "nmf"]}
                )
            
            self.model.fit(X)
            logger.info(f"Topic model trained with {self.num_topics} topics")
            
        except Exception as e:
            logger.error(f"Error fitting topic model: {str(e)}")
            raise ContentAnalysisError(
                "Failed to fit topic model",
                details={"original_error": str(e)}
            )
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to topic distributions.
        
        Args:
            texts: List of transcript segments or documents
            
        Returns:
            Array of topic distributions for each text
        """
        if self.model is None or self.vectorizer is None:
            raise ContentAnalysisError("Model not fitted yet")
        
        try:
            # Preprocess texts
            preprocessed_texts = [self._preprocess_text(text) for text in texts]
            
            # Vectorize
            X = self.vectorizer.transform(preprocessed_texts)
            
            # Transform
            return self.model.transform(X)
            
        except Exception as e:
            logger.error(f"Error transforming texts to topics: {str(e)}")
            raise ContentAnalysisError(
                "Failed to transform texts to topics",
                details={"original_error": str(e)}
            )
    
    def get_topic_words(self, n_words: int = 10) -> List[List[str]]:
        """
        Get the top words for each topic.
        
        Args:
            n_words: Number of words to return per topic
            
        Returns:
            List of word lists for each topic
        """
        if self.model is None or self.vectorizer is None:
            raise ContentAnalysisError("Model not fitted yet")
        
        try:
            feature_names = self.vectorizer.get_feature_names_out()
            topic_words = []
            
            for topic_idx, topic in enumerate(self.model.components_):
                top_words_idx = topic.argsort()[:-n_words-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_words.append(top_words)
            
            return topic_words
            
        except Exception as e:
            logger.error(f"Error getting topic words: {str(e)}")
            raise ContentAnalysisError(
                "Failed to get topic words",
                details={"original_error": str(e)}
            )
    
    def get_dominant_topic(self, text: str) -> Tuple[int, float]:
        """
        Get the dominant topic for a given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (topic_id, probability)
        """
        if self.model is None or self.vectorizer is None:
            raise ContentAnalysisError("Model not fitted yet")
        
        try:
            # Preprocess text
            preprocessed_text = self._preprocess_text(text)
            
            # Vectorize
            X = self.vectorizer.transform([preprocessed_text])
            
            # Get topic distribution
            topic_distribution = self.model.transform(X)[0]
            
            # Get dominant topic
            dominant_topic = topic_distribution.argmax()
            probability = topic_distribution[dominant_topic]
            
            return dominant_topic, probability
            
        except Exception as e:
            logger.error(f"Error getting dominant topic: {str(e)}")
            raise ContentAnalysisError(
                "Failed to get dominant topic",
                details={"original_error": str(e)}
            )


class KeywordExtractor:
    """Class for extracting keywords from transcript content."""
    
    def __init__(self, method: str = 'tfidf', max_keywords: int = 10):
        """
        Initialize the keyword extractor.
        
        Args:
            method: Method to use for keyword extraction ('tfidf', 'textrank', or 'count')
            max_keywords: Maximum number of keywords to extract
        """
        self.method = method
        self.max_keywords = max_keywords
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for keyword extraction.
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords from the given text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            if not text or text.strip() == "":
                return []
            
            preprocessed_text = self._preprocess_text(text)
            
            if self.method == 'tfidf':
                return self._extract_with_tfidf(preprocessed_text)
            elif self.method == 'textrank':
                return self._extract_with_textrank(preprocessed_text)
            elif self.method == 'count':
                return self._extract_with_count(preprocessed_text)
            else:
                raise ContentAnalysisError(
                    f"Unknown extraction method: {self.method}",
                    details={"valid_methods": ["tfidf", "textrank", "count"]}
                )
                
        except Exception as e:
            if isinstance(e, ContentAnalysisError):
                raise e
            
            logger.error(f"Error extracting keywords: {str(e)}")
            raise ContentAnalysisError(
                "Failed to extract keywords",
                details={"original_error": str(e)}
            )
    
    def _extract_with_tfidf(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of (keyword, score) tuples
        """
        # Create a one-document corpus
        corpus = [text]
        
        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english'
        )
        X = vectorizer.fit_transform(corpus)
        
        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        scores = X.toarray()[0]
        
        # Sort by score
        keywords = [(feature_names[i], scores[i]) for i in scores.argsort()[::-1]
                    if scores[i] > 0]
        
        return keywords[:self.max_keywords]
    
    def _extract_with_textrank(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using TextRank.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            # Use gensim's implementation of TextRank
            keywords = gensim_keywords(text, words=self.max_keywords, split=True)
            
            # Convert to (keyword, score) format with dummy scores
            return [(kw, 1.0 - i/len(keywords)) for i, kw in enumerate(keywords)]
            
        except Exception as e:
            logger.warning(f"Error using TextRank, falling back to TF-IDF: {str(e)}")
            return self._extract_with_tfidf(text)
    
    def _extract_with_count(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using word frequency.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of (keyword, score) tuples
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words and len(t) > 1]
        
        # Count frequencies
        counter = Counter(tokens)
        
        # Get top keywords
        keywords = counter.most_common(self.max_keywords)
        
        # Normalize scores
        if keywords:
            max_count = keywords[0][1]
            return [(word, count/max_count) for word, count in keywords]
        else:
            return []


class Summarizer:
    """Class for generating summaries of transcript content."""
    
    def __init__(self, method: str = 'extractive', ratio: float = 0.2):
        """
        Initialize the summarizer.
        
        Args:
            method: Method to use for summarization ('extractive' or 'abstractive')
            ratio: Proportion of the original text to include in the summary
        """
        self.method = method
        self.ratio = ratio
        
    def summarize(self, text: str) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        try:
            if not text or text.strip() == "":
                return ""
            
            if self.method == 'extractive':
                return self._extractive_summarize(text)
            elif self.method == 'abstractive':
                return self._abstractive_summarize(text)
            else:
                raise ContentAnalysisError(
                    f"Unknown summarization method: {self.method}",
                    details={"valid_methods": ["extractive", "abstractive"]}
                )
                
        except Exception as e:
            if isinstance(e, ContentAnalysisError):
                raise e
            
            logger.error(f"Error generating summary: {str(e)}")
            raise ContentAnalysisError(
                "Failed to generate summary",
                details={"original_error": str(e)}
            )
    
    def _extractive_summarize(self, text: str) -> str:
        """
        Generate an extractive summary using sentence ranking.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        try:
            # Use gensim's implementation of TextRank for summarization
            summary = gensim_summarize(text, ratio=self.ratio)
            
            if not summary:
                # Fall back to simple sentence extraction if gensim fails
                return self._simple_extractive_summarize(text)
            
            return summary
            
        except Exception as e:
            logger.warning(f"Error using gensim summarize, falling back to simple method: {str(e)}")
            return self._simple_extractive_summarize(text)
    
    def _simple_extractive_summarize(self, text: str) -> str:
        """
        Simple extractive summarization by selecting top sentences.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # If there are few sentences, return all
        if len(sentences) <= 3:
            return text
        
        # Calculate word frequencies
        word_frequencies = {}
        stop_words = set(stopwords.words('english'))
        
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word.isalpha() and word not in stop_words:
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1
        
        # Normalize word frequencies
        if word_frequencies:
            max_frequency = max(word_frequencies.values())
            for word in word_frequencies:
                word_frequencies[word] = word_frequencies[word] / max_frequency
        
        # Calculate sentence scores
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            score = 0
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    score += word_frequencies[word]
            sentence_scores[i] = score
        
        # Select top sentences
        num_sentences = max(1, int(len(sentences) * self.ratio))
        top_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        top_sentence_indices = sorted(top_sentence_indices)  # Sort by position in text
        
        # Combine sentences
        summary = ' '.join([sentences[i] for i in top_sentence_indices])
        
        return summary
    
    def _abstractive_summarize(self, text: str) -> str:
        """
        Generate an abstractive summary.
        
        Note: This is a placeholder for future implementation with a transformer-based model.
        Currently falls back to extractive summarization.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        logger.warning("Abstractive summarization not yet implemented, falling back to extractive")
        return self._extractive_summarize(text)


class ImportantMomentDetector:
    """Class for identifying important moments in transcript content."""
    
    def __init__(self, 
                 keyword_weight: float = 0.4,
                 emotion_weight: float = 0.3,
                 speaker_weight: float = 0.2,
                 length_weight: float = 0.1):
        """
        Initialize the important moment detector.
        
        Args:
            keyword_weight: Weight for keyword-based importance
            emotion_weight: Weight for emotion/sentiment-based importance
            speaker_weight: Weight for speaker-based importance
            length_weight: Weight for length-based importance
        """
        self.keyword_weight = keyword_weight
        self.emotion_weight = emotion_weight
        self.speaker_weight = speaker_weight
        self.length_weight = length_weight
        self.keyword_extractor = KeywordExtractor(method='tfidf')
    
    def detect_important_moments(self, 
                                segments: List[Dict[str, Any]], 
                                keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Detect important moments in the transcript.
        
        Args:
            segments: List of transcript segments with text, speaker, etc.
            keywords: Optional list of keywords to look for
            
        Returns:
            List of segments with importance scores
        """
        try:
            if not segments:
                return []
            
            # Extract all text for keyword analysis if no keywords provided
            if keywords is None:
                all_text = " ".join([segment.get('text', '') for segment in segments])
                extracted_keywords = self.keyword_extractor.extract_keywords(all_text)
                keywords = [kw for kw, _ in extracted_keywords]
            
            # Calculate scores for each segment
            scored_segments = []
            for segment in segments:
                score = self._calculate_importance_score(segment, keywords)
                segment_copy = segment.copy()
                segment_copy['importance_score'] = score
                scored_segments.append(segment_copy)
            
            # Sort by score
            scored_segments.sort(key=lambda x: x['importance_score'], reverse=True)
            
            return scored_segments
            
        except Exception as e:
            logger.error(f"Error detecting important moments: {str(e)}")
            raise ContentAnalysisError(
                "Failed to detect important moments",
                details={"original_error": str(e)}
            )
    
    def _calculate_importance_score(self, 
                                   segment: Dict[str, Any], 
                                   keywords: List[str]) -> float:
        """
        Calculate importance score for a segment.
        
        Args:
            segment: Transcript segment
            keywords: List of important keywords
            
        Returns:
            Importance score (0.0 to 1.0)
        """
        text = segment.get('text', '')
        if not text:
            return 0.0
        
        # Keyword score
        keyword_score = self._calculate_keyword_score(text, keywords)
        
        # Emotion score (placeholder)
        emotion_score = 0.0  # TODO: Implement emotion detection
        
        # Speaker score
        speaker_score = self._calculate_speaker_score(segment)
        
        # Length score
        length_score = self._calculate_length_score(text)
        
        # Combine scores with weights
        total_score = (
            self.keyword_weight * keyword_score +
            self.emotion_weight * emotion_score +
            self.speaker_weight * speaker_score +
            self.length_weight * length_score
        )
        
        return total_score
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate keyword-based importance score.
        
        Args:
            text: Segment text
            keywords: List of important keywords
            
        Returns:
            Keyword score (0.0 to 1.0)
        """
        if not keywords:
            return 0.0
        
        # Normalize text
        text = text.lower()
        words = word_tokenize(text)
        
        # Count keyword occurrences
        keyword_count = sum(1 for word in words if word.lower() in [kw.lower() for kw in keywords])
        
        # Calculate score based on keyword density
        if not words:
            return 0.0
        
        keyword_density = keyword_count / len(words)
        
        # Scale to 0.0-1.0 (assuming max density of 0.5)
        return min(1.0, keyword_density * 2)
    
    def _calculate_speaker_score(self, segment: Dict[str, Any]) -> float:
        """
        Calculate speaker-based importance score.
        
        Args:
            segment: Transcript segment
            
        Returns:
            Speaker score (0.0 to 1.0)
        """
        # If segment has a speaker role, use that
        if 'speaker_role' in segment:
            role = segment['speaker_role'].lower()
            if 'moderator' in role or 'host' in role or 'leader' in role:
                return 1.0
            elif 'expert' in role or 'panelist' in role:
                return 0.8
            else:
                return 0.5
        
        # Default score
        return 0.5
    
    def _calculate_length_score(self, text: str) -> float:
        """
        Calculate length-based importance score.
        
        Args:
            text: Segment text
            
        Returns:
            Length score (0.0 to 1.0)
        """
        # Count words
        words = word_tokenize(text)
        word_count = len(words)
        
        # Scale based on typical important statement length
        # Assume 15-25 words is ideal
        if word_count < 5:
            return 0.2
        elif word_count < 15:
            return 0.6
        elif word_count <= 40:
            return 1.0
        else:
            return 0.7  # Penalize very long segments slightly


class SentimentAnalyzer:
    """Class for analyzing sentiment in transcript content."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of the provided text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing sentiment scores:
                - 'compound': Overall sentiment (-1 to 1)
                - 'pos': Positive sentiment (0 to 1)
                - 'neu': Neutral sentiment (0 to 1)
                - 'neg': Negative sentiment (0 to 1)
        """
        if not text or not text.strip():
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0
            }
        
        try:
            return self.analyzer.polarity_scores(text)
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            raise ContentAnalysisError(
                "Failed to analyze sentiment", 
                details={"text_length": len(text), "error": str(e)}
            )
    
    def get_sentiment_label(self, compound_score: float) -> str:
        """
        Get a human-readable sentiment label based on the compound score.
        
        Args:
            compound_score: The compound sentiment score (-1 to 1)
            
        Returns:
            A string label ('Very Negative', 'Negative', 'Neutral', 'Positive', or 'Very Positive')
        """
        if compound_score <= -0.75:
            return "Very Negative"
        elif compound_score <= -0.25:
            return "Negative"
        elif compound_score < 0.25:
            return "Neutral"
        elif compound_score < 0.75:
            return "Positive"
        else:
            return "Very Positive"
    
    def analyze_segment_sentiment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment for a transcript segment.
        
        Args:
            segment: Transcript segment dictionary containing 'text' key
            
        Returns:
            The segment with sentiment analysis results added
        """
        text = segment.get('text', '')
        sentiment_scores = self.analyze_sentiment(text)
        
        # Add sentiment information to the segment
        segment_with_sentiment = segment.copy()
        segment_with_sentiment['sentiment'] = sentiment_scores
        segment_with_sentiment['sentiment_label'] = self.get_sentiment_label(
            sentiment_scores['compound']
        )
        
        return segment_with_sentiment


class ContentAnalyzer:
    """Class for analyzing content of transcripts."""
    
    def __init__(self):
        """Initialize the content analyzer."""
        self.topic_modeler = TopicModeler(num_topics=5)
        self.keyword_extractor = KeywordExtractor()
        self.summarizer = Summarizer()
        self.moment_detector = ImportantMomentDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_content(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the content of transcript segments.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            Dictionary containing analysis results:
                - 'summary': Text summary
                - 'topics': Topic information
                - 'keywords': Extracted keywords
                - 'important_moments': List of important moments
                - 'sentiment_analysis': Sentiment analysis results
        """
        if not segments:
            return {
                'summary': '',
                'topics': [],
                'keywords': [],
                'important_moments': [],
                'sentiment_analysis': {
                    'overall': {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0},
                    'by_segment': []
                }
            }
        
        try:
            # Extract texts
            texts = [segment.get('text', '') for segment in segments]
            full_text = ' '.join(texts)
            
            # Analyze sentiment
            sentiment_by_segment = [
                self.sentiment_analyzer.analyze_segment_sentiment(segment) 
                for segment in segments
            ]
            
            # Calculate overall sentiment
            overall_compound = sum(s['sentiment']['compound'] for s in sentiment_by_segment) / len(sentiment_by_segment)
            overall_pos = sum(s['sentiment']['pos'] for s in sentiment_by_segment) / len(sentiment_by_segment)
            overall_neu = sum(s['sentiment']['neu'] for s in sentiment_by_segment) / len(sentiment_by_segment)
            overall_neg = sum(s['sentiment']['neg'] for s in sentiment_by_segment) / len(sentiment_by_segment)
            
            overall_sentiment = {
                'compound': overall_compound,
                'pos': overall_pos,
                'neu': overall_neu,
                'neg': overall_neg,
                'label': self.sentiment_analyzer.get_sentiment_label(overall_compound)
            }
            
            # Get summary
            summary = self.summarizer.summarize(full_text)
            
            # Extract keywords
            keywords = self.keyword_extractor.extract_keywords(full_text)
            
            # Extract important moments
            keyword_texts = [k[0] for k in keywords[:10]]
            important_moments = self.moment_detector.detect_important_moments(segments, keyword_texts)
            
            # Get topics
            self.topic_modeler.fit(texts)
            topic_words = self.topic_modeler.get_topic_words()
            
            # Transform segments to get topic distribution
            topic_distributions = self.topic_modeler.transform(texts)
            
            topics = []
            for i, words in enumerate(topic_words):
                topic_segments = []
                for j, segment in enumerate(segments):
                    if topic_distributions[j].argmax() == i:
                        topic_segments.append(segment)
                
                topics.append({
                    'id': i,
                    'words': words,
                    'segments': topic_segments
                })
            
            return {
                'summary': summary,
                'topics': topics,
                'keywords': keywords,
                'important_moments': important_moments,
                'sentiment_analysis': {
                    'overall': overall_sentiment,
                    'by_segment': sentiment_by_segment
                }
            }
            
        except Exception as e:
            logger.error(f"Content analysis error: {str(e)}")
            raise ContentAnalysisError(
                "Failed to analyze content", 
                details={"error": str(e)}
            ) 