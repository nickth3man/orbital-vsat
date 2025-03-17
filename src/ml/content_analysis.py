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
import nltk
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.sentiment import SentimentIntensityAnalyzer
import networkx as nx

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


# Custom implementation to replace gensim.summarization.summarize
def textrank_summarize(text: str, ratio: float = 0.2) -> str:
    """
    Summarize text using TextRank algorithm.
    
    Args:
        text: Text to summarize
        ratio: Proportion of sentences to include in summary
        
    Returns:
        Summarized text
    """
    if not text or len(text.strip()) == 0:
        return ""
    
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    if len(sentences) <= 1:
        return text
    
    # Calculate number of sentences for summary
    num_sentences = max(1, int(len(sentences) * ratio))
    
    # Create similarity matrix
    stop_words = set(stopwords.words('english'))
    sentence_vectors = []
    
    # Preprocess and vectorize sentences
    for sentence in sentences:
        words = [word.lower() for word in word_tokenize(sentence) 
                 if word.isalnum() and word.lower() not in stop_words]
        sentence_vectors.append(words)
    
    # Build similarity graph
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                # Calculate Jaccard similarity
                set_i = set(sentence_vectors[i])
                set_j = set(sentence_vectors[j])
                
                if len(set_i) == 0 or len(set_j) == 0:
                    continue
                    
                similarity = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                similarity_matrix[i][j] = similarity
    
    # Create graph and apply PageRank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Rank sentences by score
    ranked_sentences = sorted(
        ((scores[i], i, s) for i, s in enumerate(sentences)), 
        reverse=True
    )
    
    # Select top sentences
    top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
    
    # Combine sentences
    return " ".join(x[2] for x in top_sentences)


# Custom implementation to replace gensim.summarization.keywords
def extract_keywords(text: str, ratio: float = 0.1, lemmatize: bool = True, 
                     scores: bool = False) -> List:
    """
    Extract keywords from text using TF-IDF.
    
    Args:
        text: Text to extract keywords from
        ratio: Proportion of words to consider as keywords
        lemmatize: Whether to lemmatize words
        scores: Whether to return scores with keywords
        
    Returns:
        List of keywords or (keyword, score) tuples if scores=True
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Tokenize and preprocess
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    
    # Filter words
    filtered_words = [word for word in words 
                      if word.isalpha() and word.lower() not in stop_words]
    
    # Lemmatize if requested
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        filtered_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    
    # Calculate number of keywords to return
    num_keywords = max(1, int(len(word_freq) * ratio))
    
    # Get top keywords
    keywords = word_freq.most_common(num_keywords)
    
    if scores:
        # Normalize scores to 0-1 range
        max_count = keywords[0][1] if keywords else 1
        return [(word, count / max_count) for word, count in keywords]
    else:
        return [word for word, _ in keywords]


class TopicModeler:
    """Class for identifying topics in transcript content."""
    
    def __init__(self, method: str = 'lda', num_topics: int = 5):
        """
        Initialize the topic modeler.
        
        Args:
            method: Method to use for topic modeling ('lda' or 'nmf')
            num_topics: Number of topics to identify
        """
        self.method = method
        self.num_topics = num_topics
        self.model = None
        self.vectorizer = None
        self.feature_names = None
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the topic model to the given texts.
        
        Args:
            texts: List of texts to fit the model to
        """
        if not texts:
            logger.warning("No texts provided for topic modeling")
            return
            
        try:
            # Vectorize texts
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
            X = self.vectorizer.fit_transform(texts)
            
            # Get feature names
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # Create and fit model
            if self.method == 'lda':
                self.model = LatentDirichletAllocation(
                    n_components=self.num_topics,
                    random_state=42
                )
            elif self.method == 'nmf':
                self.model = NMF(
                    n_components=self.num_topics,
                    random_state=42
                )
            else:
                raise ContentAnalysisError(
                    f"Unknown topic modeling method: {self.method}",
                    details={"valid_methods": ["lda", "nmf"]}
                )
                
            self.model.fit(X)
            
        except Exception as e:
            logger.error(f"Error fitting topic model: {str(e)}")
            raise ContentAnalysisError(
                "Failed to fit topic model",
                details={"original_error": str(e)}
            )
    
    def get_topics(self, num_words: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Get the top words for each topic.
        
        Args:
            num_words: Number of words to include for each topic
            
        Returns:
            List of topics, each containing (word, weight) tuples
        """
        if self.model is None or self.feature_names is None:
            logger.warning("Topic model not fitted yet")
            return []
            
        try:
            topics = []
            
            for topic_idx, topic in enumerate(self.model.components_):
                # Get top words for this topic
                top_words_idx = topic.argsort()[:-num_words-1:-1]
                top_words = [
                    (self.feature_names[i], topic[i]) for i in top_words_idx
                ]
                topics.append(top_words)
                
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return []
    
    def transform(self, text: str) -> List[Tuple[int, float]]:
        """
        Transform a text to topic distribution.
        
        Args:
            text: Text to transform
            
        Returns:
            List of (topic_id, weight) tuples
        """
        if self.model is None or self.vectorizer is None:
            logger.warning("Topic model not fitted yet")
            return []
            
        try:
            # Vectorize text
            X = self.vectorizer.transform([text])
            
            # Transform to topic distribution
            topic_distribution = self.model.transform(X)[0]
            
            # Create (topic_id, weight) tuples and sort by weight
            topics = [
                (topic_id, weight) 
                for topic_id, weight in enumerate(topic_distribution)
            ]
            topics.sort(key=lambda x: x[1], reverse=True)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error transforming text to topics: {str(e)}")
            return []


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
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation
        tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords from the given text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of (keyword, score) tuples
        """
        if not text or len(text.strip()) == 0:
            return []
            
        try:
            preprocessed_text = self._preprocess_text(text)
            
            if self.method == 'tfidf':
                return self._extract_with_tfidf(preprocessed_text)
            elif self.method == 'textrank':
                # Use custom implementation instead of gensim
                keywords = extract_keywords(text, ratio=0.1, scores=True)
                return keywords[:self.max_keywords]
            elif self.method == 'count':
                return self._extract_with_count(preprocessed_text)
            else:
                logger.warning(
                    f"Unknown keyword extraction method: {self.method}, "
                    "falling back to tfidf"
                )
                return self._extract_with_tfidf(preprocessed_text)
                
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def _extract_with_tfidf(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of (keyword, score) tuples
        """
        # Create vectorizer
        vectorizer = TfidfVectorizer(max_features=100)
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get scores
        scores = tfidf_matrix.toarray()[0]
        
        # Create (word, score) tuples and sort
        word_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores[:self.max_keywords]
    
    def _extract_with_count(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using word frequency.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of (keyword, score) tuples
        """
        # Tokenize
        tokens = text.split()
        
        # Count words
        word_counts = Counter(tokens)
        
        # Get most common words
        most_common = word_counts.most_common(self.max_keywords)
        
        # Normalize scores
        max_count = most_common[0][1] if most_common else 1
        return [(word, count / max_count) for word, count in most_common]


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
        if not text or len(text.strip()) == 0:
            return ""
            
        try:
            if self.method == 'extractive':
                return self._extractive_summarize(text)
            elif self.method == 'abstractive':
                return self._abstractive_summarize(text)
            else:
                logger.warning(
                    f"Unknown summarization method: {self.method}, "
                    "falling back to extractive"
                )
                return self._extractive_summarize(text)
                
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return text[:int(len(text) * self.ratio)]  # Fallback to simple truncation
    
    def _extractive_summarize(self, text: str) -> str:
        """
        Generate an extractive summary using sentence ranking.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        try:
            # Use custom TextRank implementation instead of gensim
            return textrank_summarize(text, ratio=self.ratio)
        except Exception as e:
            logger.warning(
                f"TextRank summarization failed: {str(e)}, "
                "falling back to simple method"
            )
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
        
        if len(sentences) <= 1:
            return text
        
        # Calculate number of sentences for summary
        num_sentences = max(1, int(len(sentences) * self.ratio))
        
        # Simple approach: take first sentence and then evenly distributed sentences
        selected_sentences = [sentences[0]]  # Always include first sentence
        
        if num_sentences > 1:
            # Select remaining sentences evenly distributed
            step = len(sentences) / (num_sentences - 1)
            indices = [min(len(sentences) - 1, int(step * i)) 
                       for i in range(1, num_sentences)]
            selected_sentences.extend([sentences[i] for i in indices])
        
        return ' '.join(selected_sentences)
    
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
        logger.info("Abstractive summarization not implemented, falling back to extractive")
        return self._extractive_summarize(text)


class ImportantMomentDetector:
    """Class for detecting important moments in transcript content."""
    
    def __init__(self, keyword_weight: float = 0.5, sentiment_weight: float = 0.3,
                 topic_weight: float = 0.2):
        """
        Initialize the important moment detector.
        
        Args:
            keyword_weight: Weight to give to keyword-based importance
            sentiment_weight: Weight to give to sentiment-based importance
            topic_weight: Weight to give to topic-based importance
        """
        self.keyword_weight = keyword_weight
        self.sentiment_weight = sentiment_weight
        self.topic_weight = topic_weight
        self.keyword_extractor = KeywordExtractor(method='tfidf', max_keywords=5)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def detect_important_moments(self, segments: List[Dict]) -> List[Dict]:
        """
        Detect important moments in the given transcript segments.
        
        Args:
            segments: List of transcript segments with text and timestamps
            
        Returns:
            List of segments with importance scores
        """
        if not segments:
            return []
            
        try:
            # Extract all text for keyword analysis
            all_text = " ".join([segment.get('text', '') for segment in segments])
            
            # Extract keywords from all text
            keywords = self.keyword_extractor.extract_keywords(all_text)
            keyword_dict = {word: score for word, score in keywords}
            
            # Score each segment
            scored_segments = []
            
            for segment in segments:
                text = segment.get('text', '')
                if not text:
                    continue
                
                # Calculate keyword score
                keyword_score = self._calculate_keyword_score(text, keyword_dict)
                
                # Calculate sentiment score
                sentiment_score = self._calculate_sentiment_score(text)
                
                # Calculate overall importance
                importance = (
                    self.keyword_weight * keyword_score +
                    self.sentiment_weight * sentiment_score
                )
                
                # Add scores to segment
                scored_segment = segment.copy()
                scored_segment['importance'] = importance
                scored_segment['keyword_score'] = keyword_score
                scored_segment['sentiment_score'] = sentiment_score
                
                scored_segments.append(scored_segment)
            
            # Sort by importance
            scored_segments.sort(key=lambda x: x['importance'], reverse=True)
            
            return scored_segments
            
        except Exception as e:
            logger.error(f"Error detecting important moments: {str(e)}")
            return segments
    
    def _calculate_keyword_score(self, text: str, keyword_dict: Dict[str, float]) -> float:
        """
        Calculate keyword-based importance score for a text.
        
        Args:
            text: Text to score
            keyword_dict: Dictionary of keywords and their scores
            
        Returns:
            Keyword importance score
        """
        words = word_tokenize(text.lower())
        score = 0.0
        
        for word in words:
            if word in keyword_dict:
                score += keyword_dict[word]
        
        # Normalize by text length
        if words:
            score = score / len(words)
        
        return score
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """
        Calculate sentiment-based importance score for a text.
        
        Args:
            text: Text to score
            
        Returns:
            Sentiment importance score
        """
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        # Use compound score, but take absolute value (strong sentiment in either direction is important)
        return abs(sentiment['compound'])


class SentimentAnalyzer:
    """Class for analyzing sentiment in transcript content."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment in the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or len(text.strip()) == 0:
            return {'pos': 0.0, 'neg': 0.0, 'neu': 0.0, 'compound': 0.0}
            
        try:
            return self.analyzer.polarity_scores(text)
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'pos': 0.0, 'neg': 0.0, 'neu': 0.0, 'compound': 0.0}
    
    def get_sentiment_label(self, text: str) -> str:
        """
        Get a sentiment label for the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment label ('positive', 'negative', or 'neutral')
        """
        scores = self.analyze_sentiment(text)
        
        if scores['compound'] >= 0.05:
            return 'positive'
        elif scores['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'


class ContentAnalyzer:
    """Main class for analyzing transcript content."""
    
    def __init__(self):
        """Initialize the content analyzer with various analysis components."""
        self.topic_modeler = TopicModeler()
        self.keyword_extractor = KeywordExtractor()
        self.summarizer = Summarizer()
        self.important_moment_detector = ImportantMomentDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not text or len(text.strip()) == 0:
            return {
                'keywords': [],
                'summary': '',
                'sentiment': {'label': 'neutral', 'scores': {}},
                'topics': []
            }
            
        try:
            # Extract keywords
            keywords = self.keyword_extractor.extract_keywords(text)
            
            # Generate summary
            summary = self.summarizer.summarize(text)
            
            # Analyze sentiment
            sentiment_scores = self.sentiment_analyzer.analyze_sentiment(text)
            sentiment_label = self.sentiment_analyzer.get_sentiment_label(text)
            
            # Fit topic model and get topics
            self.topic_modeler.fit([text])
            topics = self.topic_modeler.get_topics()
            
            return {
                'keywords': keywords,
                'summary': summary,
                'sentiment': {
                    'label': sentiment_label,
                    'scores': sentiment_scores
                },
                'topics': topics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return {
                'keywords': [],
                'summary': '',
                'sentiment': {'label': 'neutral', 'scores': {}},
                'topics': [],
                'error': str(e)
            }