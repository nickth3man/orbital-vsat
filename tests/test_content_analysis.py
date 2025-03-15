"""
Tests for the content analysis module.
"""

import unittest
import os
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.ml.content_analysis import (
    ContentAnalyzer,
    TopicModeler,
    KeywordExtractor,
    Summarizer,
    ImportantMomentDetector,
    SentimentAnalyzer,
    ContentAnalysisError
)


class TestTopicModeler(unittest.TestCase):
    """Test cases for the TopicModeler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.topic_modeler = TopicModeler(num_topics=3)
        self.test_texts = [
            "Machine learning is a subset of artificial intelligence that provides systems with the ability to learn.",
            "Natural language processing is a field of AI that enables computers to understand human language.",
            "Computer vision is an interdisciplinary field that deals with how computers can gain understanding from images.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            "Reinforcement learning is about taking suitable actions to maximize reward in a particular situation."
        ]
    
    def test_init(self):
        """Test initialization of TopicModeler."""
        self.assertEqual(self.topic_modeler.num_topics, 3)
        self.assertEqual(self.topic_modeler.model_type, 'lda')
        self.assertIsNone(self.topic_modeler.vectorizer)
        self.assertIsNone(self.topic_modeler.model)
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "Machine Learning is amazing! AI and ML are great technologies."
        processed = self.topic_modeler._preprocess_text(text)
        
        # Check that text is lowercase and stopwords/punctuation are removed
        self.assertNotIn("!", processed)
        self.assertNotIn("and", processed)
        self.assertIn("machine", processed)
        self.assertIn("learning", processed)
    
    def test_fit_lda(self):
        """Test fitting LDA topic model."""
        self.topic_modeler.fit(self.test_texts)
        
        # Check that model and vectorizer are initialized
        self.assertIsNotNone(self.topic_modeler.model)
        self.assertIsNotNone(self.topic_modeler.vectorizer)
    
    def test_fit_nmf(self):
        """Test fitting NMF topic model."""
        topic_modeler = TopicModeler(num_topics=3, model_type='nmf')
        topic_modeler.fit(self.test_texts)
        
        # Check that model and vectorizer are initialized
        self.assertIsNotNone(topic_modeler.model)
        self.assertIsNotNone(topic_modeler.vectorizer)
    
    def test_fit_invalid_model(self):
        """Test fitting with invalid model type."""
        topic_modeler = TopicModeler(num_topics=3, model_type='invalid')
        
        with self.assertRaises(ContentAnalysisError):
            topic_modeler.fit(self.test_texts)
    
    def test_transform(self):
        """Test transforming texts to topic distributions."""
        self.topic_modeler.fit(self.test_texts)
        
        # Transform a single text
        topic_dist = self.topic_modeler.transform(["Machine learning is fascinating."])
        
        # Check that the result is an ndarray with expected shape
        self.assertIsInstance(topic_dist, np.ndarray)
        self.assertEqual(topic_dist.shape, (1, 3))  # 1 document, 3 topics
        
        # Check that probabilities sum to approximately 1
        self.assertAlmostEqual(np.sum(topic_dist[0]), 1.0, places=1)
    
    def test_transform_without_fit(self):
        """Test transforming without fitting first."""
        with self.assertRaises(ContentAnalysisError):
            self.topic_modeler.transform(["Machine learning is fascinating."])
    
    def test_get_topic_words(self):
        """Test getting top words for each topic."""
        self.topic_modeler.fit(self.test_texts)
        
        # Get top words
        topic_words = self.topic_modeler.get_topic_words(n_words=5)
        
        # Check that the result has expected structure
        self.assertEqual(len(topic_words), 3)  # 3 topics
        self.assertEqual(len(topic_words[0]), 5)  # 5 words per topic
        
        # Check that words are strings
        self.assertIsInstance(topic_words[0][0], str)
    
    def test_get_topic_words_without_fit(self):
        """Test getting topic words without fitting first."""
        with self.assertRaises(ContentAnalysisError):
            self.topic_modeler.get_topic_words()
    
    def test_get_dominant_topic(self):
        """Test getting dominant topic for a text."""
        self.topic_modeler.fit(self.test_texts)
        
        # Get dominant topic
        topic_id, probability = self.topic_modeler.get_dominant_topic(
            "Machine learning and artificial intelligence are related."
        )
        
        # Check that the result has expected types
        self.assertIsInstance(topic_id, (int, np.integer))
        self.assertIsInstance(probability, (float, np.floating))
        
        # Check that topic_id is within expected range
        self.assertGreaterEqual(topic_id, 0)
        self.assertLess(topic_id, 3)
        
        # Check that probability is within expected range
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
    
    def test_get_dominant_topic_without_fit(self):
        """Test getting dominant topic without fitting first."""
        with self.assertRaises(ContentAnalysisError):
            self.topic_modeler.get_dominant_topic("Machine learning is fascinating.")


class TestKeywordExtractor(unittest.TestCase):
    """Test cases for the KeywordExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.keyword_extractor = KeywordExtractor()
        self.test_text = """
        Machine learning is a subset of artificial intelligence that provides systems with the ability to learn.
        Natural language processing enables computers to understand human language.
        Computer vision deals with how computers can gain understanding from images.
        Deep learning is based on artificial neural networks.
        """
    
    def test_init(self):
        """Test initialization of KeywordExtractor."""
        self.assertEqual(self.keyword_extractor.method, 'tfidf')
        self.assertEqual(self.keyword_extractor.max_keywords, 10)
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "Machine Learning is amazing! AI and ML are great."
        processed = self.keyword_extractor._preprocess_text(text)
        
        # Check that text is lowercase and special characters are removed
        self.assertNotIn("!", processed)
        self.assertEqual(processed, "machine learning is amazing ai and ml are great")
    
    def test_extract_keywords_tfidf(self):
        """Test extracting keywords using TF-IDF."""
        keywords = self.keyword_extractor.extract_keywords(self.test_text)
        
        # Check that keywords are returned
        self.assertGreater(len(keywords), 0)
        self.assertLessEqual(len(keywords), 10)
        
        # Check that each keyword has the expected structure
        for keyword, score in keywords:
            self.assertIsInstance(keyword, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_extract_keywords_count(self):
        """Test extracting keywords using word count."""
        extractor = KeywordExtractor(method='count')
        keywords = extractor.extract_keywords(self.test_text)
        
        # Check that keywords are returned
        self.assertGreater(len(keywords), 0)
        self.assertLessEqual(len(keywords), 10)
        
        # Check that each keyword has the expected structure
        for keyword, score in keywords:
            self.assertIsInstance(keyword, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    @patch('src.ml.content_analysis.gensim_keywords')
    def test_extract_keywords_textrank(self, mock_gensim_keywords):
        """Test extracting keywords using TextRank."""
        # Mock the gensim_keywords function
        mock_gensim_keywords.return_value = ["learning", "artificial", "intelligence", "networks"]
        
        extractor = KeywordExtractor(method='textrank')
        keywords = extractor.extract_keywords(self.test_text)
        
        # Check that keywords are returned
        self.assertEqual(len(keywords), 4)
        
        # Check that each keyword has the expected structure
        for keyword, score in keywords:
            self.assertIsInstance(keyword, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_extract_keywords_invalid_method(self):
        """Test extracting keywords with invalid method."""
        extractor = KeywordExtractor(method='invalid')
        
        with self.assertRaises(ContentAnalysisError):
            extractor.extract_keywords(self.test_text)
    
    def test_extract_keywords_empty_text(self):
        """Test extracting keywords from empty text."""
        keywords = self.keyword_extractor.extract_keywords("")
        self.assertEqual(keywords, [])


class TestSummarizer(unittest.TestCase):
    """Test cases for the Summarizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.summarizer = Summarizer()
        self.test_text = """
        Machine learning is a subset of artificial intelligence that provides systems with the ability to learn.
        Natural language processing enables computers to understand human language.
        Computer vision deals with how computers can gain understanding from images.
        Deep learning is based on artificial neural networks.
        Reinforcement learning is about taking suitable actions to maximize reward in a particular situation.
        Supervised learning uses labeled training data to learn the mapping function from the input variables to the output variable.
        Unsupervised learning is used when you have only input data without labeled responses.
        """
    
    def test_init(self):
        """Test initialization of Summarizer."""
        self.assertEqual(self.summarizer.method, 'extractive')
        self.assertEqual(self.summarizer.ratio, 0.2)
    
    @patch('src.ml.content_analysis.gensim_summarize')
    def test_summarize_extractive(self, mock_gensim_summarize):
        """Test extractive summarization."""
        # Mock the gensim_summarize function
        mock_summary = "Machine learning is a subset of artificial intelligence that provides systems with the ability to learn."
        mock_gensim_summarize.return_value = mock_summary
        
        summary = self.summarizer.summarize(self.test_text)
        
        # Check that a summary is returned
        self.assertEqual(summary, mock_summary)
    
    @patch('src.ml.content_analysis.gensim_summarize')
    def test_gensim_fallback(self, mock_gensim_summarize):
        """Test fallback when gensim fails."""
        # Mock the gensim_summarize function to fail
        mock_gensim_summarize.return_value = ""
        
        summary = self.summarizer.summarize(self.test_text)
        
        # Check that a summary is returned (fallback method)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
    
    def test_simple_extractive_summarize(self):
        """Test simple extractive summarization."""
        summary = self.summarizer._simple_extractive_summarize(self.test_text)
        
        # Check that a summary is returned
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertLess(len(summary), len(self.test_text))
    
    def test_abstractive_summarize(self):
        """Test abstractive summarization (which falls back to extractive)."""
        summarizer = Summarizer(method='abstractive')
        
        # Should use extractive as fallback
        with patch.object(summarizer, '_extractive_summarize') as mock_extractive:
            mock_extractive.return_value = "This is a mock summary."
            summary = summarizer.summarize(self.test_text)
            mock_extractive.assert_called_once()
            self.assertEqual(summary, "This is a mock summary.")
    
    def test_summarize_invalid_method(self):
        """Test summarization with invalid method."""
        summarizer = Summarizer(method='invalid')
        
        with self.assertRaises(ContentAnalysisError):
            summarizer.summarize(self.test_text)
    
    def test_summarize_empty_text(self):
        """Test summarizing empty text."""
        summary = self.summarizer.summarize("")
        self.assertEqual(summary, "")


class TestImportantMomentDetector(unittest.TestCase):
    """Test cases for the ImportantMomentDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ImportantMomentDetector()
        self.test_segments = [
            {
                "text": "Machine learning is a subset of artificial intelligence.",
                "speaker": "Speaker1",
                "speaker_role": "expert",
                "start": 0.0,
                "end": 5.0
            },
            {
                "text": "Yes, that's correct.",
                "speaker": "Speaker2",
                "speaker_role": "host",
                "start": 5.0,
                "end": 6.0
            },
            {
                "text": "Deep learning is based on artificial neural networks and is revolutionizing many fields.",
                "speaker": "Speaker1",
                "speaker_role": "expert",
                "start": 6.0,
                "end": 12.0
            },
            {
                "text": "Could you explain how reinforcement learning works?",
                "speaker": "Speaker2",
                "speaker_role": "host",
                "start": 12.0,
                "end": 15.0
            }
        ]
        self.test_keywords = ["artificial intelligence", "deep learning", "neural networks", "reinforcement learning"]
    
    def test_init(self):
        """Test initialization of ImportantMomentDetector."""
        self.assertAlmostEqual(self.detector.keyword_weight, 0.4)
        self.assertAlmostEqual(self.detector.emotion_weight, 0.3)
        self.assertAlmostEqual(self.detector.speaker_weight, 0.2)
        self.assertAlmostEqual(self.detector.length_weight, 0.1)
        self.assertIsNotNone(self.detector.keyword_extractor)
    
    def test_detect_important_moments(self):
        """Test detecting important moments."""
        moments = self.detector.detect_important_moments(self.test_segments, self.test_keywords)
        
        # Check that moments are returned
        self.assertEqual(len(moments), 4)  # Same number as input segments
        
        # Check that moments have importance scores
        for moment in moments:
            self.assertIn('importance_score', moment)
            self.assertIsInstance(moment['importance_score'], float)
            self.assertGreaterEqual(moment['importance_score'], 0.0)
            self.assertLessEqual(moment['importance_score'], 1.0)
        
        # Check that moments are sorted by score (descending)
        scores = [moment['importance_score'] for moment in moments]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_detect_important_moments_with_extracted_keywords(self):
        """Test detecting important moments with automatically extracted keywords."""
        # Patch the keyword extraction method
        with patch.object(self.detector.keyword_extractor, 'extract_keywords') as mock_extract:
            mock_extract.return_value = [("intelligence", 0.9), ("learning", 0.8), ("artificial", 0.7)]
            
            moments = self.detector.detect_important_moments(self.test_segments)
            
            # Check that keyword extraction was called
            mock_extract.assert_called_once()
            
            # Check that moments are returned
            self.assertEqual(len(moments), 4)
    
    def test_detect_important_moments_empty_segments(self):
        """Test detecting important moments with empty segments."""
        moments = self.detector.detect_important_moments([])
        self.assertEqual(moments, [])
    
    def test_calculate_importance_score(self):
        """Test calculating importance score for a segment."""
        segment = self.test_segments[2]  # Deep learning segment
        score = self.detector._calculate_importance_score(segment, self.test_keywords)
        
        # Check that score is within expected range
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_keyword_score(self):
        """Test calculating keyword score."""
        text = "Deep learning is based on artificial neural networks."
        score = self.detector._calculate_keyword_score(text, self.test_keywords)
        
        # Check that score is within expected range
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Check that score is higher for text with more keywords
        text_with_more_keywords = "Deep learning and artificial intelligence use neural networks for reinforcement learning."
        higher_score = self.detector._calculate_keyword_score(text_with_more_keywords, self.test_keywords)
        self.assertGreater(higher_score, score)
    
    def test_calculate_speaker_score(self):
        """Test calculating speaker score."""
        # Test expert
        expert_segment = {"speaker_role": "expert"}
        expert_score = self.detector._calculate_speaker_score(expert_segment)
        self.assertAlmostEqual(expert_score, 0.8)
        
        # Test host/moderator
        host_segment = {"speaker_role": "host"}
        host_score = self.detector._calculate_speaker_score(host_segment)
        self.assertAlmostEqual(host_score, 1.0)
        
        # Test default
        default_segment = {}
        default_score = self.detector._calculate_speaker_score(default_segment)
        self.assertAlmostEqual(default_score, 0.5)
    
    def test_calculate_length_score(self):
        """Test calculating length score."""
        # Very short text
        short_text = "Hello."
        short_score = self.detector._calculate_length_score(short_text)
        self.assertAlmostEqual(short_score, 0.2)
        
        # Medium text
        medium_text = "This is a medium length text with about ten words."
        medium_score = self.detector._calculate_length_score(medium_text)
        self.assertAlmostEqual(medium_score, 0.6)
        
        # Ideal length text
        ideal_text = "This is an ideal length text with about twenty words. It should get the highest score because it's not too short or too long."
        ideal_score = self.detector._calculate_length_score(ideal_text)
        self.assertAlmostEqual(ideal_score, 1.0)
        
        # Very long text
        long_text = "This is a very long text " + "with many words " * 20
        long_score = self.detector._calculate_length_score(long_text)
        self.assertAlmostEqual(long_score, 0.7)


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for the SentimentAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.test_texts = {
            "positive": "This is fantastic! I'm really happy with the results and looking forward to more.",
            "neutral": "The meeting will take place at 10am. We will discuss the project status.",
            "negative": "This is terrible. I'm very disappointed with the poor quality and delays.",
            "mixed": "While there were some positive aspects, overall the experience was frustrating.",
            "empty": ""
        }
    
    def test_init(self):
        """Test initialization of SentimentAnalyzer."""
        self.assertIsNotNone(self.sentiment_analyzer.analyzer)
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis on different text types."""
        # Test positive text
        pos_result = self.sentiment_analyzer.analyze_sentiment(self.test_texts["positive"])
        self.assertIn("compound", pos_result)
        self.assertIn("pos", pos_result)
        self.assertIn("neu", pos_result)
        self.assertIn("neg", pos_result)
        self.assertGreater(pos_result["compound"], 0.5)
        self.assertGreater(pos_result["pos"], pos_result["neg"])
        
        # Test negative text
        neg_result = self.sentiment_analyzer.analyze_sentiment(self.test_texts["negative"])
        self.assertLess(neg_result["compound"], -0.5)
        self.assertGreater(neg_result["neg"], neg_result["pos"])
        
        # Test neutral text
        neu_result = self.sentiment_analyzer.analyze_sentiment(self.test_texts["neutral"])
        self.assertGreater(0.2, abs(neu_result["compound"]))
        self.assertGreater(neu_result["neu"], 0.5)
        
        # Test empty text
        empty_result = self.sentiment_analyzer.analyze_sentiment(self.test_texts["empty"])
        self.assertEqual(empty_result["compound"], 0.0)
        self.assertEqual(empty_result["pos"], 0.0)
        self.assertEqual(empty_result["neu"], 1.0)
        self.assertEqual(empty_result["neg"], 0.0)
    
    def test_get_sentiment_label(self):
        """Test getting sentiment labels from scores."""
        self.assertEqual(self.sentiment_analyzer.get_sentiment_label(-0.9), "Very Negative")
        self.assertEqual(self.sentiment_analyzer.get_sentiment_label(-0.5), "Negative")
        self.assertEqual(self.sentiment_analyzer.get_sentiment_label(0.0), "Neutral")
        self.assertEqual(self.sentiment_analyzer.get_sentiment_label(0.5), "Positive")
        self.assertEqual(self.sentiment_analyzer.get_sentiment_label(0.9), "Very Positive")
    
    def test_analyze_segment_sentiment(self):
        """Test sentiment analysis for transcript segments."""
        # Create test segments
        segments = [
            {"text": self.test_texts["positive"], "speaker": "Speaker A"},
            {"text": self.test_texts["negative"], "speaker": "Speaker B"},
            {"text": self.test_texts["neutral"], "speaker": "Speaker C"}
        ]
        
        # Analyze segments
        results = [self.sentiment_analyzer.analyze_segment_sentiment(segment) for segment in segments]
        
        # Check that results have the expected structure
        for result in results:
            self.assertIn("text", result)
            self.assertIn("speaker", result)
            self.assertIn("sentiment", result)
            self.assertIn("sentiment_label", result)
            self.assertIn("compound", result["sentiment"])
        
        # Check that sentiment labels match expected values
        self.assertIn("Positive", results[0]["sentiment_label"])
        self.assertIn("Negative", results[1]["sentiment_label"])
        
        # Check that original segment data is preserved
        self.assertEqual(results[0]["speaker"], "Speaker A")
        self.assertEqual(results[1]["speaker"], "Speaker B")
        self.assertEqual(results[2]["speaker"], "Speaker C")


class TestContentAnalyzer(unittest.TestCase):
    """Test cases for the ContentAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.content_analyzer = ContentAnalyzer()
        self.test_segments = [
            {
                "id": 1,
                "speaker": "Speaker A",
                "text": "We need to improve our machine learning algorithms.",
                "start_time": 0.0,
                "end_time": 3.0
            },
            {
                "id": 2,
                "speaker": "Speaker B",
                "text": "I agree. The current models are not performing well enough.",
                "start_time": 3.5,
                "end_time": 7.0
            },
            {
                "id": 3,
                "speaker": "Speaker A",
                "text": "Let's focus on optimizing the neural network architecture.",
                "start_time": 7.5,
                "end_time": 11.0
            },
            {
                "id": 4,
                "speaker": "Speaker B",
                "text": "Good idea. We should also collect more training data.",
                "start_time": 11.5,
                "end_time": 14.0
            }
        ]
    
    def test_init(self):
        """Test initialization of ContentAnalyzer."""
        self.assertIsNotNone(self.content_analyzer.topic_modeler)
        self.assertIsNotNone(self.content_analyzer.keyword_extractor)
        self.assertIsNotNone(self.content_analyzer.summarizer)
        self.assertIsNotNone(self.content_analyzer.moment_detector)
        self.assertIsNotNone(self.content_analyzer.sentiment_analyzer)
    
    def test_analyze_content(self):
        """Test analyzing content."""
        results = self.content_analyzer.analyze_content(self.test_segments)
        
        # Check that results have the expected structure
        self.assertIn("summary", results)
        self.assertIn("topics", results)
        self.assertIn("keywords", results)
        self.assertIn("important_moments", results)
        self.assertIn("sentiment_analysis", results)
        
        # Check sentiment analysis results
        sentiment_results = results["sentiment_analysis"]
        self.assertIn("overall", sentiment_results)
        self.assertIn("by_segment", sentiment_results)
        
        # Check overall sentiment structure
        overall = sentiment_results["overall"]
        self.assertIn("compound", overall)
        self.assertIn("pos", overall)
        self.assertIn("neu", overall)
        self.assertIn("neg", overall)
        self.assertIn("label", overall)
        
        # Check segment sentiment structure
        by_segment = sentiment_results["by_segment"]
        self.assertEqual(len(by_segment), len(self.test_segments))
        
        for segment in by_segment:
            self.assertIn("sentiment", segment)
            self.assertIn("sentiment_label", segment)
    
    def test_analyze_content_empty_segments(self):
        """Test analyzing content with empty segments."""
        results = self.content_analyzer.analyze_content([])
        
        self.assertEqual(results["summary"], "")
        self.assertEqual(results["topics"], [])
        self.assertEqual(results["keywords"], [])
        self.assertEqual(results["important_moments"], [])
        self.assertIn("sentiment_analysis", results)
    
    def test_analyze_content_error_handling(self):
        """Test error handling during content analysis."""
        # Create a mock that raises an exception
        with patch.object(self.content_analyzer.summarizer, 'summarize', side_effect=Exception("Test error")):
            with self.assertRaises(ContentAnalysisError):
                self.content_analyzer.analyze_content(self.test_segments)


if __name__ == '__main__':
    unittest.main() 